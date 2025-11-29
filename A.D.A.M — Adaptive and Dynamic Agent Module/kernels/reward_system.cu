/*
 * =============================================================================
 * A.D.A.M Reward System - Replaces Cross-Entropy Loss
 * =============================================================================
 *
 * This module implements A.D.A.M's dual reward system for training:
 *
 * 1. Top-K Accuracy Reward:
 *    - Rewards predictions that include the target in top-K tokens
 *    - Vocab-size independent (no softmax over full vocabulary)
 *    - Graduated rewards: rank 1 = 1.0, rank 2 = 0.8, rank 3 = 0.6, etc.
 *    - Penalty for predictions outside top-K
 *
 * 2. Venn Semantic Similarity Reward:
 *    - Rewards semantically similar predictions
 *    - Uses cluster distances from Venn system
 *    - Captures "close enough" predictions (e.g., "car" vs "vehicle")
 *
 * Combined Reward:
 *    final_reward = α * topk_reward + (1-α) * venn_reward
 *    where α (REWARD_ALPHA) balances the two components
 *
 * Advantages over Cross-Entropy:
 * - No expensive softmax over 10k+ vocabulary
 * - Semantically aware (rewards similar tokens)
 * - More stable gradients (no log probabilities)
 * - Vocab-size independent scaling
 *
 * =============================================================================
 */

#ifndef REWARD_SYSTEM_CU
#define REWARD_SYSTEM_CU

// Runtime reward parameters (set from TrainingConfig)
static float reward_alpha = 0.7f;                    // Weight for Top-K reward
static int reward_top_k = 5;                         // Consider top-K predictions
static float reward_penalty_scale = 0.5f;            // Penalty scale for wrong predictions
static float reward_venn_similarity_threshold = 0.3f; // Min similarity for Venn reward

/*
 * Set reward system parameters from Python config
 */
extern "C" void set_reward_config(
    float alpha,
    int top_k,
    float penalty_scale,
    float venn_threshold
) {
    reward_alpha = alpha;
    reward_top_k = top_k;
    reward_penalty_scale = penalty_scale;
    reward_venn_similarity_threshold = venn_threshold;

    printf("✓ Reward config: alpha=%.2f, top_k=%d, penalty=%.2f, venn_thresh=%.2f\n",
           alpha, top_k, penalty_scale, venn_threshold);
}

/*
 * Top-K Accuracy Reward Kernel
 * =============================
 *
 * Computes reward based on whether target appears in top-K predictions.
 *
 * Reward structure:
 * - Rank 1 (exact match): 1.0
 * - Rank 2: 0.8
 * - Rank 3: 0.6
 * - Rank 4: 0.4
 * - Rank 5: 0.2
 * - Outside top-K: -penalty_scale
 *
 * This graduated reward encourages the model to rank correct tokens highly
 * even if they're not the absolute top prediction.
 *
 * Args:
 *   logits: [seq_len x vocab_size] raw output logits
 *   targets: [seq_len] target token IDs
 *   word_valid_mask: [max_word_vocab] mask for valid word tokens
 *   reward_out: [seq_len] output rewards per position
 *   topk_rank_out: [seq_len] output rank of target (for logging)
 *   seq_len: sequence length
 *   vocab_size: total vocabulary size
 *   top_k: number of top predictions to consider
 */
__global__ void compute_topk_reward_kernel(
    const float* logits,
    const int* targets,
    const int* word_valid_mask,
    float* reward_out,
    int* topk_rank_out,
    int seq_len,
    int vocab_size,
    int top_k
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= seq_len - 1) return;  // Predict next token

    int target = targets[pos + 1];

    if (target < 0 || target >= vocab_size) {
        reward_out[pos] = 0.0f;
        topk_rank_out[pos] = -1;
        return;
    }

    // Find top-K predictions using partial sort
    // We only need to track K highest values, not full sort

    float top_logits[10];  // Support up to top-10
    int top_indices[10];

    // Initialize with very low values
    for (int i = 0; i < top_k && i < 10; i++) {
        top_logits[i] = -1e9f;
        top_indices[i] = -1;
    }

    // Scan vocabulary and maintain top-K
    for (int v = 0; v < vocab_size; v++) {
        // Check if token is valid: char tokens always valid, word tokens check mask
        bool is_valid = (v < CHAR_VOCAB_SIZE) ||
                       (word_valid_mask[v - CHAR_VOCAB_SIZE] == 1);

        if (!is_valid) continue;

        float logit = logits[pos * vocab_size + v];

        // Insert into top-K if higher than current minimum
        for (int k = 0; k < top_k; k++) {
            if (logit > top_logits[k]) {
                // Shift lower ranks down
                for (int j = top_k - 1; j > k; j--) {
                    if (j < 10) {
                        top_logits[j] = top_logits[j - 1];
                        top_indices[j] = top_indices[j - 1];
                    }
                }
                // Insert new value
                top_logits[k] = logit;
                top_indices[k] = v;
                break;
            }
        }
    }

    // Find rank of target token
    int target_rank = -1;
    for (int k = 0; k < top_k; k++) {
        if (top_indices[k] == target) {
            target_rank = k;
            break;
        }
    }

    // Compute reward based on rank
    float reward;
    if (target_rank == -1) {
        // Target not in top-K: penalty
        reward = -reward_penalty_scale;
    } else {
        // Target in top-K: graduated reward
        // Rank 0 (1st) = 1.0, Rank 1 (2nd) = 0.8, Rank 2 (3rd) = 0.6, etc.
        reward = 1.0f - (target_rank * 0.2f);
        reward = fmaxf(reward, 0.0f);  // Clamp to [0, 1]
    }

    reward_out[pos] = reward;
    topk_rank_out[pos] = target_rank;
}

/*
 * Venn Semantic Similarity Reward Kernel
 * =======================================
 *
 * Computes reward based on semantic similarity between predicted and target tokens.
 * Uses cluster distances from the Venn semantic system.
 *
 * Intuition:
 * If the model predicts "vehicle" when the target is "car", this should be
 * rewarded (partial credit) since they're semantically similar.
 *
 * Algorithm:
 * 1. Find top prediction token
 * 2. Look up cluster memberships for prediction and target
 * 3. Compute cluster distance (average across all Venn heads)
 * 4. Reward = 1.0 - normalized_distance (closer = higher reward)
 *
 * Args:
 *   logits: [seq_len x vocab_size] raw output logits
 *   targets: [seq_len] target token IDs
 *   word_valid_mask: [max_word_vocab] mask for valid word tokens
 *   token_embeddings: [vocab_size x embed_dim] token embeddings
 *   cluster_centers: [num_heads x clusters_per_head x head_dim] Venn cluster centers
 *   reward_out: [seq_len] output Venn rewards
 *   similarity_out: [seq_len] output similarity scores (for logging)
 *   seq_len: sequence length
 *   vocab_size: total vocabulary size
 *   embed_dim: embedding dimension
 *   num_venn_heads: number of Venn heads
 *   clusters_per_head: clusters per head
 */
__global__ void compute_venn_reward_kernel(
    const float* logits,
    const int* targets,
    const int* word_valid_mask,
    const float* token_embeddings,
    const float* cluster_centers,
    float* reward_out,
    float* similarity_out,
    int seq_len,
    int vocab_size,
    int embed_dim,
    int num_venn_heads,
    int clusters_per_head
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= seq_len - 1) return;

    int target = targets[pos + 1];

    if (target < 0 || target >= vocab_size) {
        reward_out[pos] = 0.0f;
        similarity_out[pos] = 0.0f;
        return;
    }

    // Find top prediction
    float max_logit = -1e9f;
    int predicted_token = -1;

    for (int v = 0; v < vocab_size; v++) {
        bool is_valid = (v < CHAR_VOCAB_SIZE) ||
                       (word_valid_mask[v - CHAR_VOCAB_SIZE] == 1);

        if (!is_valid) continue;

        float logit = logits[pos * vocab_size + v];
        if (logit > max_logit) {
            max_logit = logit;
            predicted_token = v;
        }
    }

    if (predicted_token == -1) {
        reward_out[pos] = 0.0f;
        similarity_out[pos] = 0.0f;
        return;
    }

    // If exact match, give full reward
    if (predicted_token == target) {
        reward_out[pos] = 1.0f;
        similarity_out[pos] = 1.0f;
        return;
    }

    // Compute semantic similarity via cluster distances
    const int head_dim = embed_dim / num_venn_heads;

    float total_similarity = 0.0f;

    // Average similarity across all Venn heads
    for (int head = 0; head < num_venn_heads; head++) {
        // Get head-specific embeddings for prediction and target
        const float* pred_emb = token_embeddings + predicted_token * embed_dim + head * head_dim;
        const float* target_emb = token_embeddings + target * embed_dim + head * head_dim;

        // Find nearest clusters for both tokens
        int pred_cluster = -1;
        int target_cluster = -1;
        float pred_min_dist = 1e9f;
        float target_min_dist = 1e9f;

        const float* head_clusters = cluster_centers + head * clusters_per_head * head_dim;

        for (int c = 0; c < clusters_per_head; c++) {
            const float* cluster = head_clusters + c * head_dim;

            // Distance from prediction embedding to cluster
            float pred_dist = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float diff = pred_emb[d] - cluster[d];
                pred_dist += diff * diff;
            }

            if (pred_dist < pred_min_dist) {
                pred_min_dist = pred_dist;
                pred_cluster = c;
            }

            // Distance from target embedding to cluster
            float target_dist = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float diff = target_emb[d] - cluster[d];
                target_dist += diff * diff;
            }

            if (target_dist < target_min_dist) {
                target_min_dist = target_dist;
                target_cluster = c;
            }
        }

        // Compute cluster-level similarity
        float head_similarity;
        if (pred_cluster == target_cluster) {
            // Same cluster: high similarity
            head_similarity = 1.0f;
        } else {
            // Different clusters: compute distance between cluster centers
            const float* c1 = head_clusters + pred_cluster * head_dim;
            const float* c2 = head_clusters + target_cluster * head_dim;

            float cluster_dist = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float diff = c1[d] - c2[d];
                cluster_dist += diff * diff;
            }
            cluster_dist = sqrtf(cluster_dist);

            // Normalize distance to [0, 1] similarity
            // Assume max meaningful distance is ~2.0 in embedding space
            head_similarity = fmaxf(0.0f, 1.0f - cluster_dist / 2.0f);
        }

        total_similarity += head_similarity;
    }

    // Average across heads
    float avg_similarity = total_similarity / num_venn_heads;

    // Apply threshold
    float reward;
    if (avg_similarity < reward_venn_similarity_threshold) {
        // Too dissimilar: no reward
        reward = 0.0f;
    } else {
        // Scale similarity to reward [0, 1]
        reward = (avg_similarity - reward_venn_similarity_threshold) /
                 (1.0f - reward_venn_similarity_threshold);
        reward = fminf(reward, 1.0f);
    }

    reward_out[pos] = reward;
    similarity_out[pos] = avg_similarity;
}

/*
 * Combined Reward Kernel
 * =======================
 *
 * Combines Top-K and Venn rewards into final reward signal.
 *
 * Formula:
 *   final_reward = α * topk_reward + (1-α) * venn_reward
 *
 * Args:
 *   topk_rewards: [seq_len] Top-K rewards
 *   venn_rewards: [seq_len] Venn semantic rewards
 *   final_rewards: [seq_len] output combined rewards
 *   seq_len: sequence length
 *   alpha: weight for Top-K component (0.0-1.0)
 */
__global__ void combine_rewards_kernel(
    const float* topk_rewards,
    const float* venn_rewards,
    float* final_rewards,
    int seq_len,
    float alpha
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= seq_len - 1) return;

    float topk = topk_rewards[pos];
    float venn = venn_rewards[pos];

    // Weighted combination
    float final = alpha * topk + (1.0f - alpha) * venn;

    final_rewards[pos] = final;
}

/*
 * Reward Statistics Kernel
 * =========================
 *
 * Computes average reward across sequence (for logging).
 *
 * Uses warp-level reduction for efficiency.
 */
__global__ void compute_reward_stats_kernel(
    const float* rewards,
    float* avg_reward_out,
    int seq_len
) {
    __shared__ float shared_sum[256];

    int tid = threadIdx.x;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;

    if (pos < seq_len - 1) {
        local_sum = rewards[pos];
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(avg_reward_out, shared_sum[0] / (seq_len - 1));
    }
}

#endif // REWARD_SYSTEM_CU
