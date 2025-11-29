/*
 * =============================================================================
 * Reward-Based Backward Pass
 * =============================================================================
 *
 * This module implements gradient computation for reward-based learning.
 *
 * Key Insight:
 * Instead of cross-entropy gradients (prob - 1 if correct, prob otherwise),
 * we use reward signals directly:
 *
 * - High reward (e.g., 0.8) → Increase predicted token logit
 * - Low reward (e.g., -0.5) → Decrease predicted token logit
 * - Target token logit adjusted inversely (if different from prediction)
 *
 * This is similar to REINFORCE algorithm in reinforcement learning.
 *
 * Advantages:
 * - No softmax in backward pass (faster)
 * - Direct reward propagation
 * - Semantically-aware gradients (via Venn reward)
 *
 * =============================================================================
 */

#ifndef REWARD_BACKWARD_CU
#define REWARD_BACKWARD_CU

/*
 * Reward-Based Output Projection Backward Kernel
 * ===============================================
 *
 * Computes gradients for output weights and hidden states using reward signals.
 *
 * Gradient Formula:
 * For each position:
 * 1. Find top prediction (argmax logits)
 * 2. grad_output[predicted] += reward * hidden_state
 * 3. grad_output[target] -= reward * hidden_state (if predicted != target)
 * 4. grad_hidden += reward * output_weights[predicted]
 *
 * Args:
 *   logits: [seq_len x vocab_size] output logits
 *   targets: [seq_len] target token IDs
 *   rewards: [seq_len] final reward signals
 *   word_valid_mask: [max_word_vocab] valid word tokens
 *   hidden_states: [seq_len x embed_dim] hidden states from forward pass
 *   output_weights: [vocab_size x embed_dim] output projection weights
 *   grad_output_weights: [vocab_size x embed_dim] gradient output
 *   grad_hidden: [seq_len x embed_dim] gradient output
 *   seq_len: sequence length
 *   embed_dim: embedding dimension
 *   vocab_size: total vocabulary size
 */
__global__ void reward_output_backward_kernel(
    const float* logits,
    const int* targets,
    const float* rewards,
    const int* word_valid_mask,
    const float* hidden_states,
    const float* output_weights,
    float* grad_output_weights,
    float* grad_hidden,
    int seq_len,
    int embed_dim,
    int vocab_size
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= seq_len - 1) return;  // Predict next token

    int target = targets[pos + 1];
    float reward = rewards[pos];

    if (target < 0 || target >= vocab_size) return;

    // Find top prediction
    float max_logit = -1e9f;
    int predicted = -1;

    for (int v = 0; v < vocab_size; v++) {
        bool is_valid = (v < CHAR_VOCAB_SIZE) ||
                       (word_valid_mask[v - CHAR_VOCAB_SIZE] == 1);

        if (!is_valid) continue;

        float logit = logits[pos * vocab_size + v];
        if (logit > max_logit) {
            max_logit = logit;
            predicted = v;
        }
    }

    if (predicted == -1) return;

    // Compute gradients
    // Higher reward → increase predicted token logit
    // Lower reward → decrease predicted token logit

    // Gradient for predicted token
    for (int d = 0; d < embed_dim; d++) {
        float grad = reward * hidden_states[pos * embed_dim + d];
        atomicAdd(&grad_output_weights[predicted * embed_dim + d], grad);
    }

    // If prediction was wrong, also gradient for target token
    // (encourage model to produce target instead)
    if (predicted != target) {
        for (int d = 0; d < embed_dim; d++) {
            // Negative gradient on predicted, positive on target
            // This shifts logits toward target
            float grad_target = -reward * hidden_states[pos * embed_dim + d];
            atomicAdd(&grad_output_weights[target * embed_dim + d], grad_target);
        }
    }

    // Gradient w.r.t. hidden states
    // Backpropagate reward signal through output weights
    for (int d = 0; d < embed_dim; d++) {
        float grad_h = reward * output_weights[predicted * embed_dim + d];

        if (predicted != target) {
            // Also add contribution from target token
            grad_h -= reward * output_weights[target * embed_dim + d];
        }

        atomicAdd(&grad_hidden[pos * embed_dim + d], grad_h);
    }
}

/*
 * Simplified Embedding Backward for Rewards
 * ==========================================
 *
 * Backpropagate gradients from hidden states to embeddings.
 * This part is unchanged from cross-entropy (simple pass-through).
 *
 * Args:
 *   tokens: [seq_len] token IDs
 *   grad_hidden: [seq_len x embed_dim] gradients from hidden states
 *   grad_char_embeddings: [char_vocab x embed_dim] gradient output
 *   grad_word_embeddings: [word_vocab x embed_dim] gradient output
 *   seq_len: sequence length
 *   embed_dim: embedding dimension
 */
__global__ void reward_embedding_backward_kernel(
    const int* tokens,
    const float* grad_hidden,
    float* grad_char_embeddings,
    float* grad_word_embeddings,
    int seq_len,
    int embed_dim
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= seq_len) return;

    int token = tokens[pos];

    if (token < 0) return;

    // Determine if char or word token
    if (token < CHAR_VOCAB_SIZE) {
        // Character token
        for (int d = 0; d < embed_dim; d++) {
            atomicAdd(&grad_char_embeddings[token * embed_dim + d],
                     grad_hidden[pos * embed_dim + d]);
        }
    } else {
        // Word token
        int word_id = token - CHAR_VOCAB_SIZE;
        if (word_id < MAX_WORD_VOCAB_SIZE) {
            for (int d = 0; d < embed_dim; d++) {
                atomicAdd(&grad_word_embeddings[word_id * embed_dim + d],
                         grad_hidden[pos * embed_dim + d]);
            }
        }
    }
}

#endif // REWARD_BACKWARD_CU
