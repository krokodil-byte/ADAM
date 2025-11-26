// VectLLM Vocabulary Management with Slot Allocation
// Handles dynamic vocabulary with hot/cold architecture and slot remapping

#ifndef VOCABULARY_CU
#define VOCABULARY_CU

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// SLOT ALLOCATION SYSTEM
// ============================================================================
//
// Problem: word_id can be arbitrarily large (256-∞) but hot vocab has
//          only MAX_WORD_VOCAB_SIZE (10k) slots.
//
// Solution: Maintain a mapping table: word_id → slot (0-9999)
//           - word_id_to_slot[word_id] = assigned slot or -1 if not in hot
//           - slot_to_word_id[slot] = word_id currently in that slot
//
// This allows unlimited cold vocab while keeping hot vocab fixed-size.
// ============================================================================

typedef struct {
    int* word_id_to_slot;     // [MAX_COLD_VOCAB] word_id → slot or -1
    int* slot_to_word_id;     // [MAX_WORD_VOCAB_SIZE] slot → word_id
    int* free_slots;          // [MAX_WORD_VOCAB_SIZE] stack of free slots
    int free_slot_count;      // Number of free slots available
    int max_cold_vocab;       // Maximum word_id supported (default: 100000)
} SlotTable;

// Initialize slot table
__host__ int init_slot_table(SlotTable* table, int max_cold_vocab) {
    table->max_cold_vocab = max_cold_vocab;
    table->free_slot_count = MAX_WORD_VOCAB_SIZE;

    // Allocate GPU memory for slot mappings
    cudaMalloc(&table->word_id_to_slot, max_cold_vocab * sizeof(int));
    cudaMalloc(&table->slot_to_word_id, MAX_WORD_VOCAB_SIZE * sizeof(int));
    cudaMalloc(&table->free_slots, MAX_WORD_VOCAB_SIZE * sizeof(int));

    // Initialize word_id_to_slot to -1 (not in hot)
    cudaMemset(table->word_id_to_slot, -1, max_cold_vocab * sizeof(int));

    // Initialize slot_to_word_id to -1 (empty slot)
    cudaMemset(table->slot_to_word_id, -1, MAX_WORD_VOCAB_SIZE * sizeof(int));

    // Initialize free_slots stack (0, 1, 2, ..., 9999)
    int* h_free_slots = (int*)malloc(MAX_WORD_VOCAB_SIZE * sizeof(int));
    for (int i = 0; i < MAX_WORD_VOCAB_SIZE; i++) {
        h_free_slots[i] = i;
    }
    cudaMemcpy(table->free_slots, h_free_slots,
               MAX_WORD_VOCAB_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    free(h_free_slots);

    return 0;
}

// Free slot table
__host__ void free_slot_table(SlotTable* table) {
    if (table->word_id_to_slot) cudaFree(table->word_id_to_slot);
    if (table->slot_to_word_id) cudaFree(table->slot_to_word_id);
    if (table->free_slots) cudaFree(table->free_slots);
}

// Allocate a slot for a word_id (called from host)
__host__ int allocate_slot(SlotTable* table, int word_id) {
    if (table->free_slot_count <= 0) {
        fprintf(stderr, "No free slots available\n");
        return -1;
    }

    // Pop slot from free stack
    table->free_slot_count--;
    int slot;
    cudaMemcpy(&slot, table->free_slots + table->free_slot_count,
               sizeof(int), cudaMemcpyDeviceToHost);

    // Update mappings
    cudaMemcpy(table->word_id_to_slot + word_id, &slot,
               sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(table->slot_to_word_id + slot, &word_id,
               sizeof(int), cudaMemcpyHostToDevice);

    return slot;
}

// Free a slot (called when evicting word)
__host__ int free_slot(SlotTable* table, int word_id) {
    // Get current slot for this word_id
    int slot;
    cudaMemcpy(&slot, table->word_id_to_slot + word_id,
               sizeof(int), cudaMemcpyDeviceToHost);

    if (slot < 0) {
        return -1;  // Word not in hot vocab
    }

    // Clear mappings
    int negative_one = -1;
    cudaMemcpy(table->word_id_to_slot + word_id, &negative_one,
               sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(table->slot_to_word_id + slot, &negative_one,
               sizeof(int), cudaMemcpyHostToDevice);

    // Push slot back to free stack
    cudaMemcpy(table->free_slots + table->free_slot_count, &slot,
               sizeof(int), cudaMemcpyHostToDevice);
    table->free_slot_count++;

    return 0;
}

// Get slot for word_id (device function - for use in kernels)
__device__ inline int get_slot(const int* word_id_to_slot, int word_id, int max_cold_vocab) {
    if (word_id < CHAR_VOCAB_SIZE || word_id >= CHAR_VOCAB_SIZE + max_cold_vocab) {
        return -1;
    }
    return word_id_to_slot[word_id - CHAR_VOCAB_SIZE];
}

// ============================================================================
// VOCABULARY ACTIVATION KERNELS
// ============================================================================

// Scatter embeddings to assigned slots
__global__ void scatter_embeddings_to_slots_kernel(
    float* word_embeddings,        // [MAX_WORD_VOCAB_SIZE x EMBED_DIM]
    const float* batch_embeddings, // [num_words x EMBED_DIM]
    const int* slots,              // [num_words] assigned slots
    int num_words,
    int embed_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_words * embed_dim;

    if (idx >= total_elements) return;

    int word_idx = idx / embed_dim;
    int dim_idx = idx % embed_dim;
    int slot = slots[word_idx];

    if (slot >= 0 && slot < MAX_WORD_VOCAB_SIZE) {
        word_embeddings[slot * embed_dim + dim_idx] = batch_embeddings[idx];
    }
}

// Set valid mask for assigned slots
__global__ void set_valid_mask_for_slots_kernel(
    int* word_valid_mask,    // [MAX_WORD_VOCAB_SIZE]
    const int* slots,        // [num_words]
    int num_words
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_words) return;

    int slot = slots[idx];
    if (slot >= 0 && slot < MAX_WORD_VOCAB_SIZE) {
        word_valid_mask[slot] = 1;
    }
}

// Token remapping kernel: Convert word_ids to slots for forward pass
__global__ void remap_tokens_to_slots_kernel(
    const int* input_tokens,       // [seq_len] original tokens
    int* remapped_tokens,          // [seq_len] output (char or slot+256)
    const int* word_id_to_slot,    // [MAX_COLD_VOCAB]
    int seq_len,
    int max_cold_vocab
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= seq_len) return;

    int token = input_tokens[pos];

    if (token < CHAR_VOCAB_SIZE) {
        // Character token - pass through
        remapped_tokens[pos] = token;
    } else {
        // Word token - map to slot
        int slot = get_slot(word_id_to_slot, token, max_cold_vocab);
        if (slot >= 0) {
            // Word is in hot vocab - use slot
            remapped_tokens[pos] = CHAR_VOCAB_SIZE + slot;
        } else {
            // Word not in hot vocab - fallback to char-level
            // (This shouldn't happen if pre-loading works correctly)
            remapped_tokens[pos] = '?';  // Fallback char
        }
    }
}

// ============================================================================
// BATCH VOCABULARY SYNC (with slot allocation)
// ============================================================================

__global__ void sync_vocabulary_kernel(
    const int* word_ids,            // [num_words] words to sync
    int* word_valid_mask,           // [MAX_WORD_VOCAB_SIZE] output
    const int* word_id_to_slot,     // [MAX_COLD_VOCAB] slot mapping
    int* activated_count,           // [1] output count
    int num_words,
    int max_cold_vocab
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_words) return;

    int word_id = word_ids[idx];
    if (word_id < CHAR_VOCAB_SIZE) return;

    // Get assigned slot
    int slot = get_slot(word_id_to_slot, word_id, max_cold_vocab);

    if (slot >= 0) {
        // Word is in hot vocab - mark as valid
        word_valid_mask[slot] = 1;
        atomicAdd(activated_count, 1);
    }
}

#endif // VOCABULARY_CU
