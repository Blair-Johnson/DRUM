# Multi-Target Training Analysis and Recommendations

## Current System Analysis

### How Multi-Target Training Works

The DRUM system currently trains rules for ALL target relations simultaneously in a single model:

1. **Query Embeddings**: The model maintains a single set of query embeddings (`query_embedding_params`) that covers `num_query` embeddings where `num_query = 2 * num_relation` (forward and inverse relations). See `model.py` lines 67-70.

2. **Parallel Training**: Each batch can contain examples from different relations, and gradients are computed and applied for all relations simultaneously. The model learns embeddings for all queries in parallel through a shared RNN and attention mechanism.

3. **Batch Processing**: Training batches are created by `Data.next_train()` which can mix examples from different relations in each batch (unless `type_check=True`, in which case batches are grouped by relation).

4. **Memory Impact**: With 600 target relations:
   - Query embeddings: 1200 embeddings (600 * 2 for forward/inverse) of size `query_embed_size` (default 128)
   - Memory usage: ~614KB just for query embeddings (1200 * 128 * 4 bytes)
   - Plus RNN states, attention weights, and intermediate computations

### Key Observations

- **Confirmed**: All targets ARE optimized in parallel through shared parameters
- **Problem**: With 600 targets, the query embedding matrix alone needs 1200 rows
- **Issue**: GPU memory for intermediate computations (RNN states, attention matrices) grows with number of queries in batch
- **Constraint**: The current architecture has a single optimizer updating all parameters simultaneously

## Recommended Approach for Many Targets

### Option 1: Target Batching (Recommended)

Modify the training loop to process targets in smaller groups:

**Implementation Strategy:**
1. Add a `--target_batch_size` parameter (e.g., 50 targets at a time)
2. Modify `Data.__init__()` to:
   - Store full relation vocabulary
   - Create target groups (shards) based on `target_batch_size`
   - Track which target group is currently active
3. Modify `Data.next_train()` to:
   - Only return examples for relations in the current target group
   - Cycle through target groups across epochs
4. Modify `Model.__init__()` to:
   - Resize `query_embedding_params` based on active target group size
   - Or keep full size but mask unused queries
5. Training loop changes:
   - Train each target group for multiple epochs before moving to next
   - Save checkpoints per target group
   - Optionally: rotate through groups multiple times

**Pros:**
- Reduces memory footprint significantly
- Can scale to arbitrary number of targets
- Maintains single-model architecture
- Can share learned patterns across target groups

**Cons:**
- Longer total training time (sequential rather than parallel)
- More complex data handling
- Need to manage multiple checkpoints or combine models

**Code Changes Required:**
```python
# In data.py
class Data:
    def __init__(self, folder, ..., target_batch_size=None):
        # ... existing code ...
        if target_batch_size and target_batch_size < self.num_relation:
            self.target_batching = True
            self.target_batch_size = target_batch_size
            self.target_groups = self._create_target_groups()
            self.current_target_group = 0
        else:
            self.target_batching = False
    
    def _create_target_groups(self):
        """Divide relations into groups for batched training."""
        all_relations = list(range(self.num_relation))
        return [all_relations[i:i+self.target_batch_size] 
                for i in range(0, len(all_relations), self.target_batch_size)]
    
    def set_target_group(self, group_idx):
        """Switch to a different target group."""
        self.current_target_group = group_idx
        self.active_relations = set(self.target_groups[group_idx])
        # Filter train/test/valid to only include active relations
        self.train_active = [t for t in self.train if t[0] in self.active_relations]
        # Update num_train, num_batch_train, etc.

# In model.py - minimal changes needed
# The model naturally handles varying numbers of queries
# Just need to ensure query indices are remapped for current group

# In experiment.py
def train(self):
    if self.data.target_batching:
        for group_idx in range(len(self.data.target_groups)):
            self.data.set_target_group(group_idx)
            print(f"Training target group {group_idx+1}/{len(self.data.target_groups)}")
            # Standard training loop for this group
            self._train_target_group()
    else:
        # Original training loop
        self._train_standard()
```

### Option 2: Independent Models per Target

Train separate models for each target relation.

**Pros:**
- Simplest to implement (just run existing code multiple times)
- Perfectly parallel (can run on multiple GPUs)
- No architectural changes needed

**Cons:**
- Cannot share learned patterns across targets
- Need to manage 600 separate models
- Much more total computation (600x model training)

### Option 3: Relation Embedding + Meta-Learning

Learn a relation embedding space and train a single model that takes relation embeddings as input.

**Pros:**
- Most elegant solution
- Can generalize to unseen relations
- Shares knowledge across all targets

**Cons:**
- Requires significant architectural changes
- More complex to implement
- May need more training data per relation

## Recommendation

**For 600 targets, implement Option 1 (Target Batching)** with these parameters:
- `target_batch_size`: 50-100 targets per group
- Train each group for 5-10 epochs before switching
- Rotate through all groups 2-3 times
- Save best checkpoint per group

This provides the best balance of:
- Scalability (handles arbitrary number of targets)
- Memory efficiency (6-12x reduction in memory)
- Knowledge sharing (model still sees all data, just in groups)
- Implementation complexity (moderate changes to existing code)

## Estimated Changes Required

**Files to modify:**
1. `data.py`: ~100 lines (add target grouping logic)
2. `experiment.py`: ~50 lines (modify training loop for groups)
3. `main.py`: ~10 lines (add `--target_batch_size` flag)
4. `model.py`: ~20 lines (optional: optimize for current target group)

**Total**: ~180 lines of code changes (estimated)

**Testing strategy:**
1. Test with 2 groups of 6 relations each using family dataset
2. Verify memory usage reduction
3. Compare accuracy to non-batched training
4. Scale up to 600 targets

## Alternative: Lazy Evaluation

If implementing target batching is too complex, another option is to:
1. Keep the current architecture as-is
2. Let the user specify `--max_relations` parameter
3. Automatically subsample relations if the dataset has more than `max_relations`
4. Train multiple independent runs with different subsamples
5. Combine results at inference time

This is simpler but less elegant than Option 1.
