# Rule Extraction for DRUM Models

This module provides functionality to extract interpretable Prolog-style rules from trained DRUM (Differentiable RUle Mining) models.

## Overview

DRUM learns differentiable attention weights over predicates at each reasoning step. The rule extraction module converts these learned attention distributions into symbolic logic rules that can be:
- Easily interpreted by humans
- Used for symbolic reasoning
- Analyzed for semantic understanding
- Exported to Prolog systems

## Quick Start

### Basic Usage

After training a DRUM model, you can extract rules by adding the `--extract_rules` flag:

```bash
python src/main.py \
    --datadir=datasets/family \
    --exps_dir=exps/ \
    --exp_name=my_experiment \
    --extract_rules \
    --rule_method=top_1
```

This will create a file `exps/my_experiment/extracted_rules.pl` containing the extracted rules in Prolog format.

### Extract Rules from Existing Model

To extract rules from an already-trained model without retraining:

```bash
python src/main.py \
    --datadir=datasets/family \
    --exps_dir=exps/ \
    --exp_name=my_experiment \
    --no_train \
    --no_preds \
    --from_model_ckpt=exps/previous_experiment/ckpt/model-10 \
    --extract_rules \
    --rule_method=top_1 \
    --rule_thr=0.01
```

## Rule Extraction Methods

### Top-1 (Argmax)

The `top_1` method is the simplest and most commonly used extraction heuristic:
- At each reasoning step, selects the predicate with the highest attention weight
- Filters out rules where average attention is below the threshold
- Ignores self-loop operators (identity relations)

**Parameters:**
- `--rule_method=top_1`: Activates top-1 extraction
- `--rule_thr=<float>`: Minimum average attention threshold (default: 0.01)

**Example extracted rule:**
```prolog
% Rules for query: grandfather
grandfather(X,Y) :- father(X,Z), father(Z,Y).
```

### Top-K (Planned)

The `top_k` method (extensible for future implementation) would:
- Consider multiple high-attention predicates at each step
- Generate multiple candidate rules
- Useful for exploring alternative reasoning paths

## Output Format

Extracted rules are saved in Prolog format with the following structure:

```prolog
% Rules for query: <relation_name>
<head>(X,Y) :- <body_atom_1>(X,Z), <body_atom_2>(Z,Y).
<head>(X,Y) :- <body_atom>(X,Y).

% Rules for query: <another_relation>
...
```

### Variable Convention

- `X`: Source entity (start of reasoning chain)
- `Y`: Target entity (end of reasoning chain)
- `Z, W, V, ...`: Intermediate entities (in order)

### Inverse Relations

Inverse relations are handled by swapping variable order:
```prolog
% Regular: father(X,Z) means Z is the father of X
% Inverse: father(Y,Z) means Z is the father of Y (in the context of Y being the sibling)
uncle(X,Y) :- father(X,Z), father(Y,Z).
```

## Programmatic API

You can also extract rules programmatically:

```python
from model import Learner
from data import Data
from rule_extraction import extract_rules_from_model

# Load your trained model and data
data = Data(datadir, seed, type_check, domain_size, no_extra_facts)
learner = Learner(option)

# Extract rules
with tf.Session() as sess:
    # ... restore model from checkpoint ...
    
    rules_dict = extract_rules_from_model(
        sess=sess,
        learner=learner,
        data=data,
        queries=None,  # Use all queries from train/test
        method='top_1',
        rule_threshold=0.01
    )
    
    # rules_dict is a dictionary: {query_id: [rule_str1, rule_str2, ...]}
    for query_id, rules in rules_dict.items():
        query_name = data.parser["query"][query_id]
        print(f"Rules for {query_name}:")
        for rule in rules:
            print(f"  {rule}")
```

## Architecture

The rule extraction system is designed to be modular and extensible:

### Core Components

1. **`RuleExtractor`** (Base Class)
   - Abstract base class defining the extraction interface
   - Handles Prolog formatting
   - Subclasses implement specific extraction heuristics

2. **`Top1RuleExtractor`**
   - Implements argmax (top-1) selection
   - Filters by attention threshold
   - Handles self-loops and inverse relations

3. **`extract_rules_from_model()`**
   - Main entry point for rule extraction
   - Retrieves attention weights from model
   - Delegates to appropriate extractor
   - Returns dictionary of rules

4. **`get_attention_weights_for_query()`**
   - Interfaces with TensorFlow model
   - Extracts attention distributions
   - Handles both standard and language-query modes

### Adding New Extraction Methods

To implement a new extraction method:

```python
from rule_extraction import RuleExtractor

class MyCustomExtractor(RuleExtractor):
    def extract(self, attention_weights, query):
        """
        Args:
            attention_weights: [rank, num_steps, num_operators]
            query: Query identifier
        
        Returns:
            List of Prolog rule strings
        """
        # Your custom logic here
        rules = []
        # ... extract rules ...
        return rules
```

Then use it:
```python
from rule_extraction import extract_rules_from_model

rules = extract_rules_from_model(
    sess, learner, data,
    method='my_custom',  # Will need to register in extract_rules_from_model
    rule_threshold=0.01
)
```

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--extract_rules` | flag | False | Enable rule extraction |
| `--rule_method` | str | 'top_1' | Extraction method ('top_1', 'top_k') |
| `--rule_thr` | float | 0.01 | Minimum attention threshold for rules |

## Examples

### Family Relations

Training and extracting rules from the family dataset:

```bash
python src/main.py \
    --datadir=datasets/family \
    --exps_dir=exps/ \
    --exp_name=family_rules \
    --max_epoch=10 \
    --rank=3 \
    --extract_rules
```

**Sample output (`exps/family_rules/extracted_rules.pl`):**
```prolog
% Rules for query: grandfather
grandfather(X,Y) :- father(X,Z), father(Z,Y).

% Rules for query: grandmother  
grandmother(X,Y) :- mother(X,Z), mother(Z,Y).

% Rules for query: uncle
uncle(X,Y) :- father(X,Z), brother(Z,Y).
```

### Knowledge Graph Completion

For larger datasets like FB15k-237:

```bash
python src/main.py \
    --datadir=datasets/fb15k-237 \
    --exps_dir=exps/ \
    --exp_name=fb15k_rules \
    --max_epoch=20 \
    --rank=5 \
    --num_step=4 \
    --extract_rules \
    --rule_thr=0.02
```

## Troubleshooting

### No rules extracted

**Possible causes:**
- Threshold too high: Try lowering `--rule_thr` (e.g., from 0.01 to 0.001)
- Model not trained enough: Increase `--max_epoch`
- Low rank: Increase `--rank` to learn more diverse rules

### Too many rules

**Possible causes:**
- Threshold too low: Increase `--rule_thr`
- Too many ranks: Decrease `--rank`

### Rules don't make semantic sense

**Possible causes:**
- Model needs more training epochs
- Learning rate issues
- Dataset quality problems

## Implementation Details

### Attention Weight Structure

DRUM models maintain attention distributions organized as:
```
attention_operators_list: List[rank] of List[num_steps] of Tensor[num_operators+1]
```

Each rank represents an independent reasoning chain. Each step within a rank attends over available operators (predicates). The final operator is a self-loop (identity).

### Threshold Computation

The threshold is computed as the average attention weight over non-self-loop predicates:

```python
avg_attention = sum(attention on real predicates) / count(real predicates)
if avg_attention >= rule_threshold:
    include_rule()
```

### Memory and Performance

Rule extraction is lightweight:
- Runs after training (no gradient computation)
- Processes one query at a time
- Memory usage: O(rank × num_steps × num_operators)
- Time complexity: O(num_queries × rank × num_steps)

Typical extraction time: < 1 second for small datasets, < 1 minute for large ones.

## References

For more information about the DRUM model:
- Paper: [DRUM: End-To-End Differentiable Rule Mining On Knowledge Graphs](https://arxiv.org/abs/1911.00055)
- NeurIPS 2019

## Testing

Run the unit tests to verify the implementation:

```bash
python test_rule_extraction_unit.py
```

This tests:
- Prolog rule formatting
- Top-1 extraction logic
- Self-loop handling
- Inverse relation handling
- Threshold filtering
