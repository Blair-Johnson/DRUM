# DRUM Usage Examples

This document provides examples of using DRUM with different data formats and configurations.

## Example 1: Traditional TSV Format

Use the existing family dataset with TSV format:

```bash
python src/main.py \
  --datadir=datasets/family \
  --exps_dir=exps/ \
  --exp_name=family_demo \
  --max_epoch=10 \
  --rank=3
```

## Example 2: Prolog Format with All Predicates

Create a simple Prolog dataset and train:

**Directory structure:**
```
datasets/my_knowledge/
├── facts.pl    # Background knowledge
└── train.pl    # Target relations to learn
```

**facts.pl:**
```prolog
% Background knowledge about family relationships
father(john, mary).
father(john, bob).
mother(jane, mary).
mother(jane, bob).
father(bob, alice).
mother(sue, alice).
```

**train.pl:**
```prolog
% Target: learn grandfather relationship
grandfather(john, alice).
% Target: learn grandmother relationship  
grandmother(jane, alice).
```

**Train the model:**
```bash
python src/main.py \
  --datadir=datasets/my_knowledge \
  --exps_dir=exps/ \
  --exp_name=prolog_demo \
  --max_epoch=10
```

## Example 3: Restricted Domain (Prolog Format)

When you have many target relations but want to learn rules using only background predicates:

**facts.pl (small set of background predicates):**
```prolog
worksFor(alice, companyA).
worksFor(bob, companyB).
locatedIn(companyA, cityX).
locatedIn(companyB, cityY).
foundedBy(companyA, charlie).
```

**train.pl (many target relations):**
```prolog
% Target 1: employee location
employeeCity(alice, cityX).
employeeCity(bob, cityY).

% Target 2: founder relationships
employeeFounder(alice, charlie).

% Target 3: company affiliations
relatedCompanies(alice, bob).

% ... 600 more target relations ...
```

**Train with domain restriction:**
```bash
python src/main.py \
  --datadir=datasets/my_knowledge \
  --exps_dir=exps/ \
  --exp_name=restricted_demo \
  --restrict_domain \
  --max_epoch=10
```

The `--restrict_domain` flag ensures that learned rules only use predicates from `facts.pl` (worksFor, locatedIn, foundedBy), even though `train.pl` contains many other target predicates.

## Example 4: Training Without Test/Validation Sets

Sometimes you only have training data and want to explore:

**Directory with only training data:**
```
datasets/explore/
├── facts.pl    # Background knowledge
└── train.pl    # Training examples only
```

**Train without test/valid:**
```bash
python src/main.py \
  --datadir=datasets/explore \
  --exps_dir=exps/ \
  --exp_name=explore_demo \
  --max_epoch=20 \
  --min_epoch=20  # Disable early stopping
```

The system will automatically:
- Split 10% of training data for validation
- Skip test evaluation
- Save model checkpoints as usual

## Example 5: Extract Rules from Trained Model

After training, extract interpretable rules:

```bash
python src/main.py \
  --datadir=datasets/family \
  --exps_dir=exps/ \
  --exp_name=family_demo \
  --from_model_ckpt=exps/family_demo/ckpt/model-10 \
  --no_train \
  --extract_rules
```

This creates `exps/family_demo/extracted_rules.pl` with learned rules like:

```prolog
% Rules for query: grandfather
grandfather(X,Y) :- father(X,Z_1), father(Z_1,Y).

% Rules for query: grandmother  
grandmother(X,Y) :- mother(X,Z_1), mother(Z_1,Y).
```

## Example 6: Combining Multiple Options

Full example with all new features:

```bash
python src/main.py \
  --datadir=datasets/my_prolog_data \
  --exps_dir=exps/ \
  --exp_name=full_demo \
  --restrict_domain \
  --max_epoch=15 \
  --min_epoch=5 \
  --rank=3 \
  --learning_rate=0.001 \
  --batch_size=32
```

## Tips for Large-Scale Datasets

If you have 600+ target relations:

1. **Use restrict_domain**: Limits complexity by constraining the rule vocabulary
2. **Increase batch size**: Better GPU utilization (e.g., `--batch_size=128`)
3. **Adjust rank**: Start with lower rank (e.g., `--rank=2`) for faster training
4. **Monitor memory**: Watch GPU memory usage during training

For very large datasets, see [MULTI_TARGET_TRAINING.md](MULTI_TARGET_TRAINING.md) for strategies to batch target relations.

## Data Format Conversion

To convert TSV format to Prolog format:

**TSV to Prolog converter (Python script):**
```python
# convert_to_prolog.py
import sys

def convert_file(tsv_file, pl_file):
    with open(tsv_file) as f_in, open(pl_file, 'w') as f_out:
        f_out.write(f"% Converted from {tsv_file}\n")
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                entity1, relation, entity2 = parts
                f_out.write(f"{relation}({entity1}, {entity2}).\n")

convert_file('train.txt', 'train.pl')
convert_file('facts.txt', 'facts.pl')
```

Run: `python convert_to_prolog.py`
