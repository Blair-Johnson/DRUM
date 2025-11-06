# DRUM Examples

This directory contains example scripts demonstrating how to use DRUM's rule extraction functionality.

## extract_rules_example.py

Demonstrates how to programmatically extract rules from a trained DRUM model.

### Prerequisites

1. Train a model first:
```bash
python src/main.py --datadir=datasets/family --exps_dir=exps/ --exp_name=demo --max_epoch=10
```

2. Extract rules from the trained model:
```bash
python examples/extract_rules_example.py \
    --model_path=exps/demo/ckpt/model-10 \
    --datadir=datasets/family \
    --threshold=0.01
```

### Output

The script will:
1. Load the trained model from the checkpoint
2. Extract rules using the specified method (default: top_1)
3. Print the extracted rules to console
4. Save rules to a Prolog file

Example output:
```
==============================================================
DRUM Rule Extraction Example
==============================================================
Model: exps/demo/ckpt/model-10
Dataset: datasets/family
Method: top_1
Threshold: 0.01

Loading data...
  Entities: 1234
  Relations: 12

Building model...
  Rank: 3
  Steps: 3

Loading model checkpoint...
  Checkpoint restored from: exps/demo/ckpt/model-10

Extracting rules...
  Extracted rules for 24 queries
  Total rules: 48

Extracted Rules:
------------------------------------------------------------

father:
  father(X,Y) :- father(X,Z), father(Z,Y).

grandfather:
  grandfather(X,Y) :- father(X,Z), father(Z,Y).

...

Rules saved to: exps/demo/extracted_rules.pl

==============================================================
Done!
==============================================================
```

## Customization

You can customize the extraction by modifying the script or passing different parameters:

- `--method`: Choose extraction method (top_1, top_k)
- `--threshold`: Set minimum attention threshold
- `--output`: Specify custom output file location
- Model parameters: `--rank`, `--num_step`, etc. (must match training configuration)

See the script's help for all options:
```bash
python examples/extract_rules_example.py --help
```
