# DRUM: End-To-End Differentiable Rule Mining On Knowledge Graphs 

This is a fork of the implementation of DRUM, proposed in the following paper:

[DRUM: End-To-End Differentiable Rule Mining On Knowledge Graphs](https://papers.nips.cc/paper/9669-drum-end-to-end-differentiable-rule-mining-on-knowledge-graphs.pdf) -- [\[arXive + appendix\]](https://arxiv.org/abs/1911.00055).
Ali Sadeghian, Mohammadreza Armandpour, Patrick Ding, Daisy Zhe Wang.
NeurIPS 2019.

## Requirements
This implementation has been updated to Python3 from Python2. It is recommended to use [pixi](https://pixi.sh/dev/installation/) to construct a reproducible environment for running this project. After installing pixi, running
```
pixi shell
```
will put you into a new environment with the required dependencies. You can run `exit` to return to your original shell. You can read more about pixi [here](https://pixi.sh/dev/).

## Quick start
The following command starts training a dataset about family relations, and stores the experiment results in the folder `exps/demo/`.

```
python src/main.py --datadir=datasets/family --exps_dir=exps/ --exp_name=demo
```

To get the best performance, use different ranks for different datasets, default value is set to 3.

## Data Format Support

DRUM now supports two data formats:

### TSV Format (Original)
The traditional format uses tab-separated values with these files:
- `entities.txt`: List of entities (one per line)
- `relations.txt`: List of relations (one per line)
- `train.txt`: Training examples as `entity1\trelation\tentity2`
- `test.txt`: Test examples (optional)
- `valid.txt`: Validation examples (optional)
- `facts.txt`: Background knowledge facts

### Prolog Format (New)
You can now use Prolog-style fact files:
- `facts.pl`: Background knowledge in format `predicate(entity1, entity2).`
- `train.pl`: Training examples in format `predicate(entity1, entity2).`
- `test.pl`: Test examples (optional)
- `valid.pl`: Validation examples (optional)

The system automatically detects which format to use based on file extensions.

**Example Prolog format:**
```prolog
% facts.pl - Background knowledge
father(john, mary).
mother(jane, mary).

% train.pl - Target relations to learn
grandfather(john, alice).
```

**Training with Prolog format:**
```bash
python src/main.py --datadir=datasets/my_prolog_data --exps_dir=exps/ --exp_name=prolog_demo
```

### Restricting Domain to Background Predicates

When using Prolog format, you can restrict the learning domain to only use predicates that appear in `facts.pl`:

```bash
python src/main.py --datadir=datasets/my_data --exps_dir=exps/ --exp_name=restricted --restrict_domain
```

This is useful when:
- You have many target relations in `train.pl`
- You want to learn definitions using only a small set of background predicates from `facts.pl`
- For example: learn 600 target relations using only 10 background predicates

### Optional Validation and Test Sets

The system now handles missing validation and test sets gracefully:
- If no validation set is provided, one will be split from training data (10%)
- If no test set is provided, training will proceed without test evaluation
- This is useful for exploratory analysis with only training data

## Rule Extraction

After training a model, you can extract interpretable Prolog-style rules using the top-1 (argmax) extraction method:

```
python src/main.py --datadir=datasets/family --exps_dir=exps/ --exp_name=demo --extract_rules
```

This will create a file `exps/demo/extracted_rules.pl` containing rules like:

```prolog
% Rules for query: grandfather
grandfather(X,Y) :- father(X,Z_1), father(Z_1,Y).
```

Rules use variable names X (source), Y (target), and Z_1, Z_2, ... for intermediate entities. The top-1 method selects the predicate with highest attention weight at each reasoning step.

## Evaluation
To evaluate the prediction results, follow the steps below. The first two steps is preparation so that we can compute _filtered_ ranks (see [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) for details).

We use the experiment from Quick Start as an example. Change the folder names (datasets/family, exps/dev) for other experiments.
```
. eval/collect_all_facts.sh datasets/family
python eval/get_truths.py datasets/family
python eval/evaluate.py --preds=exps/demo/test_predictions.txt --truths=datasets/family/truths.pckl
```

## Training with Many Target Relations

For datasets with hundreds of target relations, see [MULTI_TARGET_TRAINING.md](MULTI_TARGET_TRAINING.md) for analysis and recommendations on memory-efficient training strategies.

This code partially is borrowed from [Neural LP](https://github.com/fanyangxyz/Neural-LP).
