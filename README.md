# DRUM: End-To-End Differentiable Rule Mining On Knowledge Graphs 

This is a fork of the implementation of DRUM, proposed in the following paper:

[DRUM: End-To-End Differentiable Rule Mining On Knowledge Graphs](https://papers.nips.cc/paper/9669-drum-end-to-end-differentiable-rule-mining-on-knowledge-graphs.pdf) -- [\[arXive + appendix\]](https://arxiv.org/abs/1911.00055).
Ali Sadeghian, Mohammadreza Armandpour, Patrick Ding, Daisy Zhe Wang.
NeurIPS 2019.

## Features

- **Differentiable Rule Learning**: End-to-end training of rule-based models on knowledge graphs
- **Rule Extraction**: Convert trained models into interpretable Prolog-style rules using top-1 (argmax) extraction
- **Flexible Architecture**: Configurable rank, steps, and attention mechanisms

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

Rules use variable names X (source), Y (target), and Z_1, Z_2, ... for intermediate entities. The top-1 method selects the predicate with highest attention weight at each reasoning step. You can adjust the confidence threshold with `--rule_thr` (default: 0.01).

## Evaluation
To evaluate the prediction results, follow the steps below. The first two steps is preparation so that we can compute _filtered_ ranks (see [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) for details).

We use the experiment from Quick Start as an example. Change the folder names (datasets/family, exps/dev) for other experiments.
```
. eval/collect_all_facts.sh datasets/family
python eval/get_truths.py datasets/family
python eval/evaluate.py --preds=exps/demo/test_predictions.txt --truths=datasets/family/truths.pckl
```

This code partially is borrowed from [Neural LP](https://github.com/fanyangxyz/Neural-LP).
