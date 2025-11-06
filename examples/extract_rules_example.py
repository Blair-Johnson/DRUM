#!/usr/bin/env python
"""
Example: Extract rules from a trained DRUM model.

This script demonstrates how to programmatically extract rules from a trained
DRUM model using the rule extraction API.

Usage:
    python examples/extract_rules_example.py --model_path=exps/demo/ckpt/model-5 --datadir=datasets/family
"""

import os
import sys
import argparse
import tensorflow as tf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import Learner
from data import Data
from rule_extraction import extract_rules_from_model, save_rules_to_file


def parse_args():
    parser = argparse.ArgumentParser(description='Extract rules from trained DRUM model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (e.g., exps/demo/ckpt/model-5)')
    parser.add_argument('--datadir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for extracted rules (default: rules.pl)')
    parser.add_argument('--method', type=str, default='top_1',
                       help='Rule extraction method: top_1, top_k (default: top_1)')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='Minimum attention threshold for rules (default: 0.01)')
    parser.add_argument('--rank', type=int, default=3,
                       help='Model rank (default: 3)')
    parser.add_argument('--num_step', type=int, default=3,
                       help='Number of reasoning steps (default: 3)')
    parser.add_argument('--rnn_state_size', type=int, default=128,
                       help='RNN state size (default: 128)')
    parser.add_argument('--query_embed_size', type=int, default=128,
                       help='Query embedding size (default: 128)')
    parser.add_argument('--seed', type=int, default=33,
                       help='Random seed (default: 33)')
    return parser.parse_args()


class SimpleOption:
    """Minimal options for loading the model."""
    def __init__(self, args):
        self.seed = args.seed
        self.num_step = args.num_step
        self.rank = args.rank
        self.rnn_state_size = args.rnn_state_size
        self.query_embed_size = args.query_embed_size
        self.num_layer = 1
        self.no_norm = False
        self.thr = 1e-20
        self.dropout = 0.0
        self.learning_rate = 0.001
        self.accuracy = False
        self.top_k = 10
        self.query_is_language = False


def main():
    args = parse_args()
    
    print("="*60)
    print("DRUM Rule Extraction Example")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.datadir}")
    print(f"Method: {args.method}")
    print(f"Threshold: {args.threshold}")
    print()
    
    # Load data
    print("Loading data...")
    data = Data(args.datadir, args.seed, type_check=False, 
                domain_size=128, no_extra_facts=False)
    print(f"  Entities: {data.num_entity}")
    print(f"  Relations: {data.num_relation}")
    print()
    
    # Create model options
    option = SimpleOption(args)
    option.num_entity = data.num_entity
    option.num_operator = data.num_operator
    option.num_query = data.num_query
    
    # Build model
    print("Building model...")
    learner = Learner(option)
    print(f"  Rank: {option.rank}")
    print(f"  Steps: {option.num_step}")
    print()
    
    # Load model from checkpoint
    print("Loading model checkpoint...")
    tf.set_random_seed(option.seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False
    config.allow_soft_placement = True
    
    saver = tf.train.Saver()
    
    with tf.Session(config=config) as sess:
        saver.restore(sess, args.model_path)
        print(f"  Checkpoint restored from: {args.model_path}")
        print()
        
        # Extract rules
        print("Extracting rules...")
        rules_dict = extract_rules_from_model(
            sess=sess,
            learner=learner,
            data=data,
            queries=None,  # Extract for all queries
            method=args.method,
            rule_threshold=args.threshold
        )
        
        print(f"  Extracted rules for {len(rules_dict)} queries")
        total_rules = sum(len(rules) for rules in rules_dict.values())
        print(f"  Total rules: {total_rules}")
        print()
        
        # Print extracted rules
        print("Extracted Rules:")
        print("-"*60)
        for query_id, rules in sorted(rules_dict.items()):
            query_name = data.parser["query"][query_id]
            print(f"\n{query_name}:")
            for rule in rules:
                print(f"  {rule}")
        print()
        
        # Save to file
        if args.output is None:
            output_file = os.path.join(os.path.dirname(args.model_path), '..', 'extracted_rules.pl')
        else:
            output_file = args.output
        
        save_rules_to_file(rules_dict, output_file, data)
        print(f"Rules saved to: {output_file}")
        print()
    
    print("="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
