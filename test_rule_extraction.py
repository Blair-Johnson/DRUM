#!/usr/bin/env python
"""
Simple test script for rule extraction functionality.

This script trains a small DRUM model on the family dataset for a few epochs
and then extracts rules to verify the implementation works correctly.
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Add src to path for standalone test execution
# Note: For production use, consider proper package structure with __init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import Learner
from data import Data
from experiment import Experiment


class TestOption:
    """Minimal options for testing."""
    def __init__(self):
        self.seed = 33
        self.num_step = 3
        self.num_layer = 1
        self.rank = 2  # Use 2 ranks for testing
        self.rnn_state_size = 64
        self.query_embed_size = 64
        self.batch_size = 32
        self.print_per_batch = 1
        self.max_epoch = 2  # Just 2 epochs for testing
        self.min_epoch = 1
        self.learning_rate = 0.001
        self.no_norm = False
        self.thr = 1e-20
        self.dropout = 0.0
        self.accuracy = False
        self.top_k = 10
        self.rule_thr = 1e-2
        self.get_phead = False
        self.adv_rank = False
        self.rand_break = False
        self.query_is_language = False
        
    def save(self):
        pass


def test_rule_extraction():
    """Test the rule extraction functionality."""
    print("="*60)
    print("Testing DRUM Rule Extraction")
    print("="*60)
    
    # Setup paths
    datadir = 'datasets/family'
    exps_dir = 'exps'
    exp_name = 'test_rule_extraction'
    
    if not os.path.exists(datadir):
        print(f"Error: Dataset directory {datadir} not found")
        return False
    
    # Create experiment directory
    this_expsdir = os.path.join(exps_dir, exp_name)
    os.makedirs(this_expsdir, exist_ok=True)
    ckpt_dir = os.path.join(this_expsdir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Create options
    option = TestOption()
    option.this_expsdir = this_expsdir
    option.ckpt_dir = ckpt_dir
    option.model_path = os.path.join(ckpt_dir, "model")
    option.tag = exp_name
    
    # Load data
    print("\n1. Loading data...")
    data = Data(datadir, option.seed, type_check=False, 
                domain_size=128, no_extra_facts=False)
    print(f"   - Loaded {data.num_entity} entities")
    print(f"   - Loaded {data.num_relation} relations")
    print(f"   - Training samples: {data.num_train}")
    print(f"   - Test samples: {data.num_test}")
    
    # Set data-dependent options
    option.num_entity = data.num_entity
    option.num_operator = data.num_operator
    option.num_query = data.num_query
    
    # Build model
    print("\n2. Building model...")
    learner = Learner(option)
    print(f"   - Model built with rank={option.rank}, num_step={option.num_step}")
    
    # Create session and train
    print("\n3. Training model (2 epochs)...")
    tf.set_random_seed(option.seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False
    config.allow_soft_placement = True
    
    saver = tf.train.Saver()
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        data.reset(option.batch_size)
        experiment = Experiment(sess, saver, option, learner, data)
        
        # Train for a couple of epochs
        experiment.train()
        
        # Extract rules
        print("\n4. Extracting rules...")
        rules_dict = experiment.extract_rules(method='top_1')
        
        # Print extracted rules
        print("\n5. Extracted Rules:")
        print("="*60)
        total_rules = 0
        for query_id, rules in sorted(rules_dict.items()):
            query_name = data.parser["query"][query_id]
            print(f"\nQuery: {query_name}")
            for rule in rules:
                print(f"  {rule}")
                total_rules += 1
        
        print("\n" + "="*60)
        print(f"Total rules extracted: {total_rules}")
        print(f"Rules saved to: {os.path.join(this_expsdir, 'extracted_rules.pl')}")
        
        # Verify the rules file was created
        rules_file = os.path.join(this_expsdir, 'extracted_rules.pl')
        if os.path.exists(rules_file):
            print(f"\n✓ Rules file created successfully")
            with open(rules_file, 'r') as f:
                content = f.read()
                print(f"  File size: {len(content)} bytes")
                if total_rules > 0:
                    print(f"  Preview (first 500 chars):")
                    print("  " + "-"*56)
                    for line in content[:500].split('\n'):
                        print(f"  {line}")
        else:
            print(f"\n✗ Rules file not created")
            return False
        
        experiment.close_log_file()
    
    print("\n" + "="*60)
    print("✓ Rule extraction test completed successfully!")
    print("="*60)
    return True


if __name__ == "__main__":
    try:
        success = test_rule_extraction()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
