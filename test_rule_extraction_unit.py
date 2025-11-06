#!/usr/bin/env python
"""
Unit tests for rule extraction module.

These tests verify the rule extraction logic without requiring a trained model.
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rule_extraction import Top1RuleExtractor, RuleExtractor


class MockData:
    """Mock data object for testing."""
    def __init__(self):
        self.query_is_language = False
        self.num_relation = 5
        self.num_operator = 10  # 2 * num_relation
        self.num_entity = 100
        
        # Create a simple parser
        self.parser = {
            "query": {
                0: "father",
                1: "mother", 
                2: "brother",
                3: "sister",
                4: "spouse",
                5: "inv_father",
                6: "inv_mother",
                7: "inv_brother",
                8: "inv_sister",
                9: "inv_spouse",
            },
            "operator": {
                0: {
                    0: "father", 1: "mother", 2: "brother", 3: "sister", 4: "spouse",
                    5: "inv_father", 6: "inv_mother", 7: "inv_brother", 
                    8: "inv_sister", 9: "inv_spouse"
                },
                1: {
                    0: "father", 1: "mother", 2: "brother", 3: "sister", 4: "spouse",
                    5: "inv_father", 6: "inv_mother", 7: "inv_brother", 
                    8: "inv_sister", 9: "inv_spouse"
                }
            }
        }


def test_prolog_formatting():
    """Test Prolog rule formatting."""
    print("Test 1: Prolog Rule Formatting")
    print("-" * 60)
    
    data = MockData()
    extractor = Top1RuleExtractor(data, rule_threshold=0.1)
    
    # Test simple rule
    rule1 = extractor.format_prolog_rule("grandfather", ["father", "father"])
    expected1 = "grandfather(X,Y) :- father(X,Z), father(Z,Y)."
    print(f"Simple rule: {rule1}")
    assert rule1 == expected1, f"Expected: {expected1}"
    
    # Test rule with inverse
    rule2 = extractor.format_prolog_rule("uncle", ["father", "inv_father"])
    expected2 = "uncle(X,Y) :- father(X,Z), father(Y,Z)."
    print(f"Rule with inverse: {rule2}")
    assert rule2 == expected2, f"Expected: {expected2}"
    
    # Test single atom rule
    rule3 = extractor.format_prolog_rule("parent", ["father"])
    expected3 = "parent(X,Y) :- father(X,Y)."
    print(f"Single atom rule: {rule3}")
    assert rule3 == expected3, f"Expected: {expected3}"
    
    # Test empty body
    rule4 = extractor.format_prolog_rule("entity", [])
    expected4 = "entity(X,Y)."
    print(f"Empty body rule: {rule4}")
    assert rule4 == expected4, f"Expected: {expected4}"
    
    print("✓ All Prolog formatting tests passed!\n")


def test_top1_extraction():
    """Test top-1 rule extraction logic."""
    print("Test 2: Top-1 Rule Extraction")
    print("-" * 60)
    
    data = MockData()
    extractor = Top1RuleExtractor(data, rule_threshold=0.3)
    
    # Create mock attention weights
    # Shape: [rank, num_steps, num_operators+1]
    rank = 2
    num_steps = 2
    num_operators = 11  # 10 operators + 1 self-loop
    
    # Rank 0: High confidence rule father -> father (grandfather)
    attention_rank0_step0 = np.zeros(num_operators)
    attention_rank0_step0[0] = 0.8  # father
    attention_rank0_step0[-1] = 0.2  # self-loop
    
    attention_rank0_step1 = np.zeros(num_operators)
    attention_rank0_step1[0] = 0.9  # father
    attention_rank0_step1[-1] = 0.1  # self-loop
    
    # Rank 1: Lower confidence rule (should be filtered by threshold)
    attention_rank1_step0 = np.zeros(num_operators)
    attention_rank1_step0[1] = 0.2  # mother (low confidence)
    attention_rank1_step0[-1] = 0.8  # self-loop
    
    attention_rank1_step1 = np.zeros(num_operators)
    attention_rank1_step1[2] = 0.1  # brother (low confidence)
    attention_rank1_step1[-1] = 0.9  # self-loop
    
    attention_weights = [
        [attention_rank0_step0, attention_rank0_step1],
        [attention_rank1_step0, attention_rank1_step1]
    ]
    
    # Extract rules for query 0 (father)
    rules = extractor.extract(attention_weights, query=0)
    
    print(f"Extracted {len(rules)} rules:")
    for rule in rules:
        print(f"  {rule}")
    
    # Should extract 1 rule (rank 0 meets threshold, rank 1 doesn't)
    assert len(rules) == 1, f"Expected 1 rule, got {len(rules)}"
    
    # The rule should be grandfather(X,Y) :- father(X,Z), father(Z,Y)
    expected_rule = "father(X,Y) :- father(X,Z), father(Z,Y)."
    assert rules[0] == expected_rule, f"Expected '{expected_rule}', got '{rules[0]}'"
    
    print("✓ Top-1 extraction test passed!\n")


def test_self_loop_handling():
    """Test that self-loop operators are correctly ignored."""
    print("Test 3: Self-Loop Handling")
    print("-" * 60)
    
    data = MockData()
    extractor = Top1RuleExtractor(data, rule_threshold=0.1)
    
    # Create attention weights where self-loop is selected at step 0
    rank = 1
    num_steps = 2
    num_operators = 11
    
    attention_step0 = np.zeros(num_operators)
    attention_step0[-1] = 0.9  # self-loop (should be ignored)
    attention_step0[0] = 0.1
    
    attention_step1 = np.zeros(num_operators)
    attention_step1[0] = 0.8  # father
    attention_step1[-1] = 0.2
    
    attention_weights = [[attention_step0, attention_step1]]
    
    rules = extractor.extract(attention_weights, query=0)
    
    print(f"Extracted {len(rules)} rules:")
    for rule in rules:
        print(f"  {rule}")
    
    # Should have one rule with only one atom (father) since self-loop is ignored
    assert len(rules) == 1
    expected_rule = "father(X,Y) :- father(X,Y)."
    assert rules[0] == expected_rule, f"Expected '{expected_rule}', got '{rules[0]}'"
    
    print("✓ Self-loop handling test passed!\n")


def test_inverse_relations():
    """Test handling of inverse relations."""
    print("Test 4: Inverse Relations")
    print("-" * 60)
    
    data = MockData()
    extractor = Top1RuleExtractor(data, rule_threshold=0.1)
    
    # Create attention weights for uncle rule: father -> inv_father
    # uncle(X,Y) means Y is uncle of X
    # This can be expressed as: father(X,Z), father(Y,Z)
    attention_step0 = np.zeros(11)
    attention_step0[0] = 0.8  # father
    
    attention_step1 = np.zeros(11)
    attention_step1[5] = 0.9  # inv_father
    
    attention_weights = [[attention_step0, attention_step1]]
    
    rules = extractor.extract(attention_weights, query=0)
    
    print(f"Extracted {len(rules)} rules:")
    for rule in rules:
        print(f"  {rule}")
    
    assert len(rules) == 1
    # The inv_father should swap variables
    expected_rule = "father(X,Y) :- father(X,Z), father(Y,Z)."
    assert rules[0] == expected_rule, f"Expected '{expected_rule}', got '{rules[0]}'"
    
    print("✓ Inverse relations test passed!\n")


def test_threshold_filtering():
    """Test that rules below threshold are filtered out."""
    print("Test 5: Threshold Filtering")
    print("-" * 60)
    
    data = MockData()
    
    # High threshold
    extractor_high = Top1RuleExtractor(data, rule_threshold=0.7)
    
    # Low confidence attention
    attention_step0 = np.zeros(11)
    attention_step0[0] = 0.6
    attention_step0[-1] = 0.4
    
    attention_step1 = np.zeros(11)
    attention_step1[1] = 0.5
    attention_step1[-1] = 0.5
    
    attention_weights = [[attention_step0, attention_step1]]
    
    rules_high = extractor_high.extract(attention_weights, query=0)
    print(f"With threshold 0.7: {len(rules_high)} rules")
    assert len(rules_high) == 0, "High threshold should filter out low confidence rules"
    
    # Low threshold
    extractor_low = Top1RuleExtractor(data, rule_threshold=0.3)
    rules_low = extractor_low.extract(attention_weights, query=0)
    print(f"With threshold 0.3: {len(rules_low)} rules")
    assert len(rules_low) == 1, "Low threshold should include moderate confidence rules"
    
    print("✓ Threshold filtering test passed!\n")


def run_all_tests():
    """Run all unit tests."""
    print("="*60)
    print("DRUM Rule Extraction Unit Tests")
    print("="*60)
    print()
    
    try:
        test_prolog_formatting()
        test_top1_extraction()
        test_self_loop_handling()
        test_inverse_relations()
        test_threshold_filtering()
        
        print("="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
