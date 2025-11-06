"""
Rule extraction module for DRUM models.

This module provides functionality to extract interpretable Prolog-style rules
from trained DRUM models based on learned attention distributions.
"""

import numpy as np


class RuleExtractor:
    """
    Base class for rule extraction from DRUM models.
    
    This class provides a modular interface for extracting rules using
    different heuristics. Subclasses should implement the `extract` method
    to define specific extraction strategies.
    """
    
    def __init__(self, data, rule_threshold=1e-2):
        """
        Initialize the rule extractor.
        
        Args:
            data: Data object containing parser and relation information
            rule_threshold: Minimum attention threshold for including a rule
        """
        self.data = data
        self.rule_threshold = rule_threshold
        self.parser = data.parser
        
    def extract(self, attention_weights, query):
        """
        Extract rules from attention weights for a given query.
        
        Args:
            attention_weights: Attention distributions for each step and rank.
                              Shape: [rank, num_steps, num_operators]
            query: Query identifier (relation number)
            
        Returns:
            List of rule strings in Prolog format
        """
        raise NotImplementedError("Subclasses must implement extract method")
    
    def format_prolog_rule(self, head_relation, body_atoms):
        """
        Format a rule in Prolog syntax.
        
        Args:
            head_relation: The head predicate (target relation)
            body_atoms: List of body predicates/atoms
            
        Returns:
            String representing the rule in Prolog format
        """
        if not body_atoms:
            return f"{head_relation}(X,Y)."
        
        # Single atom: relation(X,Y) :- atom(X,Y).
        if len(body_atoms) == 1:
            atom = body_atoms[0]
            if atom.startswith('inv_'):
                return f"{head_relation}(X,Y) :- {atom[4:]}(Y,X)."
            else:
                return f"{head_relation}(X,Y) :- {atom}(X,Y)."
        
        # Multiple atoms: need intermediate variables
        # relation(X,Y) :- atom1(X,Z), atom2(Z,Y).  for 2 atoms
        # relation(X,Y) :- atom1(X,Z), atom2(Z,W), atom3(W,Y).  for 3 atoms
        # Variables: X, Z, W, V, U, ..., Y
        # That's: X, chr(90)=Z, chr(91)=W, chr(92)=V, ..., Y
        
        body_parts = []
        
        for i, atom in enumerate(body_atoms):
            # Current variable
            if i == 0:
                current_var = 'X'
            else:
                # Z, W, V, U, T, S, R, Q, P, O, ...
                # Start from 'Z' (ord 90)
                current_var = chr(90 + i - 1)
            
            # Next variable
            if i == len(body_atoms) - 1:
                next_var = 'Y'
            else:
                # Next intermediate: Z, W, V, ...
                next_var = chr(90 + i)
            
            # Handle inverse relations
            if atom.startswith('inv_'):
                # For inverse, swap the variables
                body_parts.append(f"{atom[4:]}({next_var},{current_var})")
            else:
                body_parts.append(f"{atom}({current_var},{next_var})")
        
        body_str = ', '.join(body_parts)
        return f"{head_relation}(X,Y) :- {body_str}."


class Top1RuleExtractor(RuleExtractor):
    """
    Extract rules using the top-1 (argmax) heuristic.
    
    This extractor selects the predicate with the highest attention weight
    at each step for each rank to form the rule body.
    """
    
    def extract(self, attention_weights, query):
        """
        Extract rules using argmax selection at each step.
        
        Args:
            attention_weights: Attention distributions for each step and rank.
                              Shape: [rank, num_steps, num_operators]
            query: Query identifier (relation number)
            
        Returns:
            List of rule strings in Prolog format
        """
        rules = []
        
        # Get the head relation name from the query
        if not self.data.query_is_language:
            head_relation = self.parser["query"][query]
        else:
            head_relation = self.parser["query"](query)
        
        rank = len(attention_weights)
        
        for r in range(rank):
            # Get attention for this rank
            rank_attentions = attention_weights[r]  # [num_steps, num_operators]
            
            # Extract body atoms by taking argmax at each step
            body_atoms = []
            max_attention_sum = 0.0
            non_selfloop_attention_sum = 0.0
            non_selfloop_count = 0
            
            for step_attention in rank_attentions:
                # Find the operator with maximum attention
                max_idx = np.argmax(step_attention)
                max_attention = step_attention[max_idx]
                max_attention_sum += max_attention
                
                # Skip the self-loop operator (last one)
                if max_idx == len(step_attention) - 1:
                    # Self-loop, don't add to body
                    continue
                
                # Track attention on non-self-loop predicates
                non_selfloop_attention_sum += max_attention
                non_selfloop_count += 1
                
                # Get operator name from parser
                if not self.data.query_is_language:
                    operator_name = self.parser["operator"][query][max_idx]
                else:
                    operator_name = self.parser["operator"][max_idx]
                
                body_atoms.append(operator_name)
            
            # Only include rules that meet the threshold
            # Must have at least one body atom and average attention on 
            # non-self-loop predicates should be above threshold
            if non_selfloop_count > 0 and len(rank_attentions) > 0:
                avg_attention = non_selfloop_attention_sum / non_selfloop_count
                if avg_attention >= self.rule_threshold:
                    rule_str = self.format_prolog_rule(head_relation, body_atoms)
                    rules.append(rule_str)
        
        return rules


class TopKRuleExtractor(RuleExtractor):
    """
    Extract rules using top-k selection at each step.
    
    This extractor selects the top-k predicates at each step and generates
    multiple rules by combining them.
    """
    
    def __init__(self, data, rule_threshold=1e-2, k=3):
        """
        Initialize the top-k rule extractor.
        
        Args:
            data: Data object containing parser and relation information
            rule_threshold: Minimum attention threshold for including a rule
            k: Number of top predicates to consider at each step
        """
        super().__init__(data, rule_threshold)
        self.k = k
    
    def extract(self, attention_weights, query):
        """
        Extract rules using top-k selection at each step.
        
        Currently returns top-1 as placeholder. Can be extended to generate
        multiple rules by exploring top-k paths.
        
        Args:
            attention_weights: Attention distributions for each step and rank.
            query: Query identifier
            
        Returns:
            List of rule strings in Prolog format
        """
        # For now, delegate to top-1 as a starting point
        # Future implementation can explore k-best paths
        top1_extractor = Top1RuleExtractor(self.data, self.rule_threshold)
        return top1_extractor.extract(attention_weights, query)


def extract_rules_from_model(sess, learner, data, queries=None, 
                             method='top_1', rule_threshold=1e-2, **kwargs):
    """
    Extract Prolog-formatted rules from a trained DRUM model.
    
    This is the main interface function for rule extraction. It retrieves
    attention weights from the model and converts them into interpretable
    rules using the specified extraction method.
    
    Args:
        sess: TensorFlow session with trained model
        learner: Trained DRUM Learner model instance
        data: Data object containing dataset and parser information
        queries: List of query identifiers to extract rules for.
                If None, uses all queries that appear in train/test.
        method: Extraction method to use ('top_1', 'top_k')
        rule_threshold: Minimum attention threshold for including a rule
        **kwargs: Additional parameters for specific extraction methods
                 (e.g., k for top_k)
    
    Returns:
        Dictionary mapping query identifiers to lists of Prolog rule strings.
        Format: {query_id: [rule_str1, rule_str2, ...], ...}
    
    Example:
        >>> rules = extract_rules_from_model(sess, learner, data, method='top_1')
        >>> for query, rule_list in rules.items():
        ...     print(f"Rules for {data.parser['query'][query]}:")
        ...     for rule in rule_list:
        ...         print(f"  {rule}")
    """
    # Use queries that appear in training/test if not specified
    # data.query_for_rules contains all relation IDs that appear in train/test data
    # including both regular and inverse relations (see data.py initialization)
    if queries is None:
        queries = data.query_for_rules
    
    # Select the appropriate extractor
    if method == 'top_1':
        extractor = Top1RuleExtractor(data, rule_threshold)
    elif method == 'top_k':
        k = kwargs.get('k', 3)
        extractor = TopKRuleExtractor(data, rule_threshold, k)
    else:
        raise ValueError(f"Unknown extraction method: {method}")
    
    # Extract rules for each query
    all_rules = {}
    
    for query in queries:
        # Get attention weights for this query from the model
        attention_weights = get_attention_weights_for_query(
            sess, learner, data, query
        )
        
        # Extract rules using the selected method
        rules = extractor.extract(attention_weights, query)
        
        if rules:  # Only include if we extracted any rules
            all_rules[query] = rules
    
    return all_rules


def get_attention_weights_for_query(sess, learner, data, query):
    """
    Retrieve attention weights for a specific query from the trained model.
    
    Args:
        sess: TensorFlow session
        learner: DRUM Learner model
        data: Data object
        query: Query identifier
        
    Returns:
        Attention weights as numpy array with shape [rank, num_steps, num_operators]
    """
    # Prepare query input based on data type
    if not data.query_is_language:
        queries = [query]
    else:
        queries = [query]
    
    # Create dummy data for running the model
    # Note: We only care about attention weights, not actual predictions
    hh = [0] * len(queries)
    tt = [0] * len(queries)
    
    # Create empty sparse matrix database (dummy values since we only fetch attention)
    # Each relation has forward and inverse, so we need num_operator // 2 matrices
    # (e.g., if num_operator=10, we have 5 base relations × 2 for forward/inverse = 10 operators)
    # Format: (indices, values, shape) for sparse tensor
    # Using minimal dummy values: single entry at (0,0) with value 0.0
    mdb = {r: ([(0, 0)], [0.], (data.num_entity, data.num_entity))
           for r in range(data.num_operator // 2)}
    
    # Fetch attention weights from the model
    # attention_operators_list is a list of rank elements, each containing
    # attention tensors for each step
    attention_weights_per_rank = []
    
    for i_rank in range(learner.rank):
        # Get attention for this rank
        to_fetch = learner.attention_operators_list[i_rank]
        
        # Run the graph to get attention values
        feed = {}
        if not data.query_is_language:
            # Repeat query for num_step-1 steps, then append END token (data.num_query)
            # This is the input format expected by the DRUM model
            feed[learner.queries] = [[q] * (learner.num_step - 1) + [data.num_query]
                                     for q in queries]
        else:
            # For language queries, repeat the query words and append END token
            feed[learner.queries] = [[q] * (learner.num_step - 1)
                                     + [[data.num_vocab] * data.num_word]
                                     for q in queries]
        
        feed[learner.heads] = hh
        feed[learner.tails] = tt
        # Populate database with dummy sparse tensors (we only need attention, not predictions)
        # Each entry: (indices_list, values_list, shape_tuple) for TensorFlow SparseTensor
        # Using single dummy entry [(0,0)] with value [0.] as placeholder
        for r in range(data.num_operator // 2):  # num_operator // 2 because of forward/inverse pairs
            feed[learner.database[r]] = ([(0, 0)], [0.], (data.num_entity, data.num_entity))
        
        # Fetch attention for all steps in this rank
        attention_this_rank = sess.run(to_fetch, feed)
        
        # attention_this_rank is a list of num_steps tensors
        # Each tensor contains splits for each operator
        # We need to concatenate them back
        step_attentions = []
        for step_splits in attention_this_rank:
            # step_splits is a list of tensors, one per operator
            # Each is shape [batch_size, 1], concatenate along last dimension
            step_attention = np.concatenate([s[0] for s in step_splits], axis=0)
            step_attentions.append(step_attention)
        
        attention_weights_per_rank.append(step_attentions)
    
    return attention_weights_per_rank


def save_rules_to_file(rules_dict, output_file, data):
    """
    Save extracted rules to a file.
    
    Args:
        rules_dict: Dictionary of query -> rules list
        output_file: Path to output file
        data: Data object for parsing query names
    """
    with open(output_file, 'w') as f:
        for query, rules in sorted(rules_dict.items()):
            # Write query header
            if not data.query_is_language:
                query_name = data.parser["query"][query]
            else:
                query_name = data.parser["query"](query)
            
            f.write(f"% Rules for query: {query_name}\n")
            
            # Write each rule
            for rule in rules:
                f.write(f"{rule}\n")
            
            f.write("\n")  # Blank line between queries
    
    return output_file
