import torch


def group_by_len(indices):
    d = {}
    for i, t in enumerate(indices):
        d.setdefault(len(t), []).append(i)
    return d


def find_duplicate_prefixes(nn_inputs):
    """
    Find duplicate prefixes in the input sequences and return their indices.
    
    Args:
        nn_inputs: Tensor of shape (batch_size, seq_len) containing token sequences
        
    Returns:
        Dictionary mapping each unique prefix tuple to a list of (seq_idx, pos) tuples
    """
    batch_size, seq_len = nn_inputs.shape
    prefix_to_indices = {}  # (prefix tuple) -> list of (seq_idx, pos) tuples
    
    # Process each sequence and position
    for seq_idx in range(batch_size):
        seq = nn_inputs[seq_idx]
        
        for pos in range(seq_len):
            # Get the prefix up to this position
            prefix = tuple(seq[:pos+1].cpu().numpy().tolist())
            
            # Add this occurrence to our mapping
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append((seq_idx, pos))
    
    return prefix_to_indices


def deduplicate_tensor(prefix_to_indices, tensor, aggregation_fn=None, debug=False, tolerance=1e-6):
    """
    Deduplicate a single tensor based on prefix indices.
    
    Args:
        prefix_to_indices: Dictionary mapping prefixes to lists of (seq_idx, pos) tuples
        tensor: Tensor of shape (batch_size, seq_len, ...) to deduplicate
        aggregation_fn: Function to aggregate duplicate values (default: take first occurrence)
                        Should accept a list of tensor values and return a single value
        debug: Whether to print debug information
        tolerance: Tolerance for tensor value differences
        
    Returns:
        Tuple of (unique_tensor_values, unique_prefixes)
    """
    # Dictionary to store unique tensor values
    unique_data = {}  # (prefix tuple) -> (tensor_value, count)
    
    # Debug information
    if debug:
        value_diffs = {}  # (prefix tuple) -> max difference observed
        inconsistent_prefixes = []
    
    # Process each unique prefix
    for prefix, indices_list in prefix_to_indices.items():
        first_seq_idx, first_pos = indices_list[0]
        value = tensor[first_seq_idx, first_pos]
        count = 1
        
        # If we have an aggregation function and multiple occurrences, prepare to aggregate
        if aggregation_fn is not None and len(indices_list) > 1:
            values = [value]
            
            # Collect all values for this prefix
            for seq_idx, pos in indices_list[1:]:
                current_value = tensor[seq_idx, pos]
                values.append(current_value)
                
                if debug:
                    # Check if values are the same
                    diff = torch.max(torch.abs(value - current_value)).item()
                    
                    if diff > tolerance:
                        if prefix not in inconsistent_prefixes:
                            inconsistent_prefixes.append(prefix)
                        
                        current_max_diff = value_diffs.get(prefix, 0)
                        value_diffs[prefix] = max(current_max_diff, diff)
                
                count += 1
            
            # Aggregate the values
            value = aggregation_fn(values)
        
        # Store the unique value
        unique_data[prefix] = (value, count)
    
    # Convert to tensor
    unique_prefixes = list(unique_data.keys())
    unique_values = torch.stack([unique_data[p][0] for p in unique_prefixes])
    
    return unique_values, unique_prefixes


def deduplicate_data(inputs, probs, beliefs):
    """
    Deduplicate inputs, probs, and beliefs based on duplicate prefixes.
    
    Args:
        inputs: Input tensor to find duplicate prefixes
        probs: Probability tensor to deduplicate
        beliefs: Belief tensor to deduplicate
        
    Returns:
        tuple: (deduplicated probs, deduplicated beliefs, deduplicated indices)
    """
    prefix_to_indices = find_duplicate_prefixes(inputs)
    dedup_probs, dedup_indices = deduplicate_tensor(prefix_to_indices, probs, aggregation_fn=sum)
    # normalize the probs to sum to 1
    dedup_probs = dedup_probs / dedup_probs.sum()
    dedup_beliefs, _ = deduplicate_tensor(prefix_to_indices, beliefs, aggregation_fn=None)
    return dedup_probs, dedup_beliefs, dedup_indices, prefix_to_indices
