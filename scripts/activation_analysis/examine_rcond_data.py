import joblib
import os
import collections

def nested_dict_factory():
    """Returns a defaultdict that defaults to a regular dictionary."""
    return collections.defaultdict(dict)

# Load and examine the data structure
data_dir = "run_predictions_RCOND_FINAL/20241121152808_48"

# Check ground truth data
try:
    gt_data = joblib.load(os.path.join(data_dir, "ground_truth_data.joblib"))
    print("Ground truth data keys:", list(gt_data.keys()))
    print("  - probs shape:", gt_data['probs'].shape if 'probs' in gt_data else 'N/A')
    print("  - beliefs shape:", gt_data['beliefs'].shape if 'beliefs' in gt_data else 'N/A')
    print("  - indices length:", len(gt_data['indices']) if 'indices' in gt_data else 'N/A')
except Exception as e:
    print(f"Error loading ground truth data: {e}")

print("\n" + "="*50 + "\n")

# Check checkpoint data
try:
    ckpt_data = joblib.load(os.path.join(data_dir, "checkpoint_0.joblib"))
    print("Checkpoint data keys (layers):", list(ckpt_data.keys()))
    
    # Check one layer's data
    first_layer = list(ckpt_data.keys())[0]
    print(f"\nData for layer '{first_layer}':")
    layer_data = ckpt_data[first_layer]
    for key, value in layer_data.items():
        if hasattr(value, 'shape'):
            print(f"  - {key}: shape {value.shape}")
        elif isinstance(value, dict):
            print(f"  - {key}: dict with {len(value)} keys")
        elif isinstance(value, (list, tuple)):
            print(f"  - {key}: {type(value).__name__} with {len(value)} items")
        else:
            print(f"  - {key}: {type(value).__name__} = {value}")
except Exception as e:
    print(f"Error loading checkpoint data: {e}")