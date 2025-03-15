# Minimal Debugging Script for Activation Analysis

This folder contains a minimal debugging script (`minimal_debug.py`) structured as VS Code cells for interactive execution and debugging.

## Purpose

The minimal debugging script provides a simplified way to debug the activation analysis pipeline by:

1. Using VS Code cells for interactive execution
2. Hardcoding specific sweep, run, and checkpoint values
3. Running only the essential parts of the regression analysis
4. Outputting detailed logging information for debugging
5. Saving results to a dedicated debug folder

## Setup

Before running the script, make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Important: Simplified Configuration

This debug script uses a simplified configuration file (`minimal_config.py`) instead of the regular `config.py` to avoid a self-reference issue. The simplified config:

- Removes the class-based configuration that causes circular reference issues
- Sets sensible defaults for debugging (reduced dataset sizes, fewer iterations)
- Avoids the `AnalysisConfig` class that was causing the NameError

If you get a NameError related to AnalysisConfig, make sure you're using the imports from `minimal_config.py`, not from `config.py`.

## Using VS Code Cells

The script is divided into interactive cells, each marked with a `# %%` comment. In VS Code with the Python extension installed, you'll see a "Run Cell" button above each cell. This allows you to:

1. Run individual sections of code independently
2. Inspect variables between steps
3. Rerun specific parts without restarting the entire script
4. Experiment with different parameters without modifying the entire flow

To use the interactive cells:
- Install the VS Code Python extension if you haven't already
- Open the script in VS Code
- Look for the "Run Cell" button above each `# %%` marker
- Run cells sequentially, or selectively as needed for debugging

## Robust Error Handling

The script now includes robust error handling to help diagnose issues:

- Each major step is wrapped in try/except blocks
- Detailed error messages are logged
- If a step fails, the script creates dummy data to continue execution
- This allows you to debug later parts of the pipeline even if earlier parts fail

## Configuration

Edit the configuration parameters in the "Configure analysis parameters" cell:

```python
# Hardcoded values - modify these to match your specific model
sweep_id = "20250304052839"  # Replace with your specific sweep ID
run_id = "transformer4L1"    # Replace with your specific run ID
checkpoint = "checkpoint_10000.pt"  # Replace with your specific checkpoint
model_type = "transformer"   # Can be 'transformer' or 'rnn'
device = DEFAULT_DEVICE      # Uses CPU by default for reliable debugging
```

## Customization

- To use fewer regularization values for faster debugging, modify the `debug_rcond_values` parameter in the "Run regression analysis" cell.

- To use GPU acceleration on a CUDA-enabled machine, change the device in the configuration cell.

- To include classical beliefs (like Markov models), uncomment the code in the "Generate classical beliefs" cell.

- To visualize weights, uncomment the code in the "Explore and visualize weights" cell.

## Debugging Tips

1. Increase the logging verbosity in minimal_config.py:
   ```python
   LOG_LEVEL = logging.DEBUG
   ```

2. If you're getting memory errors, try using a smaller subset of the data by modifying the MSP data loading part.

3. To debug specific components, add more print statements or logging in the relevant cells.

4. You can add additional visualization cells to inspect different aspects of the data or results.

5. If you know a specific cell is failing, you can modify the dummy data creation in the except blocks to better simulate real data.

## Output Structure

The script will save results to:
- `analysis/minimal_debug/{run_id}_{checkpoint}_results.csv`

This CSV file contains the regression analysis results, sorted by variance explained (R²). 