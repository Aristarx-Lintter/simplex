#!/usr/bin/env python3
"""
Dump model configurations to a text file for documentation.

This script extracts and dumps model configurations from S3 storage
for various transformer and LSTM experiments.
"""

from epsilon_transformers.analysis.load_data import S3ModelLoader
import json
import pprint


def dump_configs_to_file():
    """Extract and dump configurations to a text file."""
    
    # Initialize S3 loader
    s3_loader = S3ModelLoader(use_company_credentials=True)
    
    # Define all the sweep/run pairs from the regression analysis
    model_runs = [
        # Mess3
        ("20241121152808", "run_55_L4_H64_LSTM_uni_mess3", "LSTM - Mess3"),
        ("20241205175736", "run_23_L4_H4_DH16_DM64_mess3", "Transformer - Mess3"),
        ("20241121152808", "run_63_L4_H64_GRU_uni_mess3", "GRU - Mess3"),
        ("20241121152808", "run_71_L4_H64_RNN_uni_mess3", "RNN - Mess3"),

        # Fanizza (double circle)
        ("20241121152808", "run_53_L4_H64_LSTM_uni_fanizza", "LSTM - Fanizza"),
        ("20250422023003", "run_1_L4_H4_DH16_DM64_fanizza", "Transformer - Fanizza"),
        ("20241121152808", "run_61_L4_H64_GRU_uni_fanizza", "GRU - Fanizza"),
        ("20241121152808", "run_69_L4_H64_RNN_uni_fanizza", "RNN - Fanizza"),

        # Tom Quantum A
        ("20241121152808", "run_49_L4_H64_LSTM_uni_tom_quantum", "LSTM - Tom Quantum"),
        ("20241205175736", "run_17_L4_H4_DH16_DM64_tom_quantum", "Transformer - Tom Quantum"),
        ("20241121152808", "run_57_L4_H64_GRU_uni_tom_quantum", "GRU - Tom Quantum"),
        ("20241121152808", "run_65_L4_H64_RNN_uni_tom_quantum", "RNN - Tom Quantum"),
        
        # Post Quantum (moon)
        ("20241121152808", "run_48_L4_H64_LSTM_uni_post_quantum", "LSTM - Post Quantum"),
        ("20250421221507", "run_0_L4_H4_DH16_DM64_post_quantum", "Transformer - Post Quantum"),
        ("20241121152808", "run_56_L4_H64_GRU_uni_post_quantum", "GRU - Post Quantum"),
        ("20241121152808", "run_64_L4_H64_RNN_uni_post_quantum", "RNN - Post Quantum"),
    ]
    
    with open('model_configs_dump.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL CONFIGURATIONS DUMP\n")
        f.write("="*80 + "\n\n")
        
        for sweep_id, run_id, description in model_runs:
            f.write("\n" + "="*60 + "\n")
            f.write(f"{description}\n")
            f.write(f"Sweep: {sweep_id}, Run: {run_id}\n")
            f.write("="*60 + "\n\n")
            
            try:
                # Get checkpoints
                checkpoints = s3_loader.list_checkpoints(sweep_id, run_id)
                f.write(f"Total checkpoints: {len(checkpoints)}\n\n")
                
                if len(checkpoints) == 0:
                    f.write("ERROR: No checkpoints found\n\n")
                    continue
                
                # Try to load the first checkpoint to get configuration
                try:
                    model, run_config = s3_loader.load_checkpoint(
                        sweep_id, 
                        run_id, 
                        checkpoints[0],
                        device='cpu'
                    )
                    
                    # Pretty print the entire configuration
                    f.write("Full Configuration:\n")
                    f.write("-"*40 + "\n")
                    f.write(pprint.pformat(run_config, indent=2, width=120))
                    f.write("\n\n")
                    
                    # Also save as JSON for easier parsing
                    f.write("JSON Format:\n")
                    f.write("-"*40 + "\n")
                    f.write(json.dumps(run_config, indent=2))
                    f.write("\n\n")
                    
                except Exception as load_error:
                    # If model loading fails, try to get configs directly from S3
                    error_str = str(load_error)
                    if "attn_scale" in error_str or "unexpected keyword argument" in error_str:
                        f.write("NOTE: Model loading failed due to version compatibility (attn_scale), ")
                        f.write("attempting to load raw configuration files...\n\n")
                        
                        try:
                            # Try to load run configs directly
                            configs = s3_loader.load_run_configs(sweep_id, run_id)
                            
                            if configs.get('run_config'):
                                f.write("Run Configuration (from YAML):\n")
                                f.write("-"*40 + "\n")
                                f.write(pprint.pformat(configs['run_config'], indent=2, width=120))
                                f.write("\n\n")
                                
                                f.write("JSON Format:\n")
                                f.write("-"*40 + "\n")
                                f.write(json.dumps(configs['run_config'], indent=2))
                                f.write("\n\n")
                            else:
                                f.write("ERROR: Could not load run configuration\n\n")
                                
                        except Exception as config_error:
                            f.write(f"ERROR loading raw configs: {str(config_error)}\n\n")
                    else:
                        f.write(f"ERROR loading configuration: {str(load_error)}\n\n")
                
            except Exception as e:
                f.write(f"ERROR accessing checkpoints: {str(e)}\n\n")
    
    print("Configurations dumped to model_configs_dump.txt")


def dump_specific_configs(model_runs_list, output_file="custom_configs_dump.txt"):
    """
    Dump configurations for a specific list of model runs.
    
    Args:
        model_runs_list (list): List of tuples (sweep_id, run_id, description)
        output_file (str): Output filename
    """
    s3_loader = S3ModelLoader(use_company_credentials=True)
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CUSTOM MODEL CONFIGURATIONS DUMP\n")
        f.write("="*80 + "\n\n")
        
        for sweep_id, run_id, description in model_runs_list:
            f.write("\n" + "="*60 + "\n")
            f.write(f"{description}\n")
            f.write(f"Sweep: {sweep_id}, Run: {run_id}\n")
            f.write("="*60 + "\n\n")
            
            try:
                # Get checkpoints
                checkpoints = s3_loader.list_checkpoints(sweep_id, run_id)
                f.write(f"Total checkpoints: {len(checkpoints)}\n\n")
                
                # Load the first checkpoint to get configuration
                model, run_config = s3_loader.load_checkpoint(
                    sweep_id, 
                    run_id, 
                    checkpoints[0],
                    device='cpu'
                )
                
                # Pretty print the entire configuration
                f.write("Full Configuration:\n")
                f.write("-"*40 + "\n")
                f.write(pprint.pformat(run_config, indent=2, width=120))
                f.write("\n\n")
                
                # Also save as JSON for easier parsing
                f.write("JSON Format:\n")
                f.write("-"*40 + "\n")
                f.write(json.dumps(run_config, indent=2))
                f.write("\n\n")
                
            except Exception as e:
                f.write(f"ERROR loading configuration: {str(e)}\n\n")
    
    print(f"Configurations dumped to {output_file}")


if __name__ == "__main__":
    dump_configs_to_file() 