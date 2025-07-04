# Figure Generation Commands

## Overview

All figure scripts now support CLI interfaces and can automatically download data from HuggingFace or use local data. The scripts use selective downloading to only fetch the required models instead of the entire dataset.

## Commands to Generate All Figures

```bash
# Generate Figure 2 (Belief Grid Visualization - 3 rows: Mess3, TomQA, Moon Process)
python Fig2.py --data-source huggingface

# Generate Figure 3 (Multi-Panel Analysis)
python Fig3.py --data-source huggingface

# Generate Figure 4 (Representational Similarity Analysis)
python Fig4.py --data-source huggingface

# Generate Appendix Figures (specific model types)
python FigAppendix.py --data-source huggingface --model-type Transformer
python FigAppendix.py --data-source huggingface --model-type LSTM
python FigAppendix.py --data-source huggingface --model-type GRU
python FigAppendix.py --data-source huggingface --model-type RNN

# Generate all appendix figures at once
python FigAppendix.py --data-source huggingface --model-type all

# Force re-download of data (if needed)
python Fig2.py --data-source huggingface --force-download
```

## Output Directory

All figures are saved to the `Figs/` directory with simple names:
- `Fig2.png` - Belief grid visualization
- `Fig3.png` - Multi-panel analysis  
- `Fig4.png` - Representational similarity analysis
- `FigAppendix_[ModelType].png` - Extended model comparisons

## Data Sources

- `--data-source huggingface`: Download from HuggingFace dataset (selective download)
- `--data-source local`: Use local data directory
- `--data-source auto`: Try local first, fallback to HuggingFace

## Selective Downloading

The scripts now use intelligent selective downloading that:
- Only fetches the specific model files needed for each figure (~30-50 files) instead of the entire dataset (~6500+ files)
- Dynamically discovers the latest checkpoint for each model by querying HuggingFace
- Downloads first, last, and some intermediate checkpoints for comprehensive coverage
- Significantly reduces download time and storage requirements

## Advanced Options

- `--force-download`: Force re-download of all files (useful if cache is corrupted)
- `--data-source local`: Use local data directory instead of HuggingFace
- `--data-source auto`: Try local first, fallback to HuggingFace if not found