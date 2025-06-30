# Files to Remove for Public Release

This document lists files and directories that should be removed when preparing the repository for public release. **DO NOT DELETE these files yet** - this is just a tracking list.

## Documentation Files (Internal)
- `/scripts/activation_analysis/belief_regression_methods.md` - Internal methodology notes
- `/probing_methodology_neurips.md` - Internal methodology document
- `/experimental_methods.md` - Internal experimental methods (if not needed)

## Temporary and Cache Directories
- `/wandb/` - Weights & Biases run logs
- `/temp/` - Temporary files
- `/logs/` - Log files
- `/.egg-info/` - Python package info
- `/__pycache__/` - Python cache files (all occurrences)
- `/.pytest_cache/` - Pytest cache
- `/.ipynb_checkpoints/` - Jupyter notebook checkpoints (all occurrences)

## Old and Backup Files
- `/old_files/` - Outdated analysis scripts
- `/scripts/activation_analysis/backup_large_notebooks/` - Notebook backups
- Any files with `_old`, `_backup`, `_test` in the name

## Large Data Files
- `/data/` - Experiment data (consider keeping only essential checkpoints)
- `/analysis/` - Large analysis outputs (if any)
- `/process_data/` - Consider which process data files are essential

## Development and Debug Files
- `/scripts/activation_analysis/DEBUG_README.md`
- `/scripts/activation_analysis/README_drive.md`
- `/scripts/activation_analysis/README_precompute.md`
- Any debug notebooks or test scripts

## Dashboard and Visualization Development
- `/dashboard/` - If not part of public release
- `/qubit-viz/` - If not essential for reproducing paper results

## Configuration Files to Review
- Various experiment config variants that might be redundant
- Learning rate sweep configs if not essential

## Notebooks to Review
- Check which notebooks in `/scripts/activation_analysis/` are essential vs. exploratory
- Consider consolidating or removing redundant analysis notebooks

## Notes:
1. Keep all core implementation files
2. Keep essential configuration files
3. Keep figure generation scripts
4. Keep README.md files
5. Keep any data necessary to reproduce paper figures

## Before Removal Checklist:
- [ ] Backup entire repository
- [ ] Verify all essential functionality works without these files
- [ ] Update README.md to reflect removed content
- [ ] Ensure figure scripts are self-contained