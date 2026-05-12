# AeroSense: Aircraft-State-to-Flow Prediction

This repository provides a clean PyTorch implementation of **AeroSense**, a state-to-flow framework for short-term terminal-airspace traffic flow prediction.

AeroSense predicts future AP/AR traffic flow directly from the instantaneous set of aircraft states, without relying on historical flow look-back windows. Each sample is a variable-cardinality set of aircraft states at time `t`; the target is the future AP and AR traffic volume within `(t, t + horizon]`.

## Repository structure

```text
AeroSense
 ├─ dataloader.py              # Load one processed aircraft-state set data file
 ├─ model.py                   # AeroSense neural architecture
 ├─ module.py                  # Shared MLP, masked attention, pooling, decoders
 ├─ train.py                   # Training and validation from the single data file
 ├─ test.py                    # Evaluation from the single data file
 ├─ utils.py                   # Metrics, seed control, JSON and logging utilities
 ├─ requirements.txt           # Package requirements
 ├─ LICENSE                    # License file
 ├─ optDir
 │   └─ opt.json               # Default arguments
 ├─ data
 │   ├─ aerosense_demo_500.pkl # Single processed demo data file
 │   └─ README.md              # Processed data format description
 ├─ examples
 │   └─ make_toy_data.py       # Generate a single toy/demo data file
 └─ tools
     └─ export_processed_data.py
```

## Installation

```bash
pip install -r requirements.txt
```

Recommended environment:

- Python >= 3.8
- PyTorch >= 1.12
- NumPy
- pandas
- scikit-learn
- tqdm

## Processed demo data

This release uses one processed file:

```text
data/aerosense_demo_500.pkl
```

The scripts split this single file into train/validation/test subsets in memory. By default, the split ratios are:

```json
{
  "train_ratio": 0.8,
  "val_ratio": 0.1,
  "chronological_split": true
}
```

The included 500-sample data file is a **demo processed subset** for verifying the code pipeline. 

## Data format

The processed data file should be a pickle dictionary:

```python
{
    "X": [array_1, array_2, ...],      # each array: [num_aircraft_t, 18]
    "y": np.ndarray,                  # [num_samples, 2], target order: [AP, AR]
    "timestamps": [...],              # optional
    "feature_names": [...],           # recommended
    "metadata": {...}                 # optional
}
```

The model input is a variable-cardinality aircraft-state set. The default feature order is:

```text
0  latitude
1  longitude
2  height
3  speed
4  climbOrDescendSpeed
5  direction
6  dialSpeed
7  dialHeight
8  dist2area_AP
9  dist2area_AR
10 approach_factor_AP
11 approach_factor_AR
12 is_in_AP
13 is_in_AR
14 hour_sin
15 hour_cos
16 minute_sin
17 minute_cos
```

The target order is:

```text
y[:, 0] = future AP traffic flow
y[:, 1] = future AR traffic flow
```

## Quick start

Train on the included 500-sample demo file:

```bash
python train.py --opt optDir/opt.json
```

Evaluate the saved checkpoint:

```bash
python test.py --opt optDir/opt.json --checkpoint checkpoints/best_model.pt
```

The scripts write outputs to `checkpoints/`:

```text
checkpoints/best_model.pt
checkpoints/training_log.csv
checkpoints/test_metrics.json
checkpoints/test_predictions.csv
```

## Generate a toy single-file dataset

If you want to replace the included demo file with a synthetic toy file:

```bash
python examples/make_toy_data.py --output data/aerosense_demo_500.pkl --num_samples 500
```

## Citation

If you use this implementation, please cite our AeroSense paper:

```bibtex
@misc{wang2026aerosense,
  title={Unlocking air traffic flow prediction through microscopic aircraft-state modeling},
  author={Wang, Bin and Liu, Anqi and Zhao, Jiangtao and Huang, Yanyong and He, Peilan and Jiang, Guiyuan and Hong, Feng and Yu, Yanwei and Li, Tianrui},
  year={2026},
  eprint={2605.10083},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2605.10083},
  url={https://arxiv.org/abs/2605.10083}
}
```
## License

This project is released under the Apache License 2.0.
