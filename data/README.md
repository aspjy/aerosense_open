# Demo Processed Data

This repository uses a **single processed demo data file**:

```text
data/aerosense_demo_500.pkl
```

The training and testing scripts split this single file into train/validation/test subsets in memory according to the ratios in `optDir/opt.json`.

## Pickle format

The file should contain a Python dictionary:

```python
{
    "X": [x_0, x_1, ...],      # list of arrays, x_i shape: [num_aircraft_i, 18]
    "y": y,                   # array shape: [num_samples, 2], order: [AP, AR]
    "timestamps": timestamps, # optional
    "feature_names": feature_names,
    "metadata": {...}         # optional
}
```

The code also accepts:

```python
{
    "samples": [(x_0, y_0), (x_1, y_1), ...],
    "timestamps": timestamps,
    "feature_names": feature_names
}
```

## Feature order

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

## Target order

```text
y[:, 0] = future AP traffic flow
y[:, 1] = future AR traffic flow
```

