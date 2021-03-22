# S-pwROC
Temporal generalization of ROC curves for weakly labelled anomalies in time-series scenarios.

## Installation:

    The package can be installed as a local package with pip:
    ```
    pip install -e .
    ```

## Usage:
    evaluacion-cli filter <algorithm> [--header=header] [--format=format] [--filter_planned=filter_planned]
    evaluacion-cli roc_curve <algorithm> [--window_size=window_size] [--header=header] [--format=format] [--agg_method=agg_method] [--filter_planned=filter_planned]
    evaluacion-cli roc_surface <algorithm> [--num_windows=num_windows] [--header=header] [--format=format] [--agg_method=agg_method] [--filter_planned=filter_planned]
    evaluacion-cli open_surface <algorithm> [--agg_method=agg_method] [--filter_planned=filter_planned]
    evaluacion-cli summarise_surface <algorithm> [--agg_method=agg_method] [--filter_planned=filter_planned]
    evaluacion-cli (-h | --help)

## Options:
    <algorithm>                 Name of the folder with the results to evaluate.
    --window_size=window_size   Size of the window previous to a maintenance to be considered as an anomaly [default: 6].
    --header=header             Original files include the header [default: False].
    --format=format             Format of the results [default: us].
    --agg_method=agg_def        Aggregation method [default: mean].
    --filter_planned=filter_planned Filter planned maintenances according to internal code [default: False].
    --num_windows=num_windows   Num windows [default: -1].
    -h --help                   Show this screen.

## Considerations:
- `algorithm` results are suposed to be in a subfolder of a folder called `results/`.
- `format` options:
  - us: The data files contain two columns, unix (integer) and scores (double).
  - TLS: The data files contain three columns, Timestamp (integer), Label (double) and Score (double).
  - TL: The data files contain two columns, Timestamp (integer) and Score (double).
- `filter`:
  - Function to filter the instances that occur during a maintenance.
- `agg_method` options:
  - mean
  - median
  - NAB: Numenta Anomaly Benchmark weighting system.
- `num_windows`:
  - Default option `-1` sets the used windows to be: {1, 2, 3, 4, 5, 6, 12, 18, 24, 36, 48}
