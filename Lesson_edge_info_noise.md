# Lesson: Edge Information Noise in v1.4

## Purpose

This lesson summarizes what we learned from v1.4 experiments about the trade-off between richer edge information and model robustness, and defines the practical design rule for the Infra2Graph prototype.

## One-Page Comparison Figure

![v1.4 edge information vs noise summary](output/v1_4_experiment_comparison/v1_4_edge_info_noise_summary.png)

Source metrics:
- `output/v1_4_experiment_comparison/v1_4_experiment_comparison.csv`

## Key Observations

1. Best overall configuration was Exp-4-lite (`kNN=3`, no edge_attr, weak quantile weighting).
2. Topology update (`graph snap` -> `kNN`) consistently improved global fit and/or Top-20 recall.
3. Adding distance-based edge attributes helped in some settings but was unstable across scales.
4. Strong quantile weighting degraded both R2 and Top-20 recall, indicating over-emphasis can distort learning.

## Result Snapshot

- Exp-4-lite: `R2=0.3516`, `MAE=3.1347`, `RMSE=6.0203`, `Top-20 Recall=0.70`
- Exp-2 kNN: `R2=0.2977`, `MAE=3.0773`, `RMSE=6.2655`, `Top-20 Recall=0.70`
- Exp-1 baseline: `R2=0.2078`, `MAE=3.3249`, `RMSE=6.6544`, `Top-20 Recall=0.65`

## Interpretation

- Edge topology quality is the first-order driver of performance.
- Edge attributes are second-order and sensitive to noise from distance scaling.
- For the prototype stage, robust retrieval of high-impact bridges is more important than maximizing one metric in isolation.

## Design Rule for Infra2Graph Prototype

1. Default to `kNN bridge-street edges (k=3)`.
2. Keep edge attributes optional and enable only with validated scaling.
3. Use weak quantile weighting to maintain high-impact sensitivity without recall collapse.
4. Evaluate with both global regression metrics (`R2`, `MAE`, `RMSE`) and operational metric (`Top-20 Recall`).

## Decision

v1.4 final reference for Infra2Graph prototype:
- **Exp-4-lite**
- Rationale: best balance between accuracy and high-risk bridge retrieval under edge-information noise.
