# Paper Benchmark Summary

- Benchmark: `synthetic_10methods`
- Profile: `smoke`
- Generated: `2026-03-18T03:43:36.786903+00:00`

## Synthetic Shortcut-Dim Recovery
```text
            method  precision  recall   f1
bias_direction_pca       0.70    0.70 0.70
demographic_parity        NaN     NaN  NaN
    equalized_odds        NaN     NaN  NaN
         frequency        NaN     NaN  NaN
         geometric       0.70    0.70 0.70
              hbac       0.50    0.50 0.50
             probe       0.75    0.75 0.75
               sis       0.55    0.55 0.55
       statistical       0.80    0.80 0.80
```

## Synthetic False Positive Controls
- Mean FP rate when 1/9 methods flag: 0.0000
- Mean FP rate when >=8/9 methods flag: 0.0000
- Mean FP rate when 9/9 methods flag: 0.0000

## Artifacts
- Runs table: `output/paper_benchmark/synthetic_10methods/runs_paper.csv`
- Synthetic PR table: `output/paper_benchmark/synthetic_10methods/synthetic_dim_pr.csv`
- Synthetic FP table: `output/paper_benchmark/synthetic_10methods/synthetic_fp_control.csv`
- Correction control table: `output/paper_benchmark/synthetic_10methods/synthetic_correction_control.csv`
- CheXpert methods table: `output/paper_benchmark/synthetic_10methods/chexpert_method_results.csv`
- CheXpert convergence table: `output/paper_benchmark/synthetic_10methods/chexpert_convergence_matrix.csv`
- Baseline comparison table: `output/paper_benchmark/synthetic_10methods/external_baseline_comparison.csv`
- Figures directory: `output/paper_benchmark/synthetic_10methods/figures`