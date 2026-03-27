# Paper Benchmark Summary

- Benchmark: `synthetic_13methods`
- Profile: `smoke`
- Generated: `2026-03-18T05:32:38.971625+00:00`

## Synthetic Shortcut-Dim Recovery
```text
            method  precision  recall   f1
bias_direction_pca       0.70    0.70 0.70
demographic_parity        NaN     NaN  NaN
    equalized_odds        NaN     NaN  NaN
         frequency        NaN     NaN  NaN
               gce        NaN     NaN  NaN
         geometric       0.70    0.70 0.70
          groupdro        NaN     NaN  NaN
              hbac       0.50    0.50 0.50
             probe       0.75    0.75 0.75
               sis       0.55    0.55 0.55
               ssa        NaN     NaN  NaN
       statistical       0.80    0.80 0.80
```

## Synthetic False Positive Controls
- Mean FP rate when 1/12 methods flag: 0.0000
- Mean FP rate when >=11/12 methods flag: 0.0000
- Mean FP rate when 12/12 methods flag: 0.0000

## Artifacts
- Runs table: `output/paper_benchmark/synthetic_13methods/runs_paper.csv`
- Synthetic PR table: `output/paper_benchmark/synthetic_13methods/synthetic_dim_pr.csv`
- Synthetic FP table: `output/paper_benchmark/synthetic_13methods/synthetic_fp_control.csv`
- Correction control table: `output/paper_benchmark/synthetic_13methods/synthetic_correction_control.csv`
- CheXpert methods table: `output/paper_benchmark/synthetic_13methods/chexpert_method_results.csv`
- CheXpert convergence table: `output/paper_benchmark/synthetic_13methods/chexpert_convergence_matrix.csv`
- Baseline comparison table: `output/paper_benchmark/synthetic_13methods/external_baseline_comparison.csv`
- Figures directory: `output/paper_benchmark/synthetic_13methods/figures`