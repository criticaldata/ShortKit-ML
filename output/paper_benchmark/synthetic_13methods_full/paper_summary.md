# Paper Benchmark Summary

- Benchmark: `synthetic_13methods_full`
- Profile: `full`
- Generated: `2026-03-18T09:13:03.607535+00:00`

## Synthetic Shortcut-Dim Recovery
```text
            method  precision   recall       f1
bias_direction_pca   0.949333 0.949333 0.949333
demographic_parity        NaN      NaN      NaN
    equalized_odds        NaN      NaN      NaN
         frequency        NaN      NaN      NaN
               gce        NaN      NaN      NaN
         geometric   0.949333 0.949333 0.949333
          groupdro        NaN      NaN      NaN
              hbac   0.574889 0.574889 0.574889
             probe   0.927556 0.927556 0.927556
               sis   0.620222 0.620222 0.620222
               ssa        NaN      NaN      NaN
       statistical   0.945111 0.945111 0.945111
```

## Synthetic False Positive Controls
- Mean FP rate when 1/12 methods flag: 0.0000
- Mean FP rate when >=11/12 methods flag: 0.0000
- Mean FP rate when 12/12 methods flag: 0.0000

## Artifacts
- Runs table: `output/paper_benchmark/synthetic_13methods_full/runs_paper.csv`
- Synthetic PR table: `output/paper_benchmark/synthetic_13methods_full/synthetic_dim_pr.csv`
- Synthetic FP table: `output/paper_benchmark/synthetic_13methods_full/synthetic_fp_control.csv`
- Correction control table: `output/paper_benchmark/synthetic_13methods_full/synthetic_correction_control.csv`
- CheXpert methods table: `output/paper_benchmark/synthetic_13methods_full/chexpert_method_results.csv`
- CheXpert convergence table: `output/paper_benchmark/synthetic_13methods_full/chexpert_convergence_matrix.csv`
- Baseline comparison table: `output/paper_benchmark/synthetic_13methods_full/external_baseline_comparison.csv`
- Figures directory: `output/paper_benchmark/synthetic_13methods_full/figures`