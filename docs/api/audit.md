# Audit API

The audit module implements the attribute-aware shortcut-potential audit used in the paper: a conditional probe as the single primary test, five secondary detectors, and a conditional permutation null (the attribute shuffled within task-label strata, between patients when patient ids are given). See [The paper's audit](../methods/overview.md#the-papers-audit) for the method description.

## Running an audit

```python
from shortcut_detect.audit import AuditConfig, conditional_audit

cfg = AuditConfig(n_permutations=200, n_permutations_probe=500, seed=42)
result = conditional_audit(E, A, Y, config=cfg, groups=patient_ids)
probe = result.results["probe"]
print(probe.statistic, probe.p_value, probe.p_bh, probe.flag)
```

## Class Reference

### AuditConfig

::: shortcut_detect.audit.conditional.AuditConfig
    options:
      show_root_heading: true
      show_source: true

### conditional_audit

::: shortcut_detect.audit.conditional.conditional_audit
    options:
      show_root_heading: true
      show_source: true

### AuditResult

::: shortcut_detect.audit.conditional.AuditResult
    options:
      show_root_heading: true
      show_source: true

### DetectorResult

::: shortcut_detect.audit.conditional.DetectorResult
    options:
      show_root_heading: true
      show_source: true

### conditional_permutations

::: shortcut_detect.audit.conditional.conditional_permutations
    options:
      show_root_heading: true
      show_source: true

### benjamini_hochberg

::: shortcut_detect.audit.conditional.benjamini_hochberg
    options:
      show_root_heading: true
      show_source: true
