# Scanpy quality-control seed

Scanpy `calculate_qc_metrics` computes observation- and variable-level quality-control metrics.
Typical observation summaries include total counts, detected genes, and percentages associated
with declared QC variables such as mitochondrial genes. Filtering thresholds remain scientific
decisions and must be inspected for each dataset rather than treated as universal defaults.

Source basis: Scanpy 1.11.5 official API documentation for `scanpy.pp.calculate_qc_metrics`.
