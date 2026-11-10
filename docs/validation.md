# Validation

## Regression against the original analysis

The phase step of this package (`ata phase`, default configuration) was run on archived
DeepLabCut coordinates of the original study and compared with the per-animal table of the
original analysis scripts (head onset and end line, tolerance 1e-6 min). The HMM was trained on
the same 17 reference recordings as in the original analysis. No coordinates or results are
part of the repository; the table lists only the differences.

| animal | checkpoint | HMM training | \|Δ head onset\| (min) | \|Δ end line\| (min) |
|---|---|---|---|---|
| animal 1 | default | 17 reference recordings | 0 | 0 |
| animal 2 | per-animal choice (posture-specific fine-tune) | 17 reference recordings | 0 | 0 |
| animal 3 | per-animal choice (earlier snapshot of another model) | 17 reference recordings | 0 | 0 |

The differences are exactly zero (identical floating-point values).

A wider check with the same settings covered every animal of that analysis: 38 animals, of
which 17 are the reference recordings themselves (leave-one-out models) and 9 use a per-animal
checkpoint. Maximum |Δ head onset| = 0 and maximum |Δ end line| = 0.

Software used for the regression: Python 3.11.18, numpy 2.4.6, scipy 1.17.1, pandas 3.1.1,
hmmlearn 0.3.3, scikit-learn 1.9.1. hmmlearn's EM is deterministic for fixed initial parameters
(`init_params=""`); other hmmlearn, numpy or scipy versions may change results in the last digits.

## Unit tests

`pytest` (synthetic data only) checks the step signals, normalisation and likelihood mask, peak
counting and binning, recovery of known chest / head boundaries by the HMM on synthetic
recordings, the end-line rule, model JSON round trip and determinism, leave-one-out handling,
the overwrite guard and run manifest, per-bin max downsampling of the stacked plot, the bar plot
source table, the Excel workbook (theme, chart XML order), the review and picker pages (valid
HTML, embedded JSON schema), the checkpoint-choice conversion and the configuration round trip.
