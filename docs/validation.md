# Validation

## Regression against the original analysis

The phase step of this package (`ata phase`; end-line rule of version 0.1 set by configuration, `end_line.chest_weight: 0`, `head_weight: 1`, `window_s: 5`) was run on archived
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

### End-line rule of version 0.2 (default)

Version 0.2 changes the default end line to the weighted rule of [method.md](method.md#4-end-line).
The same regression was repeated on archived coordinates of the original study with the original
analysis tables as reference (tolerance 1e-6 min):

| set | HMM | end-line rule | animals | max \|Δ head onset\| (min) | max \|Δ end line\| (min) |
|---|---|---|---|---|---|
| reference recordings | leave-one-out (17 recordings) | 0.25 chest + 0.75 head, 12 s, 0.15 (default) | 17 | 0 | 0 |
| reference recordings | leave-one-out (17 recordings) | head only, 5 s, 0.15 (configuration of version 0.1) | 17 | 0 | 0 |
| non-reference recordings (default model coordinates) | trained on all 17 | 0.25 chest + 0.75 head, 12 s, 0.15 (default) | 12 | 0 | 0 |

All differences are exactly zero. The weighted rule ends the last twitching later than the previous rule: on the 17 reference
recordings every end line is later, by 1.7-5.8 s (median 3.5 s); the range of the end line is
2.76-4.85 min with the previous rule and 2.81-4.95 min with the weighted rule.

**Choice of the weighted rule.** The weights, the 12-s window and the threshold were selected among
360 envelope candidates (signals: head, chest, sum, mean, 0.25 chest + 0.75 head, 0.75 chest +
0.25 head, max; windows 3, 5, 8, 12 s; thresholds 0.03-0.40) and 4 last-peak candidates by
agreement with last-twitching times marked by the investigator on video in 68 animals of six
batches: mean absolute difference 2.95 s (previous rule 4.37 s); with the rule selected on five batches
and tested on the sixth, 3.19 s. This is an exploratory choice on a small set; the rule is a
convention of the analysis, not a measurement of the true last twitch.

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
