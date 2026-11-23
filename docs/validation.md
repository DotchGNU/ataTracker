# Validation

## Version 0.2: stored model, offset refinement and frame-rate check

`ata phase` (default configuration of version 0.2: stored HMM, offset refinement 0.25 chest +
0.75 head, 12 s, 0.15, frame-rate check required) was run on archived DeepLabCut coordinates of
three cohorts of the original study and compared with the per-animal values of the original
analysis scripts, which load the same stored model. The model file was the original store with
the real recording names (`phase.model_file` set to it), so the training recordings were
segmented with their stored leave-one-out models. Values were compared in the unit of the
package table (min); "0" means identical floating-point values. No coordinates or results are
part of the repository.

| cohort | animals | HMM | coordinates | compared with | values | max \|Δ\| (min) |
|---|---|---|---|---|---|---|
| A (the training cohort) | 17 | stored leave-one-out models (17 distinct models) | one checkpoint for all | HMM state edges of the original scripts | chest onset, head onset, post onset, recording length | 0 |
| A | 17 | as above | as above | per-animal table of the original scripts | head onset, offset | 0 |
| B | 8 | stored full model | per-animal checkpoints (4 checkpoints) | recomputation with the stored model; cohort master workbook | head onset, post onset, offset | 0 |
| C | 10 | stored full model | per-animal checkpoints (4 checkpoints) | as for B | head onset, post onset, offset | 0 |

Further checks on the same data:

* The packaged model (`builtin`, anonymous training IDs) gives the same values as the original
  store for cohort C (max |Δ| 0, same parameter hash).
* `--refit` on cohort A (17 leave-one-out refits) gives the same values as the stored models
  (max |Δ| 0, same parameter hashes).
* `ata hmm fit` on the 17 training recordings writes a full model and 17 leave-one-out models
  whose parameter hashes equal those of the original store (18 of 18), with the same keys,
  configuration and fit settings.
* A changed peak threshold (0.4) or a missing model file stops `ata phase` with an error before
  any recording is analysed.

**Frame-rate check on the original videos.** Original recordings and DeepLabCut input videos of
the three cohorts passed (26, 16 and 20 videos; `r_frame_rate` 30000/1001 or 2997/100, measured
29.97 frames/s). The check failed, as intended, on:

| videos | container `r_frame_rate` | `avg_frame_rate` / measured | result |
|---|---|---|---|
| re-encoded copies of cohort B (8) | 30000/1001 | 24.00 frames/s (variable frame interval) | FAIL |
| re-encoded copies of another cohort (13) | 24/1 | 24.00 frames/s | FAIL |
| recordings of a manual-observation set (12 of 16) | 60000/1001 | 59.94 frames/s | FAIL |

The videos of cohort B that DeepLabCut analysed for this regression are 480 x 272 crops made from
the re-encoded copies; their container and measured rates are 29.97 frames/s, so the check passes
on them. The check reads the files it is given; listing every intermediate copy in
`fps_check.dlc_input_dirs` is what exposes a rate change upstream of the crop.

## Version 0.1: regression against the original analysis

The phase step of this package (`ata phase` with `phase.refit: true`; offset rule of version 0.1 set by configuration, `offset_refinement.chest_weight: 0`, `head_weight: 1`, `window_s: 5`) was run on archived
DeepLabCut coordinates of the original study and compared with the per-animal table of the
original analysis scripts (head onset and offset, tolerance 1e-6 min). The HMM was trained on
the same 17 reference recordings as in the original analysis. No coordinates or results are
part of the repository; the table lists only the differences.

| animal | checkpoint | HMM training | \|Δ head onset\| (min) | \|Δ offset\| (min) |
|---|---|---|---|---|
| animal 1 | default | 17 reference recordings | 0 | 0 |
| animal 2 | per-animal choice (posture-specific fine-tune) | 17 reference recordings | 0 | 0 |
| animal 3 | per-animal choice (earlier snapshot of another model) | 17 reference recordings | 0 | 0 |

The differences are exactly zero (identical floating-point values).

A wider check with the same settings covered every animal of that analysis: 38 animals, of
which 17 are the reference recordings themselves (leave-one-out models) and 9 use a per-animal
checkpoint. Maximum |Δ head onset| = 0 and maximum |Δ offset| = 0.

### Offset refinement rule of version 0.2 (default)

Version 0.2 changes the default offset rule to the weighted rule of [method.md](method.md#4-offset-refinement) (models refitted, `phase.refit: true`).
The same regression was repeated on archived coordinates of the original study with the original
analysis tables as reference (tolerance 1e-6 min):

| set | HMM | offset rule | animals | max \|Δ head onset\| (min) | max \|Δ offset\| (min) |
|---|---|---|---|---|---|
| reference recordings | leave-one-out (17 recordings) | 0.25 chest + 0.75 head, 12 s, 0.15 (default) | 17 | 0 | 0 |
| reference recordings | leave-one-out (17 recordings) | head only, 5 s, 0.15 (configuration of version 0.1) | 17 | 0 | 0 |
| non-reference recordings (default model coordinates) | trained on all 17 | 0.25 chest + 0.75 head, 12 s, 0.15 (default) | 12 | 0 | 0 |

All differences are exactly zero. The weighted rule ends the last twitching later than the previous rule: on the 17 reference
recordings every offset is later, by 1.7-5.8 s (median 3.5 s); the range of the offset is
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
recordings, the offset-refinement rule (weighted and head-only), model JSON round trip and determinism, leave-one-out handling,
the overwrite guard and run manifest, per-bin max downsampling of the stacked plot, the bar plot
source table, the Excel workbook (theme, chart XML order), the review and picker pages (valid
HTML, embedded JSON schema), the checkpoint-choice conversion and the configuration round trip.
Version 0.2 adds: the frame-rate check with a stand-in ffprobe (rates, frame counting when
`nb_frames` is missing, the re-encoded 24-frames/s and 59.94-frames/s cases, the stop before the
analysis and the manifest record), the stored model (packaged model against the default
configuration, a stop for every changed configuration key, missing and altered files, stored
versus refitted results including leave-one-out models, a missing leave-one-out model, export
with `ata hmm fit`) and the legacy `end_line` configuration section.
