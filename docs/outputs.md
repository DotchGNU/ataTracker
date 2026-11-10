# Outputs

Every step writes `run_manifest.json` into its output folder: `step`, `started`, `finished`,
`duration_s`, `command`, `versions` (Python, atatracker, numpy, scipy, pandas, hmmlearn,
matplotlib, openpyxl), `params` (the full configuration), `inputs` and `outputs` (counts). Steps
that add pages to another step's folder write `run_manifest_<step>.json`. Times are in minutes
(`*_min`) or seconds (`*_s`) as the column name says. Every plot comes as `<name>.png` (300 dpi,
transparent), `<name>.svg` (text kept as text) and `source_<name>.csv` (the plotted values).

## `ata phase`

| file | content |
|---|---|
| `phase_model.json` | fitted HMM trained on all reference recordings: `states`, `startprob`, `transmat` (4 x 4), `lambdas` (4 x 2, chest and head peaks per bin), `n_train`, `train_names` |
| `source_phase-per-animal.csv` | one row per animal (columns below) |
| `source_reference-quality-max.csv` | `metric`, `reference_max`: the flag thresholds |
| `twitching/<animal>.png/.svg`, `twitching/source_<animal>.csv` | twitching plot and its per-frame values |
| `bar_phase-durations.*` | written when `inputs.groups_csv` is set |

`source_phase-per-animal.csv`:

| column | meaning |
|---|---|
| `animal`, `group` | name from the file name; group from `inputs.groups_csv` |
| `checkpoint_tag`, `selection` | coordinate set used; `chosen` (from `checkpoint_choice`) or `default` |
| `coord_file`, `frames`, `fps`, `dur_min` | coordinate file name, frames, frame rate, recording length |
| `hmm_chest_on_min`, `hmm_post_on_min` | HMM chest-state and post-state onsets |
| `head_on_min` | head onset (chest-to-head transition) |
| `end_min` | end line (last twitching) |
| `chest_phase_min`, `head_phase_min` | head onset; end line - head onset |
| `head_peaks_in_head_phase`, `chest_peaks_in_chest_phase` | peak counts |
| `hmm_leave_one_out` | true if the animal is a reference recording and was segmented by a model trained without it |
| `manual_last_s`, `end_minus_manual_s` | manual last twitching (from `inputs.manual_csv`) and the difference |
| quality metrics | `head_lik_lt05`, `chest_lik_lt05`, `head_lik_lt09`, `bodylen_violation`, `head_jump_gt`, `outlier_frac`, `head_step_p99`, `head_step_p999`, `chest_step_p99`, `chest_step_p999`, `hc_dist_median`, `hc_dist_cv` (docs/method.md section 6) |
| `metric_flags` | `;`-separated reasons for review |

`twitching/source_<animal>.csv`: `animal, checkpoint_tag, t_s, chest_signal, head_signal,
chest_display, head_display, head_onset_s, last_twitching_s` (one row per frame; `*_signal` =
normalised amplitude, `*_display` = display trace, docs/method.md section 5).

`source_bar_phase-durations.csv`: `panel, group, n, mean, sem, animal, x, y` (one row per dot).

## `ata screen`

| file | content |
|---|---|
| `source_checkpoint-screen.csv` | one row per animal and checkpoint: the per-animal columns above plus `tag`, `family`, `iteration` (from `<family>-<N>k` tags), `set_order`, `is_default` |
| `png/<animal>/<tag>.png/.svg`, `png/<animal>/source_<tag>.csv` | twitching plot per checkpoint (dashed onset / end lines) |
| `phase_model.json` | as above |

## `ata review`

| file | content |
|---|---|
| `checkpoint-picker.html` | in the screen folder; exports `checkpoint-choices.json`: `choices` (`animal`, `display_name`, `note`, `ticked` [`tag`, `family`, `head_onset_min`, `end_min`]) and `overlap` (`tag`, `n_animals`, `animals`) |
| checkpoint choice csv (`ata review choices`) | `animal, checkpoint_tag, ticked, note` |
| `twitching-review.html` | in the phase folder; exports `twitching-review-<set>.json`: `set`, `animals` (`animal`, `group`, `weights`, `problem`, `note`, `head_on_min`, `end_min`, `metric_flags`) |
| `viewer/viewer.html`, `viewer/data/<animal>.js` | synchronised viewer; videos are linked, not copied |

The pages keep choices in the browser's localStorage until exported.

## `ata figures`

| file | content |
|---|---|
| `twitching-stack.*` | stacked plot; source: `group, row, animal, trace, t_min, value` (the drawn points, after optional per-bin max downsampling) |
| `bar_phase-durations.*` | as above |
| `phase-durations.xlsx` | sheets `data_raw` (one row per animal: `label, animal, group, chest_phase_min, head_phase_min, head_on_min, end_min, checkpoint_tag, y_jitter`) and `chart` (formulas + native chart) |

## `ata evaluate`

| step | files |
|---|---|
| `models` | `source_per-video-metrics.csv` (`tag, animal, frames` + quality metrics); with labels `source_label-error.csv` (`tag, animal, frame, head_err, chest_err, head_along_body, head_lik`) and `source_label-error-summary.csv` (`tag, n, head_err_median, head_err_p90, head_within, chest_err_median, chest_within, head_bias_along_body`) |
| `grid` | `source_grid-raw.csv`, `source_grid-ranked.csv` (`chest, head, thr, pct, lik_min, end_thr, end_win`, objectives, `rank`) |

## `ata crop`, `ata label`, `ata dlc`

| step | files |
|---|---|
| `crop tool` | `crop_tool.html`, `ref/<clip>.png`, `suggest_cache.json`, `state.json` (page state, through the helper) |
| `crop apply` | `<out_dir>/<name>.mp4`, `run_manifest_crop-apply.json` (with the ffmpeg commands) |
| `label` | `frames/<video>/img<frame>.png`, `prelabels.json`, `data.js`, `labeler.html`, labeller exports |
| `dlc analyze` | DeepLabCut output (`<video>DLC_<scorer>_<snapshot>.csv/.h5`) in `--dest` |
| `dlc screen` | `dlc-coords_<tag>/` per snapshot, `_tmp_configs/<tag>/config.yaml` |
| `dlc pseudo-labels` | csv `video, image, cx, cy, cl, hx, hy, hl` |
