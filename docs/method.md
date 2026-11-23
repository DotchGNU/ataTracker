# Method

This page states the algorithm exactly as implemented, in the order it runs. Every value in the
tables is a configuration key (`configs/default.yaml`); the defaults are the settings of the
original analysis.

## 1. Input

DeepLabCut 2.2.3 tracks two body parts per frame in a fixed-size crop of each recording
(default 480 x 272 px): `CHEST` and `HEAD`. Each coordinate table (`*DLC_*.csv`, three header
rows) gives x, y (px) and the likelihood of both points. The animal name is the file name before
`inputs.name_separator` (`DLC_`).

Every time of the analysis is frame index / `signal.fps` (default 30000/1001 = 29.97 frames/s).

**Frame-rate check** (`atatracker.fpscheck`, `ata check-fps`; run first by `ata phase` and
`ata screen` when `fps_check.require` is true). For every video in `fps_check.original_dirs`
(original recordings) and `fps_check.dlc_input_dirs` (the videos DeepLabCut analysed), ffprobe
gives the first video stream's `r_frame_rate`, `avg_frame_rate`, `nb_frames` and duration
(stream duration, else container duration). When the container has no `nb_frames`, ffprobe counts
the frames (`fps_check.count_mode`: `packets` = `-count_packets`, `decode` = `-count_frames`). The
measured rate is `nb_frames / duration`. A video fails when any of the three rates deviates from
`signal.fps` by more than `fps_check.tolerance_rel` (0.005, relative), or when the measured rate
cannot be computed; a rate the container does not report (`0/0`) is listed but does not fail on
its own. Any failure stops the step before the analysis, with the per-video table in the error
message and in `source_fps_check.csv`; the frame rate is never changed by the package.

Two cases the check is built for: a re-encoded copy whose container reports `r_frame_rate`
30000/1001 while it holds about 24 frames per second of recording (variable frame interval;
`avg_frame_rate` and `nb_frames / duration` give 24.0), and recordings made at 59.94 frames/s.
Both would make every time wrong (by -20 % and +100 %) without any other error.

## 2. Motion signal and peaks (`atatracker.signal`)

For each point separately:

1. **Step signal** per frame from the frame-to-frame displacement (dx, dy) (first frame 0):

   | mode | step |
   |---|---|
   | `y` | \|dy\| |
   | `x` | \|dx\| |
   | `xy` | sqrt(dx² + dy²) |
   | `max` | max(\|dx\|, \|dy\|) |
   | `pca` | \|projection of (dx, dy) on the first right-singular vector of all steps of this recording\| |

   Defaults: chest `y`, head `max` (an animal lying on its side moves its head sideways).
2. **Likelihood mask**: frames with likelihood < `signal.likelihood_min` (0.9) and the frame
   after each of them are set to NaN.
3. **Normalisation**: `a = (d - min(d)) / (P(d) - min(d))` with `P` the `signal.norm_percentile`
   (99.9) percentile over the unmasked frames, then clipped at 1 and NaN set to 0. If `P <= min`
   the denominator is 1.
4. **Peaks**: `scipy.signal.find_peaks(a, height=peak_threshold, distance=...)` with
   `peak_threshold` 0.5 and a minimum distance of 4 frames (chest) or 24 frames (head).
5. **Counts**: peaks per `phase.bin_s` (1 s) bin; bins `[0, 1), [1, 2), ...`, as many as
   `ceil(duration / bin_s)`. Duration = frames / `signal.fps` (30000/1001 frames/s).

The two count series form the `(T, 2)` observation matrix of the HMM.

## 3. Phase segmentation (`atatracker.phase`)

A four-state, left-to-right hidden Markov model with Poisson emissions (two independent counts
per bin), states 0 *pre*, 1 *chest*, 2 *head*, 3 *post*, fitted with `hmmlearn.hmm.PoissonHMM`.

**Stored model (default).** The model is fitted once and stored; every run applies the stored
parameters (`phase.model_file`, default `builtin` = the reference model shipped with the
package, trained on 17 recordings of the original study; [hmm-model.md](hmm-model.md)). Before it
is used, the file's parameter hash is verified and the configuration it was trained under is
compared with the current one (`signal.chest_mode`, `head_mode`, `peak_threshold`,
`norm_percentile`, `likelihood_min`, `peak_distance_chest_frames`, `peak_distance_head_frames`,
`phase.bin_s`, `signal.fps`; numbers within 1e-9). A missing file or any difference stops the
run. A target recording that is a training recording of the stored model is segmented with its
stored leave-one-out model (`phase.leave_one_out`, models in `phase.loo_dir`); every other target
uses the full model.

**Fitting** (`phase.refit: true` or `--refit`, and `ata hmm fit`). The training set is the
recordings in `inputs.reference_dir`; with `phase.refit` a target that is itself a reference
recording is segmented with a model fitted without it. The fit is:

**Initial state paths** (only used to set the starting rates), per reference recording:

* chest state from the first bin where the chest count summed over a centred window of
  `chest_onset_window_s` (5) bins reaches `chest_onset_min_peaks` (5) (bin 0 if never);
* head state from the *rule onset*: the first `rule_bin_s` (10 s) bin up to `rule_max_s` (300 s)
  where the min-max scaled head count exceeds the min-max scaled chest count (if never: half of the
  last peak time);
* post state from the bin after the last peak (chest or head).

**Initial parameters.** Rates = mean counts of each state over the initial paths, at least
`lambda_floor` (0.01); a state no path visits gets `empty_state_rate` (0.1). Start probabilities
`start_prob` (0.6, 0.4, 0, 0). Transitions: stay `stay_prob` (0.99), move to the next state
`move_prob` (0.01), all other transitions 0; the post state is absorbing.

**Fit.** EM (Baum-Welch) on all reference recordings together (`fit(X, lengths)`), updating start
probabilities, transitions and rates (`params="stl"`, `init_params=""`), at most `hmm_n_iter` (50)
iterations, tolerance `hmm_tol` (1e-4), `random_state` `random_seed` (0). Zero transitions stay
zero, so the model remains left-to-right.

**Decoding.** Viterbi path (`predict`) of each target. The onset of a state is the first bin in
that state (NaN if never visited). **Head onset** = onset of the head state. Without a head state
both head onset and offset are NaN ("no head phase found").

## 4. Offset refinement

The HMM gives the onsets; the end of twitching (the offset, last twitching) is then placed by a
rule on the motion signal (`atatracker.phase.offset_refinement`, configuration section
`offset_refinement`).

Signal per frame: `x(t) = chest_weight * chest(t) + head_weight * head(t)` (0.25 and 0.75) from
the normalised amplitudes of section 2. The envelope is the centred rolling maximum of `x` over
`window_s` (12 s, `int(12 * fps)` frames, `min_periods=1`).

`offset = last time t in [head onset, stop + search_after_post_s]` at which the envelope is
>= `threshold` (0.15); `stop` is the HMM post onset or, without one, the recording end;
`search_after_post_s` = 40 s. If no frame qualifies, `offset = stop`.

The rule of version 0.1 (head amplitude only, 5-s window, threshold 0.15) is the setting
`chest_weight: 0`, `head_weight: 1`, `window_s: 5`. How the rule was chosen and how the two
rules differ: [validation.md](validation.md).

Derived per animal: chest phase = head onset (min), head phase = offset - head onset (min),
head peaks within [onset, offset], chest peaks before the onset. In the output tables the offset
is the column `end_min`.

## 5. Display traces (plots only)

Counting never uses these. Head trace before the onset is attenuated by a soft gate
`a * (gate_floor + (1 - gate_floor) / (1 + exp(-(t - onset) / gate_tau_s)))` (floor 0, tau 3 s);
after the offset both traces are drawn as flat lines at `post_chest_level` (0.03) and
`post_head_level` (0.015) so the recording length stays visible.

## 6. Tracking quality and model choice (`atatracker.evaluate`)

Label-free metrics per recording (`quality`):

| metric | definition |
|---|---|
| `head_lik_lt05`, `chest_lik_lt05` | fraction of frames with likelihood < `likelihood_low` (0.5) |
| `bodylen_violation` | fraction of frames whose head-chest distance has robust z = (d - median) / (1.4826 MAD) beyond `body_length_z` (4) |
| `head_jump_gt` | fraction of frames whose head step > `head_jump_px` (15 px) |
| `outlier_frac` | fraction of frames failing any of the three rules above |
| `head_step_p99`, `head_step_p999`, `chest_step_p99`, `chest_step_p999` | percentiles of the frame-to-frame step (px) |
| `hc_dist_median`, `hc_dist_cv` | median head-chest distance (px) and MAD / median |

`ata phase` flags an animal when `head_lik_lt05`, `head_step_p999` or `bodylen_violation` exceeds
the maximum over the reference recordings (only when `inputs.reference_dir` is set), when it has
no head phase, or when |offset - manual last twitching| > `end_vs_manual_flag_s` (60 s).

**Label error** (`ata evaluate models --labels`): Euclidean distance (px) between tracked and
hand-labelled points; summary = median, 90th percentile, fraction within 10 px, and the median
signed head error along the chest-to-head direction.

**Checkpoint screening.** Flagged animals are re-analysed with the coordinates of several
checkpoints (`ata screen`); the researcher inspects the twitching plots and videos and ticks one
checkpoint per animal (`ata review picker`); `ata phase` then uses the chosen checkpoint for those
animals and the default checkpoint for all others (`inputs.checkpoint_choice`).

**Settings search** (`ata evaluate grid`, optional): chest mode x head mode x peak threshold x
percentile x likelihood mask (5 x 5 x 3 x 3 x 3 = 675 combinations by default), each with
offset threshold x window (4 x 3; head amplitude only, `chest_weight = 0`; the HMM is refitted for
every combination). Objectives without ground truth: (A) spread (median absolute
deviation) of offset minus manual last twitching, (B) share of animals whose head onset agrees
within 0.3 min between two checkpoints, (C) share of animals without a usable head phase
(< 10 head peaks). Rank = mean of the three ranks.

## 7. Parameter table (defaults)

| key | default | unit |
|---|---|---|
| `signal.fps` | 30000/1001 | frames/s |
| `fps_check.require` / `tolerance_rel` | true / 0.005 | — / relative |
| `phase.model_file` / `refit` / `leave_one_out` | `builtin` / false / true | — |
| `signal.chest_mode` / `head_mode` | `y` / `max` | — |
| `signal.likelihood_min` | 0.9 | — |
| `signal.norm_percentile` | 99.9 | percentile |
| `signal.peak_threshold` | 0.5 | normalised amplitude |
| `signal.peak_distance_chest_frames` / `head` | 4 / 24 | frames |
| `phase.bin_s` | 1.0 | s |
| `phase.hmm_n_iter` / `hmm_tol` / `random_seed` | 50 / 1e-4 / 0 | — |
| `phase.start_prob` | 0.6, 0.4, 0, 0 | — |
| `phase.stay_prob` / `move_prob` | 0.99 / 0.01 | — |
| `phase.lambda_floor` / `empty_state_rate` | 0.01 / 0.1 | peaks per bin |
| `phase.chest_onset_min_peaks` / `chest_onset_window_s` | 5 / 5 | peaks / s |
| `phase.rule_bin_s` / `rule_max_s` | 10 / 300 | s |
| `offset_refinement.chest_weight` / `head_weight` | 0.25 / 0.75 | — |
| `offset_refinement.threshold` / `window_s` / `search_after_post_s` | 0.15 / 12 / 40 | amplitude / s / s |
| `display.gate_tau_s` / `gate_floor` | 3 / 0 | s / — |
| `quality.body_length_z` / `head_jump_px` / `likelihood_low` | 4 / 15 / 0.5 | z / px / — |

## Software

* DeepLabCut 2.2.3 — Mathis, A. et al. DeepLabCut: markerless pose estimation of user-defined
  body parts with deep learning. *Nat. Neurosci.* 21, 1281–1289 (2018).
  [PMID 30127430](https://pubmed.ncbi.nlm.nih.gov/30127430/) ·
  [doi:10.1038/s41593-018-0209-y](https://doi.org/10.1038/s41593-018-0209-y)
* hmmlearn 0.3.3 `PoissonHMM` (<https://github.com/hmmlearn/hmmlearn>); SciPy `find_peaks`;
  pandas rolling maximum; ffmpeg 6.1.1 (libx264) for cropping; ffprobe for the frame-rate check.
