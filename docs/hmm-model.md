# Stored HMM

The phase HMM (docs/method.md section 3) is fitted once and stored. `ata phase` and `ata screen`
load the stored parameters, so every run and every batch is segmented by the same model.

## The reference model shipped with the package

`src/atatracker/data/hmm_reference-17_full.json` (`phase.model_file: builtin`):

| | |
|---|---|
| model | 4-state left-to-right Poisson HMM, states pre → chest → head → post, two count series (chest and head peaks per 1-s bin) |
| training set | 17 recordings of one cohort of the original study, fitted together (EM, at most 50 iterations, tolerance 1e-4, seed 0; converged after 9 iterations) |
| training names | replaced by anonymous IDs `ref01` … `ref17` for publication; the parameters are unchanged |
| parameter hash | `9a9100d891fa273c` |
| configuration | chest `y`, head `max`, peak threshold 0.5, 99.9th percentile, likelihood mask 0.9, peak distance 4 / 24 frames, 1-s bins, 30000/1001 frames/s |
| software | hmmlearn 0.3.3, numpy 2.4.6 |

The file holds only probabilities, Poisson rates, the configuration, the fit settings and the
anonymous training IDs. Its leave-one-out models are not shipped: they are only meaningful for
the 17 training recordings themselves.

Fitted parameters (rates in peaks per 1-s bin; transitions per bin):

| state | start | stay | move on | chest rate | head rate |
|---|---|---|---|---|---|
| pre | 1.0 | 0.9549 | 0.0451 | 0.0760 | 0.0027 |
| chest | 1.9e-81 | 0.9867 | 0.0133 | 2.9477 | 0.0071 |
| head | 0 | 0.9915 | 0.0085 | 0.5857 | 0.2883 |
| post | 0 | 1 | — | 0.0085 | 0.0025 |

## Checks on loading

| check | on failure |
|---|---|
| the file exists | stop (`FileNotFoundError`) |
| `params_hash` = SHA-256 (first 16 hex digits) of the canonical JSON of `params` | stop |
| 4 states `pre, chest, head, post`, 2 count series | stop |
| stored `config` = current configuration (table below; numbers within 1e-9) | stop, listing every differing key |
| a target is a training recording and `phase.leave_one_out` is true, but no leave-one-out model for it is in `phase.loo_dir` | stop |

| key in the file | configuration |
|---|---|
| `SIG_CHEST`, `SIG_HEAD` | `signal.chest_mode`, `signal.head_mode` |
| `THR` | `signal.peak_threshold` |
| `NORM_HI_PCT` | `signal.norm_percentile` |
| `LIK_MIN` | `signal.likelihood_min` |
| `DIST_CHEST`, `DIST_HEAD` | `signal.peak_distance_chest_frames`, `signal.peak_distance_head_frames` (frames) |
| `BIN_S` | `phase.bin_s` (s) |
| `FPS` | `signal.fps` (frames/s) |

The offset-refinement settings are not part of the model; they can change without refitting.

## Your own model

```bash
ata hmm fit   -c my-config.yaml --out /path/to/hmm-model    # fit on inputs.reference_dir: full + one leave-one-out model per recording
ata hmm check -c my-config.yaml                             # load phase.model_file and check it against the configuration
```

Then set `phase.model_file: /path/to/hmm-model/hmm_full.json`. The leave-one-out models
(`hmm_loo_<name>.json`) are found in the same folder (or `phase.loo_dir`) by their `kind` and
`left_out` entries, so file names do not matter. `--no-loo` writes the full model only.

`phase.refit: true` (or `ata phase --refit`) fits on `inputs.reference_dir` on every run instead
of loading a model (the behaviour of version 0.1). In the regression (hmmlearn 0.3.3, numpy 2.4.6) a refit on the same
recordings gave bit-identical parameters; other library versions may differ in the last digits.

## File format

One JSON object per model:

| key | content |
|---|---|
| `kind`, `left_out` | `full` (with `left_out` null) or `loo` (with the left-out recording) |
| `n_components`, `n_features`, `states`, `signals` | 4, 2, `[pre, chest, head, post]`, `[chest_count_per_s, head_count_per_s]` |
| `params` | `startprob` (4), `transmat` (4 x 4), `lambdas` (4 x 2, chest and head peaks per bin) |
| `params_hash` | first 16 hex digits of SHA-256 of `json.dumps(params, sort_keys=True)` |
| `training_animals`, `n_training` | training recordings |
| `config` | the configuration keys of the table above |
| `fit_settings` | `iter`, `tol`, `seed`, `params`, `init_params`, `start_prob`, `init_stay`, `init_advance`, `lambda_floor`, `init_paths` |
| `converged`, `n_iter_run` | EM convergence and iterations |
| `hmmlearn_version`, `numpy_version` | library versions of the fit |

`ata phase` writes the full model it applied as `phase_model.json` in this format and records
`source`, `model_file`, `params_hash`, `n_training` and the number of leave-one-out models under
`hmm` in `run_manifest.json`; the per-animal table has the hash of the model used for each
animal (`hmm_params_hash`).
