# Workflow

```
raw videos ─► crop ─► label ─► fine-tune ─► infer ─► phase ─► screen / review checkpoints ─► phase (final) ─► figures
             ata crop  ata label  ata dlc        ata dlc   ata phase  ata screen, ata review          ata phase        ata figures
```

Two environments are used (see the README): `ata-dlc` (DeepLabCut 2.2.3, Python 3.7) for the
DeepLabCut steps and `ata-analysis` (Python >= 3.10) for everything else. The `ata` command runs
in `ata-analysis`; `ata dlc ...` starts the runner with `dlc.python`, the interpreter of
`ata-dlc`. Start from a copy of the default configuration:

```bash
ata config my-config.yaml          # then edit every /path/to/... value
```

## 1. Crop raw videos — `ata crop`

```bash
ata crop tool  -c my-config.yaml          # writes <crop.tool_dir>/crop_tool.html
ata crop serve -c my-config.yaml          # keep running while the page is open (localhost only)
ata crop apply -c my-config.yaml --crops-json /path/to/crops.json --dry-run
ata crop apply -c my-config.yaml --crops-json /path/to/crops.json
```

In the page: (1) mark segments (raw file, start, end; one or two animals), (2) detect the moving
animals (motion energy) and name each crop, (3) adjust the boxes and export `crops.json`. The
crop width never goes below `crop.crop_w` raw px so the pixel scale matches earlier batches.
`apply` cuts each crop with ffmpeg to `out_w x out_h`; existing files are skipped unless
`--overwrite`. With `crop.reference_dir` set, earlier clips are shown as thumbnails and matched
against the raw videos to suggest positions.

## 2. Label frames — `ata label`

```bash
ata dlc analyze -c my-config.yaml --videos /path/to/crops --dest /path/to/coords_base   # predictions of an existing model
ata label -c my-config.yaml --videos /path/to/crops --coords /path/to/coords_base --out /path/to/labelling
ata crop serve -c my-config.yaml --save-root /path/to/labelling                        # lets the page save its export
```

Per video, `label.n_outlier` frames with the highest failure score (body-length outlier, head
jump, low likelihood) and `label.n_uniform` evenly spaced frames are extracted, at least
`label.min_gap_s` apart. Points start at the model prediction; the user corrects and confirms
them and exports `labels_export_<time>.json` (saved next to `frames/` through the helper, or
downloaded).

## 3. Fine-tune — `ata dlc build-project` / `ata dlc train`

```bash
# optional: replace the HEAD labels of the base project by the predictions of another model
ata dlc pseudo-labels -c my-config.yaml --model-config /path/to/other/config.yaml --labeled-data /path/to/base/labeled-data --scratch /path/to/scratch --out /path/to/pseudo.csv
ata dlc build-project -c my-config.yaml --base-config /path/to/base/config.yaml --labels /path/to/labelling/labels_export.json --video-dir /path/to/crops --dest /path/to/new-project --base-head pseudo --pseudo-csv /path/to/pseudo.csv
ata dlc train -c my-config.yaml --project /path/to/new-project/config.yaml
```

`--base-head` decides the HEAD labels of the base project's frames: `keep`, `nan` (chest only;
use when the label convention differs) or `pseudo` (predictions with likelihood >=
`dlc.finetune.pseudo_likelihood_min`). `--holdout v1,v2` leaves videos out for an honest test.
A per-animal model for an unusual posture: `--only-videos <video> --new-only` with
`dlc.finetune.train_fraction: 1.0`. Training starts from `dlc.finetune.init_weights`; the
learning-rate schedule counts steps from that snapshot. The base project is only read.

## 4. Infer — `ata dlc analyze`

```bash
ata dlc analyze -c my-config.yaml --videos /path/to/crops --dest /path/to/coords_<tag>
```

Use one output folder per checkpoint, named after its tag (docs/model-weights.md); list them in
`inputs.coord_sets`, the default checkpoint first. For the GPU, `LD_LIBRARY_PATH` must contain
the `lib` folder of `ata-dlc` (environment-dlc.yml).

## 5. Phase analysis — `ata phase`

```bash
ata phase -c my-config.yaml --out /path/to/output/phase
```

Trains the HMM on `inputs.reference_dir`, segments every animal of the first coordinate set and
writes the per-animal table, twitching plots and `phase_model.json` (docs/outputs.md).

## 6. Screen checkpoints for flagged animals — `ata screen`, `ata review`

1. Look at `metric_flags` in `source_phase-per-animal.csv` and at the review page:
   ```bash
   ata review review -c my-config.yaml --dir /path/to/output/phase --set-name batch1
   ata review viewer -c my-config.yaml --dir /path/to/output/phase --videos /path/to/crops
   ```
2. Run the candidate snapshots on the flagged animals' videos (one folder per snapshot):
   ```bash
   ata dlc screen -c my-config.yaml --videos /path/to/flagged_videos --dest /path/to/screen_coords
   ```
   `dlc.screen` lists the snapshot families (project, iteration, steps). Each snapshot runs from a
   temporary config copy; finished snapshots are skipped, so the step can be resumed.
3. Add those folders to `inputs.coord_sets`, list the animals in `inputs.screen_animals`, then
   ```bash
   ata screen -c my-config.yaml --out /path/to/output/checkpoint-screen
   ata review picker -c my-config.yaml --dir /path/to/output/checkpoint-screen
   ```
4. Tick one checkpoint per animal in `checkpoint-picker.html` (after checking the videos), export
   the JSON and convert it:
   ```bash
   ata review choices -c my-config.yaml --json /path/to/checkpoint-choices.json --out /path/to/checkpoint-choice.csv
   ```
5. Set `inputs.checkpoint_choice` to that csv and run `ata phase` again into a new folder. Animals
   not in the csv keep the default checkpoint (all snapshots of the default stay available).

## 7. Figures — `ata figures`

```bash
ata figures all -c my-config.yaml --phase-dir /path/to/output/phase --labels /path/to/row-labels.csv
```

Needs `inputs.groups_csv` (animal, group) in the phase run. `stack`: one row per animal and
group, aligned with `bars` / `xlsx` (same x range and plot width, `figures.stack_*`). Set
`figures.stack_svg_bins` (e.g. 1200) for a compact SVG. `xlsx`: native Excel chart, every number
a formula over the `data_raw` sheet.

## Optional evaluation

```bash
ata evaluate models -c my-config.yaml --labels /path/to/labels_export.json   # metrics per checkpoint
ata evaluate grid   -c my-config.yaml                                        # settings search
```
