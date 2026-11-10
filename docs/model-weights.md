# Model weights

No weights are included in this repository. Trained DeepLabCut snapshots may be published
separately (for example on Hugging Face); this page fixes how they are named and described.

## Checkpoint naming

```
<prefix>_<batch>_<n>labels_[ft-<parent>_]<iteration>
```

| part | meaning | example (placeholder) |
|---|---|---|
| `<prefix>` | project prefix | `ata` |
| `<batch>` | short name of the training set (videos labelled in this stage) | `batchA` |
| `<n>labels` | number of **new** labelled frames added in this training stage | `200labels` |
| `ft-<parent>` | present only for a fine-tuned model: the checkpoint it started from, as `<parent batch>-<iteration>` | `ft-batchA-350k` |
| `<iteration>` | training step of the snapshot, in thousands with `k` | `380k` |

Examples: `ata_batchA_760labels_350k` (trained from the ImageNet backbone),
`ata_batchB_207labels_ft-batchA-350k_380k` (fine-tuned from the first),
`ata_batchC_23labels_ft-batchB-380k_390k` (per-posture model fine-tuned from the second).

Inside an analysis, a checkpoint is referred to by a short **tag** `<family>-<N>k` (the folder
`dlc-coords_<tag>/` holds its coordinates; `ata screen` reads family and step from the tag).
Keep a table `tag -> published name` next to the weights.

## Model card (one per checkpoint)

Facts that are the same for every checkpoint go here, not into the name:

| field | content |
|---|---|
| framework | DeepLabCut 2.2.3 (TensorFlow-1 graph, run through TensorFlow 2.11 `compat.v1`) |
| backbone | `resnet_50` |
| key points | `CHEST`, `HEAD` |
| HEAD convention | e.g. nose tip, or eye / forehead region (state it; conventions differ between label sets) |
| input | crop size (px) and frame rate (frames/s) of the training videos |
| training set | videos, number of frames, how frames were selected, pseudo labels (if any) |
| parent | the `init_weights` of the training (`pose_cfg.yaml`), i.e. ImageNet or a parent checkpoint |
| schedule | learning rates and steps after the start snapshot |
| evaluation | label error on held-out videos, label-free metrics (docs/method.md section 6) |
| intended use | which recordings the checkpoint was chosen for; videos seen during training |

## Files of one snapshot

`snapshot-<step>.index`, `snapshot-<step>.meta`, `snapshot-<step>.data-00000-of-00001`,
`pose_cfg.yaml` and the project `config.yaml` (with `project_path` and video paths replaced by
placeholders before publishing).
