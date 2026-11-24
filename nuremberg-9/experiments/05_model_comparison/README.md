# Nine-writer model comparison

This experiment compares the existing CHURRO-3B block-16 classifier with three
MarVen baselines retrained on Nuremberg Letterbooks. ResNet18, ResNet50 and Swin-Tiny
use ImageNet initialization and fine-tune on whole Nuremberg lines. The CHURRO block-16
reference comes from [`01_balanced_probe`](../01_balanced_probe/methods.md), which trained a
linear head on the frozen CHURRO encoder.

## Evaluation protocol

- All four always receive the same whole, masked grayscale line and produce one prediction per line. ResNet and Swin match
  CHURRO's aspect-preserving resize policy, including its 28-pixel grid and 50,176 to 1,003,520 pixel budget. Each backbone
  uses the channel normalization from its pretrained weights. The baselines process exact-size groups within a minibatch.
- The nine classes are S1, S2, S6, S7, S8, S9, S10, S11 and S12. Preparation verifies that the validation and test IDs and
  labels match the saved CHURRO run. Evaluation matches all 7,636 test rows again by `(page_id, line_id)`.
- For training use seed 42, square-root class sampling, effective-number class weights with `beta=0.999`, and no augmentation.
  The baselines follow CHURRO in using validation macro-F1 and ending early after 10 epochs without improvement, with
  a learning-rate plateau patience of 3 and a 200-epoch ceiling. The MarVen learning rate and weight decay settings are
  `3e-4 / 1e-4` for ResNet and `1e-4 / 0.05` for Swin. All three use AdamW with a logical batch size of 32.
- Compare macro-F1 first, then balanced accuracy, accuracy, recall and sample support per writer, and confusion matrices.
  The JSON gives differences from CHURRO in percentage points. These scores compare the four trained classifiers under the
  same data protocol. Pretraining, numerical normalization and backbone fine-tuning are model-specific.
- S11 and S12 share pages across splits and have 14 and 7 test lines. The separate unseen-page results exclude test pages that
  contributed any training or validation line. The report lists the writers still represented in that subset.

## Explanations

The gallery uses the same selected test crops for every model. The default draws three lines per writer with seed 42,
independent of model predictions. All four use Grad-CAM for the predicted line class. Swin-Tiny and CHURRO also show
native self-attention received by each token, averaged across attention heads and queries. The Swin projection follows
its last-stage windows. All maps are shown in the original crop coordinates and scaled independently for display.

The [CHURRO reference preview](reference/README.md) includes the saved whole-line scores, a two-line explanation panel,
and the shared list of 27 test lines.

## Run in tmux on BoXX

In VS Code Remote SSH, open `/home/esulutas/churro-nuremberg-comparison` on BoXX. This checkout contains Nuremberg-9
and the shared code. Join the `churro` tmux session from the VS Code terminal.

```bash
tmux attach -t churro
```

Run the comparison from the prompt inside the session. The script uses the prepared BoXX Python environment, weights,
and grayscale line crops. It trains the three baselines, scores them against the saved CHURRO line model, and generates
the score figures and explanation galleries. Press `Ctrl+B` followed by `D` to detach while the run continues.

```bash
CUDA_VISIBLE_DEVICES=1,2 bash nuremberg-9/scripts/run_model_comparison.sh
```

ResNet50 and ResNet18 run in sequence on GPU 1 while Swin-Tiny runs on GPU 2. Stage logs are under
`logs/model_comparison/`. A rerun reuses completed stages. `compare.py` also runs by itself on a laptop with the tracked
CHURRO predictions.

## Outputs

Under `results/whole_line/`, each of `churro/`, `resnet18/`, `resnet50/`, and `swin_tiny/` contains `fit/` for
checkpoints, predictions and metrics, and `explanations/` for the model's own gallery and raw maps. The CHURRO fit files
reference the saved whole-line run in `01_balanced_probe`.

- `comparison/metrics/` holds the main and unseen-page score tables, per-writer recall, and the four-model confusion figure.
  Each figure is available as a 300 dpi PNG.
- `comparison/selected_lines.csv` records the shared test lines used by all four models.
- `comparison/explanations/line_explanations.png` is the joint 300 dpi publication panel. The same folder contains all
  joint line galleries and raw maps. `catalog.json` records which lines were used in the publication panel.

The paper report links to these outputs and builds its own figures from the saved metrics. [Four-model report](../../report/02-model-comparison/README.md)
