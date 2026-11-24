# Balanced Probe results · CHURRO-3B

The CHURRO-3B visual encoder was frozen. A linear classifier was trained on block-16 features extracted from whole lines, using class weighting and square-root oversampling. These scores are from seed 42 on 7,636 test lines from nine writers.

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| CHURRO-3B block 16 + whole-line linear classifier | 95.80% | 91.00% | 88.66% |

[Confusion matrix](blk16_confusion.png) · [Recall by writer](blk16_recall.png) · [Training curves](blk16_curves.png) · [Metrics](blk16_metrics.json)
