# Additional experiments for ResNet18 ResNet50 and Swin-Tiny

The existing four-model benchmark tests whole-line writer identification. This follow-up asks whether each of the three ImageNet-initialized backbones benefits from the same additional experiments previously run with CHURRO-3B.

| Question | Procedure |
|---|---|
| Does line augmentation help? | Five matched seeds per backbone. The augmented condition adds the same train-only copies per writer with the same deterministic transforms as CHURRO. Validation and test use the original line manifests. |
| Are individual words or whole lines more useful? | Train a word model for seeds 42–44. Single-word classification uses the accepted test words. Word votes and whole-line predictions use the same held-out lines. Word voting averages the log-probabilities of the words in a line. |
| Do the two line methods make different errors? | Count errors unique to each method and shared errors on the same test lines. Compare disagreement between seeds within each method and across methods. |
| Does an ensemble improve prediction? | Evaluate the 50/50 ensemble and a weight chosen using validation balanced accuracy, over 15 seed pairs. Also calculate same-method ensembles as controls. |

The word cache uses grayscale crops, a 3 px margin and the CHURRO quality threshold of 0.30. Preparation checks all split counts against CHURRO and matches every test word by page, line, label and token index. All whole-line evaluations retain the existing Nuremberg-9 split. ResNet18, ResNet50 and Swin-Tiny are fine-tuned end to end for the word and line variants, as they were in the four-model benchmark. CHURRO's results use frozen visual features and linear classifiers.

Run from the repository with bash nuremberg-9/experiments/07_model_details/run.sh. The script queues GPU 1 and GPU 2 in tmux, waits for the earlier ResNet18 repeat session, and reuses complete outputs. The two lane logs are in results/logs. The first seed-42 whole-line runs and completed ResNet18 repeats are reused. Each model's outputs are in results under its own name. The paper report refreshes after each completed model.

The existing whole-line Grad-CAM panels remain the common visual explanation. Swin-Tiny also has native attention panels. The ResNets use Grad-CAM.
