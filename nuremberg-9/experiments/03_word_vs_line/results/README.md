# Word vs Line results · CHURRO-3B

All three configurations use the same frozen CHURRO-3B visual encoder (block 16). The single-word and word-vote scores share the word classifier. The whole-line scores reuse the Balanced Probe classifier.

| CHURRO-3B configuration | Test units | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Word classifier, one prediction per word | 59,740 words | 89.8% | 80.7% | 75.7% |
| Word classifier, votes combined per line | 7,094 lines | 96.6% | 91.8% | 90.7% |
| Balanced Probe, one prediction per whole line | 7,094 lines | 96.1% | 92.7% | 90.0% |

The table shows the seed-42 run. Both line configurations are evaluated on the same lines. There are additional word-classifier runs with seeds 43 and 44.

[Seed 42 figure](words/churro-3b-word-vs-line.png) · [Seed 42 metrics](words/metrics.json) · [Seed 43 metrics](words_s43/metrics.json) · [Seed 44 metrics](words_s44/metrics.json)
