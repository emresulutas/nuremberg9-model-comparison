# Nuremberg-9 dataset and split

The task is to identify the writer of a masked handwritten line from the Nuremberg Letterbooks. The nine writers are S1, S2, S6, S7, S8, S9, S10, S11 and S12. Class sizes differ substantially. S1 has 12,600 training lines and S12 has 31. The comparison uses Macro-F1 and balanced accuracy alongside accuracy to show performance across writers.

The original seven writers retain their page-level split. S11 and S12 have five and three pages respectively, so their lines were split individually to include both writers in training, validation and test. The full test contains 7,636 lines. The unseen-page analysis excludes any page represented in training or validation and covers 7,583 lines from seven writers.

[Split definition](../split_v2_9scribes.json) · [Counts by writer and split](split_table.png) · [Counts CSV](split_table.csv) · [Class-size chart](lines_per_scribe_split.png)

A shortcut that predicts a book’s most common writer achieved 21.4% balanced accuracy. The CHURRO-3B whole-line classifier achieved 91.0% on the same test split. This comparison measures performance against that specific shortcut. [Shortcut baseline](shortcut_baseline.png)
