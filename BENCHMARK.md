# Benchmark protocol

- Test set: the 28 annotated frames of the original benchmark export.
- These frames are excluded from every training run (see training/prepare_dataset.py,
  which removes them by frame id before building the dataset).
- Matching rule: greedy one-to-one, a predicted panel is a true positive if its mask
  IoU with an unmatched ground-truth panel is >= the threshold; unmatched ground
  truth = false negative, unmatched prediction = false positive.
- Post-processing at test time is identical to detect.py: panel class only,
  min mask area 640 px, confidence 0.45, mask-level NMS 0.35.

| Model | F1 @ IoU>=0.50 | F1 @ IoU>=0.80 | F1 @ IoU>=0.90 | mean IoU |
|---|---|---|---|---|
| panel_seg_v5_l960 | 0.957 | 0.916 | 0.794 | 0.929 |
| panel_seg_v5_m960 | 0.945 | 0.886 | 0.805 | 0.928 |

Reproduce with:

```bash
cd training
python benchmark.py --weights ../models/panel_seg_v5_l960.pt --benchmark <benchmark_export_dir>
```
