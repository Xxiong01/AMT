# Dataset layout

MFT25 is not redistributed. Set `data_root` in `configs/datasets/mft25.yaml`.
The private CCT validation data are available upon reasonable request. To run
that protocol, instantiate `configs/datasets/cct_template.yaml` and provide:

```
data/CCT_FIXED/<clip>/img1/*.jpg
data/CCT_FIXED/<clip>/det/det.txt
data/CCT_FIXED/<clip>/gt/gt.txt
```

Both protocols use MOTChallenge text files and official TrackEval.
