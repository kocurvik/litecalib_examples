# LiteCalib examples

Large files of the LiteCalib project page ([kocurvik.github.io/litecalib](https://kocurvik.github.io/litecalib)),
kept out of the page's repository and served from this repository's GitHub Pages
(`https://kocurvik.github.io/litecalib_examples/`).

- `assets/` – the exported assets as they are (point clouds as `.npz`, photos, calibrations)
- `web/` – what the page loads, written from `assets/` by `convert.py`: subsampled point clouds, small copies of the
  photos and `manifest.json`
- `videos/` – the video (AV1 WebM, H.264 MP4 fallback)

After changing `assets/`, regenerate `web/` (needs numpy and Pillow):

```
python convert.py
```