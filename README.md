# fast_foto

Preprocessing + Kodak Portra 400 emulation for digital images (JPEG/PNG/RAW/DNG) in Python 3.13, outputting 16‑bit PNG by default with metadata sidecar.

## Features

- 7-stage preprocessing: analysis, exposure, white balance, highlight recovery, shadow lift, tone/contrast, optional noise.
- Portra 400 emulation: warm shift, per-hue HSL tweaks, per-channel + master tone curves, optional grain.
- Linear-light pipeline with neutral RAW ingest (no auto bright, no camera WB, gamma=(1,1)).
- CLI from day one (`fast-foto`) writing PNG16 plus JSON metadata by default.

## Install

```bash
uv sync
```

Or via pip (editable):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[noise]
```

`[noise]` adds `opencv-python` for bilateral denoising (optional).

## CLI usage

```bash
fast-foto INPUT --out OUTPUT.png [--preset portrait|landscape|high-iso]
                [--no-preprocess] [--no-portra]
                [--grain] [--seed N]
                [--metadata-out META.json]
                [--verbose]
```

- Default output: 16-bit PNG, metadata at `OUTPUT.png.json`.
- Presets: `portrait`, `landscape`, `high-iso` (tune preprocessing params).
- `--no-preprocess` runs only Portra emulation; `--no-portra` runs only preprocessing.
- Grain is off unless `--grain` is set.

## Library usage (Python)

```python
from preprocessing.core import preprocess_for_portra_emulation
from portra400_emulation.core import apply_portra400_emulation
from preprocessing.io import prepare_srgb_output, save_png16

rgb_linear, pre_meta = preprocess_for_portra_emulation("input.dng")
rgb_portra, portra_meta = apply_portra400_emulation(rgb_linear, grain=False)
rgb_srgb = prepare_srgb_output(rgb_portra)
save_png16("output.png", rgb_srgb)
```

## Notes on RAW handling

- RAW/DNG ingested with `rawpy` using: `use_camera_wb=False`, `use_auto_wb=False`, `user_wb=(1,1,1,1)`, `gamma=(1,1)`, `no_auto_bright=True`, `bright=1.0`, `output_color=sRGB`, `output_bps=16`, `demosaic=AHD`.
- Internal working space: linear-light sRGB float32 `[0,1]` throughout; convert to sRGB only for HSL steps and final output.

## Tests

```bash
python -m unittest discover tests
```

## Outputs

- Image: 16-bit PNG (default) via Pillow raw encoder.
- Metadata: JSON sidecar capturing load info, preprocessing stats, Portra parameters, and output paths.
