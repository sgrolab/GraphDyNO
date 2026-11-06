# Cell Tracking Challenge (CTC) data — format reference

This folder holds the two **training** datasets (raw images + full annotations) the paper
uses from the [Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/)
(Ulman et al. 2017, *Nat. Methods*; Maška et al. 2023, *Nat. Methods*): DIC-C2DH-HeLa and
PhC-C2DL-PSC. Only their `man_track.txt` files are versioned here — unzip the two training
zips into this folder to get the images and masks (see the README). Test/challenge zips
(raw images only, annotations withheld) are not needed: no ground truth in them.

```
real_cell_data/
├── <Dataset>/
│   ├── 01/, 02/                     # raw frames, two independent sequences
│   ├── 01_GT/TRA/, 02_GT/TRA/        # gold tracking markers (hand-annotated, ALWAYS full coverage)
│   ├── 01_GT/SEG/, 02_GT/SEG/        # gold segmentation masks (hand-annotated, PARTIAL coverage)
│   ├── 01_ST/SEG/, 02_ST/SEG/        # silver segmentation masks (algorithm-consensus, ~full coverage)
│   └── 01_ERR_SEG/, 02_ERR_SEG/      # per-frame candidate segmentations used to build ST/error-check
```

`01_ST`, `01_ERR_SEG` don't exist for every dataset (see table below) — `Fluo-C2DL-Huh7`
ships gold (`_GT`) annotations only.

## The two annotation types, and why they're not interchangeable

**TRA (tracking) masks are NOT cell outlines.** Each `man_trackNNN.tif` labels every cell
with a **small blob placed somewhere inside the cell**, not its boundary — this is a
"tracking marker," sufficient to say *where* a cell is and *which track it belongs to*,
cheap enough for annotators to mark on every single frame. TRA coverage is **always
complete**: every cell, every frame, hand-annotated by three experts and merged by
majority vote (round dots, not cell shapes).

**SEG masks are true instance segmentations** (full cell boundary), but expensive to
hand-annotate, so **gold SEG (`_GT/SEG`) only covers a partial, representative subset of
frames** (e.g. `DIC-C2DH-HeLa/01_GT/SEG` has 9 of 84 frames; `Fluo-N2DL-HeLa/01_GT/SEG` has
28 of 92). **Silver SEG (`_ST/SEG`)** — built by fusing the best-performing algorithms'
outputs from past CTC submissions, checked against the TRA ground truth — reaches **near-100%
frame coverage** and is what you want for "silver truth is fine" full-boundary masks.
`_ERR_SEG`
holds the raw per-algorithm candidate masks that silver-truth fusion draws from; not needed
unless you're studying segmentation-algorithm error modes specifically.

**Practical rule:** use `_GT/TRA` for anything involving trajectories/lineages (always
complete). Use `_ST/SEG` for full-boundary instance masks (near-complete). Use `_GT/SEG`
only to spot-check `_ST/SEG` quality on the frames where hand-verified truth exists.

## `man_track.txt` — the lineage table

One row per track, space-separated, in `<seq>_GT/TRA/man_track.txt`:

```
L  B  E  P
```
- `L` — track label (matches pixel values in `man_trackNNN.tif`)
- `B` — first frame the track appears (0-indexed)
- `E` — last frame the track appears
- `P` — parent track's label, or `0` if the track starts with no parent (no division)

Example, `DIC-C2DH-HeLa/01_GT/TRA/man_track.txt`:
```
1 0 29 0      # track 1: frames 0–29, no parent
3 30 83 1     # track 3: frames 30–83, parent is track 1  → division at frame 30
4 30 83 1     # track 4: frames 30–83, parent is track 1  → sibling of track 3
```
Tracks 3 and 4 both starting at frame 30 with parent 1 means cell 1 divided into cells 3
and 4 between frames 29 and 30. A track with no children just ends at `E` (cell left the
frame, died, or the movie ended).

## Per-dataset notes

The paper uses DIC-C2DH-HeLa and PhC-C2DL-PSC; the rest of the CTC 2D training corpus is
listed for reference.

| Dataset | Modality | Cell type | seq 01 frames | Peak concurrent cells | Divisions | Has `_ST` |
|---|---|---|---:|---:|---:|:---:|
| PhC-C2DL-PSC | Phase contrast | Pancreatic stem cells | 300 | 661 | 1168 | yes |
| Fluo-N2DL-HeLa | Fluorescence | HeLa (human), H2b-GFP | 92 | 137 | 188 | yes |
| Fluo-N2DH-SIM+ | Fluorescence (**synthetic**) | Simulated HL60 nuclei | 65 | 47 | 59 | no — GT/SEG is full-coverage (synthetic ⇒ perfect, no silver needed) |
| Fluo-C2DL-Huh7 | Fluorescence | Human hepatocarcinoma (YFP-TIA-1) | 30 | 32 | 3 | **no** — GT only |
| Fluo-N2DH-GOWT1 | Fluorescence | Mouse ES cells, GFP | 92 | 25 | 5 | yes |
| BF-C2DL-MuSC | Brightfield | Mouse muscle stem cells | 1376 | 23 | 70 | yes |
| DIC-C2DH-HeLa | DIC | HeLa (human) | 84 | 18 | 18 | yes |
| BF-C2DL-HSC | Brightfield | Mouse hematopoietic stem cells | 1764 | 12 | 22 | yes |
| Fluo-C2DL-MSC | Fluorescence | Rat mesenchymal stem cells | 48 | 11 | 0 | yes |
| PhC-C2DH-U373 | Phase contrast | Glioblastoma-astrocytoma (human) | 115 | 7 | 1 | yes |

"Peak concurrent cells" = the maximum number of tracks alive in any single frame
(computed from `man_track.txt` alone, no image loading needed).
"Divisions" = number of tracks with a nonzero parent, i.e. total mitotic events across the
sequence (**not** number of dividing lineages — one division produces 2 new tracks, so a
dataset with only division events but each producing exactly 2 children has
divisions ≈ 2× the number of division *events*).

Two independent sequences (`01`, `02`) ship per dataset; the table above uses `01` only.
`02` exists on disk with identical structure.

## Reading the data in Python

**Do not use plain `tifffile.imread`** — the CTC mask TIFFs are LZW-compressed, and
`tifffile` needs the optional `imagecodecs` package to decode LZW, which is not installed
in this project's venv. **`PIL.Image.open` reads them natively** (Pillow ships its own LZW
decoder) — this is what `graph_dyno.real_data.ctc.load_tif` uses:

```python
from PIL import Image
import numpy as np
mask = np.array(Image.open("DIC-C2DH-HeLa/01_GT/TRA/man_track000.tif"))  # uint16, labels
raw  = np.array(Image.open("DIC-C2DH-HeLa/01/t000.tif"))                 # uint8, grayscale
```

Two maintained third-party libraries also load CTC data directly (neither is a
dependency of this repo): `byotrack.dataset.ctc` and `traccuracy.loaders.load_ctc_data`.
