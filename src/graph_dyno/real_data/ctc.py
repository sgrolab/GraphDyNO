"""Loading and discovery for Cell Tracking Challenge datasets.

Layout of one dataset directory (see real_cell_data/CTC_DATA_FORMAT.md):

    <Dataset>/
      01/            raw frames  t000.tif ...
      01_GT/TRA/     tracking markers + man_track.txt (always full coverage)
      01_GT/SEG/     gold instance masks   (usually PARTIAL frame coverage)
      01_ST/SEG/     silver instance masks (near-full coverage)  <- what we use

The three small helpers `load_tif`, `sorted_frames` and `parse_man_track` are
deliberately *copied* from `real_cell_data/visualize_ctc.py` rather than
imported: that file lives outside any package, and it calls
`matplotlib.use("Agg")` at import time, which would hijack the matplotlib
backend for train.py / plot.py / the whole test session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# Repo root — used to resolve a relative `params.root` without depending on the
# process CWD (Hydra changes it), and without a `${hydra:...}` interpolation
# that would land unresolved in the hashed config.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Datasets whose only segmentation is a partial gold standard with per-frame
# relabelings — the label IDs are NOT track IDs, so every track would be
# silently corrupted. Refuse rather than produce garbage.
_UNUSABLE_DATASETS = {
    "Fluo-C2DL-Huh7": (
        "no _ST/SEG exists and 01_GT/SEG covers only 8 of 30 frames with "
        "per-frame relabelings (label IDs are not track IDs)"
    ),
}

_FRAME_NUM_RE = re.compile(r"(\d+)")


def load_tif(path: str | Path) -> np.ndarray:
    """Read a CTC TIFF.

    Uses PIL, not tifffile: the masks are LZW-compressed and tifffile needs the
    optional `imagecodecs` package (not installed) to decode LZW.
    """
    return np.array(Image.open(path))


@dataclass(frozen=True)
class SegFrame:
    """A SEG mask read through a fixed spatial crop — see `discover_video(roi=)`.

    `Video.seg_paths` holds these instead of bare paths when a video is
    cropped, so the crop travels with the path into every worker process and
    every reader sees the same sub-image. Picklable by construction.
    """

    path: Path
    #: half-open pixel bounds (y0, y1, x0, x1) into the full frame
    roi: tuple[int, int, int, int]


def load_seg(ref: str | Path | SegFrame) -> np.ndarray:
    """Read one SEG frame, applying the video's crop if it has one.

    The crop is applied to the label image itself, so everything downstream —
    `touches_border`, the contact graph, centroids, rasterization — treats the
    ROI edge exactly like the FOV edge.
    """
    if isinstance(ref, SegFrame):
        y0, y1, x0, x1 = ref.roi
        return np.ascontiguousarray(load_tif(ref.path)[y0:y1, x0:x1])
    return load_tif(ref)


def sorted_frames(dir_path: Path, pattern: str) -> list[Path]:
    files = sorted(Path(dir_path).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern} in {dir_path}")
    return files


def parse_man_track(path: Path) -> dict[int, dict[str, int]]:
    """Parse man_track.txt: track_id, birth_frame, end_frame, parent_id."""
    tracks: dict[int, dict[str, int]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        L, B, E, P = (int(x) for x in line.split())
        tracks[L] = {"birth": B, "end": E, "parent": P}
    return tracks


def frame_index_from_name(path: Path) -> int:
    """Extract the frame index encoded in `man_seg017.tif` / `t017.tif`."""
    m = _FRAME_NUM_RE.search(Path(path).stem)
    if m is None:
        raise ValueError(f"no frame index in filename: {path}")
    return int(m.group(1))


def find_seg_dir(dataset_dir: Path, seq: str, n_frames: int) -> tuple[Path, str]:
    """Pick the segmentation directory for one sequence.

    Prefers `<seq>_ST/SEG` (silver: near-full coverage, and its label IDs *are*
    track IDs). Falls back to `<seq>_GT/SEG` only when that is full-coverage
    (true for the synthetic Fluo-N2DH-SIM+). Raises otherwise.

    Returns:
        (seg_dir, kind) where kind is "ST" or "GT".
    """
    dataset_dir = Path(dataset_dir)
    name = dataset_dir.name
    if name in _UNUSABLE_DATASETS:
        raise ValueError(
            f"Dataset '{name}' cannot be used as a real-data video: "
            f"{_UNUSABLE_DATASETS[name]}."
        )

    st_dir = dataset_dir / f"{seq}_ST" / "SEG"
    if st_dir.exists() and any(st_dir.glob("man_seg*.tif")):
        return st_dir, "ST"

    gt_dir = dataset_dir / f"{seq}_GT" / "SEG"
    if gt_dir.exists():
        n_gt = len(list(gt_dir.glob("man_seg*.tif")))
        if n_gt >= n_frames:
            return gt_dir, "GT"
        raise ValueError(
            f"{name}/{seq}: no _ST/SEG, and _GT/SEG covers only {n_gt} of "
            f"{n_frames} frames. A partial gold standard cannot back a "
            f"trajectory — its label IDs are per-frame relabelings."
        )

    raise FileNotFoundError(f"{name}/{seq}: no SEG directory found")


@dataclass
class Video:
    """One CTC dataset+sequence — the frame-owning unit."""

    dataset: str
    seq: str
    root: Path
    seg_dir: Path
    seg_kind: str
    #: one entry per frame; None where the SEG directory has no mask for it.
    #: Read with `load_seg` — entries are `SegFrame`s when `roi` is set.
    seg_paths: list[Path | SegFrame | None]
    n_frames: int
    #: of the (possibly cropped) field of view
    height: int
    width: int
    #: label -> {"birth", "end", "parent"} from man_track.txt
    tracks: dict[int, dict[str, int]] = field(repr=False, default_factory=dict)
    #: (y0, y1, x0, x1) crop into the full frame, or None for the whole FOV
    roi: tuple[int, int, int, int] | None = None

    @property
    def name(self) -> str:
        return f"{self.dataset}/{self.seq}"

    @property
    def n_missing_seg(self) -> int:
        return sum(p is None for p in self.seg_paths)


def resolve_root(root: str | Path) -> Path:
    """Resolve `params.root`, treating relative paths as repo-relative."""
    p = Path(root).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def discover_video(root: str | Path, dataset: str, seq: str,
                   frame_start: int = 0, frame_end: int | None = None,
                   roi=None) -> Video:
    """Locate one dataset+sequence and index its frames and lineage table.

    Args:
        root: directory holding the dataset folders (e.g. `real_cell_data`).
        dataset: dataset folder name, e.g. "DIC-C2DH-HeLa".
        seq: "01" or "02".
        frame_start / frame_end: optional half-open time window into the video.
        roi: optional half-open spatial crop `(y0, y1, x0, x1)` in full-frame
            pixels. Cuts a dense video's cell count while keeping its local
            density — and so its contact graph — intact, which dropping tracks
            (`max_tracks_per_video`) does not. Cells cut by the ROI edge are
            border cells, handled by `border_policy` like the FOV edge.

    Returns:
        A Video with `seg_paths` aligned to the (possibly windowed) frame range.
    """
    root_path = resolve_root(root)
    dataset_dir = root_path / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")

    raw_dir = dataset_dir / seq
    tra_dir = dataset_dir / f"{seq}_GT" / "TRA"
    raw_files = sorted_frames(raw_dir, "t*.tif")
    n_raw = len(raw_files)

    seg_dir, seg_kind = find_seg_dir(dataset_dir, seq, n_raw)

    # Index SEG files by the frame number in their filename — coverage can have
    # holes, and the i-th file is not necessarily frame i.
    by_frame: dict[int, Path] = {}
    for p in sorted(seg_dir.glob("man_seg*.tif")):
        by_frame[frame_index_from_name(p)] = p

    f0 = int(frame_start)
    f1 = n_raw if frame_end is None else min(int(frame_end), n_raw)
    if not (0 <= f0 < f1):
        raise ValueError(
            f"{dataset}/{seq}: empty frame window [{f0}, {f1}) over {n_raw} frames"
        )

    seg_paths: list[Path | None] = [by_frame.get(f) for f in range(f0, f1)]

    man_track_txt = tra_dir / "man_track.txt"
    if not man_track_txt.exists():
        raise FileNotFoundError(f"{dataset}/{seq}: missing {man_track_txt}")
    tracks = parse_man_track(man_track_txt)

    # Frame shape from the first available mask.
    first = next((p for p in seg_paths if p is not None), None)
    if first is None:
        raise FileNotFoundError(
            f"{dataset}/{seq}: no SEG masks in frame window [{f0}, {f1})"
        )
    h, w = load_tif(first).shape[:2]

    if roi is not None:
        y0, y1, x0, x1 = (int(v) for v in roi)
        if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
            raise ValueError(
                f"{dataset}/{seq}: roi (y0, y1, x0, x1) = {(y0, y1, x0, x1)} "
                f"is not inside the {h}x{w} frame"
            )
        roi = (y0, y1, x0, x1)
        seg_paths = [None if p is None else SegFrame(p, roi) for p in seg_paths]
        h, w = y1 - y0, x1 - x0

    # Shift birth/end into window-local frame coordinates and clip.
    n_frames = f1 - f0
    local_tracks = {}
    for label, info in tracks.items():
        b = info["birth"] - f0
        e = info["end"] - f0
        if e < 0 or b > n_frames - 1:
            continue  # entirely outside the window
        local_tracks[label] = {
            "birth": max(0, b),
            "end": min(n_frames - 1, e),
            "parent": info["parent"],
        }

    return Video(
        dataset=dataset,
        seq=seq,
        root=root_path,
        seg_dir=seg_dir,
        seg_kind=seg_kind,
        seg_paths=seg_paths,
        n_frames=n_frames,
        height=h,
        width=w,
        tracks=local_tracks,
        roi=roi,
    )
