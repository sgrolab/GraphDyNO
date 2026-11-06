#!/usr/bin/env python3
"""
Web-based training monitor for viewing training plots over HTTP.
Optimized for low latency playback over SSH/remote connections.

Usage:
    python monitor_training_web.py [--port 8080] [--experiments-dir ./experiments]

Then open http://localhost:8080 in a browser.
"""

import asyncio
import json
import os
import re
import argparse
from pathlib import Path
from typing import Optional
import hashlib
import time

from aiohttp import web, WSMsgType
import mlflow

# Global state
class AppState:
    def __init__(self, experiments_dir: Path):
        self.experiments_dir = experiments_dir
        self.all_runs: list[Path] = []
        self.current_run_index = 0
        self.training_plots_dir: Optional[Path] = None
        self.epochs: list[int] = []
        self.has_embedding: bool = False
        self.image_cache: dict[str, tuple[bytes, str, float]] = {}  # path -> (data, etag, mtime)
        self.mlflow_run_ids: dict[str, Optional[str]] = {}  # hydra_output_dir -> run_id
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.websockets: set[web.WebSocketResponse] = set()

state: Optional[AppState] = None


def is_png_complete(path: Path, min_size: int = 1000) -> bool:
    """Check if a PNG file exists and is fully written."""
    if not path.exists():
        return False
    try:
        if path.stat().st_size < min_size:
            return False
        with open(path, 'rb') as f:
            header = f.read(8)
            if header[:4] != b'\x89PNG':
                return False
            f.seek(-8, 2)
            end = f.read(8)
            if b'IEND' not in end:
                return False
        return True
    except (OSError, IOError):
        return False


def scan_all_runs(experiments_dir: Path) -> list[Path]:
    """Scan for all experiment runs."""
    candidates = []

    try:
        experiments_dir.stat()
    except OSError:
        return []

    # Check regular runs
    try:
        for date_dir in experiments_dir.iterdir():
            if date_dir.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}', date_dir.name):
                for time_dir in date_dir.iterdir():
                    if time_dir.is_dir() and re.match(r'\d{2}-\d{2}-\d{2}', time_dir.name):
                        training_plots = time_dir / "training_plots"
                        if training_plots.exists():
                            mtime = training_plots.stat().st_mtime
                            candidates.append((mtime, training_plots))
    except OSError:
        pass

    # Check multirun
    multirun_dir = experiments_dir / "multirun"
    if multirun_dir.exists():
        try:
            for date_dir in multirun_dir.iterdir():
                if date_dir.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}', date_dir.name):
                    for time_dir in date_dir.iterdir():
                        if time_dir.is_dir() and re.match(r'\d{2}-\d{2}-\d{2}', time_dir.name):
                            for run_dir in time_dir.iterdir():
                                if run_dir.is_dir() and run_dir.name.isdigit():
                                    training_plots = run_dir / "training_plots"
                                    if training_plots.exists():
                                        mtime = training_plots.stat().st_mtime
                                        candidates.append((mtime, training_plots))
        except OSError:
            pass

    candidates.sort(key=lambda x: x[0])
    return [path for _, path in candidates]


def get_mlflow_run_name(hydra_output_dir: str) -> Optional[str]:
    """Look up the MLflow run name for a given Hydra output directory."""
    # Resolve to absolute path to match what's stored in MLflow
    resolved = str(Path(hydra_output_dir).resolve())

    # Check cache first
    if resolved in state.mlflow_run_ids:
        return state.mlflow_run_ids[resolved]

    run_name = None
    try:
        runs = mlflow.search_runs(
            filter_string=f"tags.hydra_output_dir = '{resolved}'",
            search_all_experiments=True,
            max_results=1,
        )
        if not runs.empty:
            run_name = runs.iloc[0]["tags.mlflow.runName"]
    except Exception:
        pass

    state.mlflow_run_ids[resolved] = run_name
    return run_name


def get_available_epochs(training_plots_dir: Path) -> tuple[list[int], bool]:
    """Get list of available epoch numbers where all images are complete.
    Returns (epochs, has_embedding)."""
    if training_plots_dir is None:
        return [], False

    try:
        training_plots_dir.stat()
    except OSError:
        return [], False

    candidate_epochs = set()
    run_has_embedding = False
    try:
        with os.scandir(training_plots_dir) as entries:
            for entry in entries:
                match = re.match(r'(?:training_epoch|generated_images_epoch|embedding_params_epoch)_(\d{4})\.png', entry.name)
                if match:
                    candidate_epochs.add(int(match.group(1)))
                    if not run_has_embedding and entry.name.startswith('embedding_params_epoch_'):
                        run_has_embedding = True
    except OSError:
        pass

    epochs = []
    for epoch in candidate_epochs:
        training_path = training_plots_dir / f"training_epoch_{epoch:04d}.png"
        generated_path = training_plots_dir / f"generated_images_epoch_{epoch:04d}.png"
        if not (is_png_complete(training_path) and is_png_complete(generated_path)):
            continue
        if run_has_embedding:
            embedding_path = training_plots_dir / f"embedding_params_epoch_{epoch:04d}.png"
            if not is_png_complete(embedding_path):
                continue
        epochs.append(epoch)

    return sorted(epochs), run_has_embedding


def refresh_state():
    """Refresh the global state."""
    state.all_runs = scan_all_runs(state.experiments_dir)
    if state.all_runs:
        # Pick the latest run that has at least one complete epoch,
        # falling back to the very latest run if none qualify.
        for i in range(len(state.all_runs) - 1, -1, -1):
            epochs, has_embedding = get_available_epochs(state.all_runs[i])
            if epochs:
                state.current_run_index = i
                state.training_plots_dir = state.all_runs[i]
                state.epochs = epochs
                state.has_embedding = has_embedding
                return
        # No run has complete epochs yet — use the latest anyway
        state.current_run_index = len(state.all_runs) - 1
        state.training_plots_dir = state.all_runs[state.current_run_index]
        state.epochs = []
        state.has_embedding = False
    else:
        state.training_plots_dir = None
        state.epochs = []
        state.has_embedding = False



def get_image_cached(path: Path) -> tuple[bytes, str]:
    """Get image data with caching. Returns (data, etag)."""
    path_str = str(path)

    try:
        mtime = path.stat().st_mtime
    except OSError:
        raise web.HTTPNotFound()

    # Check cache
    if path_str in state.image_cache:
        cached_data, cached_etag, cached_mtime = state.image_cache[path_str]
        if cached_mtime == mtime:
            return cached_data, cached_etag

    # Load fresh
    if not is_png_complete(path):
        raise web.HTTPNotFound()

    with open(path, 'rb') as f:
        data = f.read()

    etag = hashlib.md5(data).hexdigest()
    state.image_cache[path_str] = (data, etag, mtime)

    # Limit cache size (keep last 200 images)
    if len(state.image_cache) > 200:
        oldest = sorted(state.image_cache.items(), key=lambda x: x[1][2])[:50]
        for key, _ in oldest:
            del state.image_cache[key]

    return data, etag


async def broadcast_update(message: dict):
    """Send update to all connected WebSocket clients."""
    if not state.websockets:
        return

    data = json.dumps(message)
    dead = set()
    for ws in state.websockets:
        try:
            await ws.send_str(data)
        except Exception:
            dead.add(ws)
    state.websockets -= dead


async def monitor_loop():
    """Background task that checks for new epochs."""
    while state.monitoring:
        old_max = max(state.epochs) if state.epochs else 0
        old_run = state.training_plots_dir

        refresh_state()

        new_max = max(state.epochs) if state.epochs else 0

        if state.training_plots_dir != old_run or new_max > old_max:
            run_path = str(state.training_plots_dir.parent) if state.training_plots_dir else None
            mlflow_run_id = get_mlflow_run_name(run_path) if run_path else None
            await broadcast_update({
                'type': 'update',
                'epochs': state.epochs,
                'max_epoch': new_max,
                'run_index': state.current_run_index,
                'total_runs': len(state.all_runs),
                'run_path': run_path,
                'mlflow_run_id': mlflow_run_id,
                'has_embedding': state.has_embedding,
            })

        await asyncio.sleep(2.0)


# HTTP Handlers

async def handle_index(request: web.Request) -> web.Response:
    """Serve the main HTML page."""
    return web.Response(text=HTML_PAGE, content_type='text/html')


async def handle_state(request: web.Request) -> web.Response:
    """Get current state."""
    refresh_state()
    run_path = str(state.training_plots_dir.parent) if state.training_plots_dir else None
    mlflow_run_id = get_mlflow_run_name(run_path) if run_path else None
    return web.json_response({
        'epochs': state.epochs,
        'max_epoch': max(state.epochs) if state.epochs else 0,
        'run_index': state.current_run_index,
        'total_runs': len(state.all_runs),
        'run_path': run_path,
        'mlflow_run_id': mlflow_run_id,
        'monitoring': state.monitoring,
        'has_embedding': state.has_embedding,
    })


async def handle_image(request: web.Request) -> web.Response:
    """Serve an image with caching."""
    epoch = int(request.match_info['epoch'])
    img_type = request.match_info['type']  # 'training' or 'generated'

    if state.training_plots_dir is None:
        raise web.HTTPNotFound()

    if img_type == 'training':
        filename = f"training_epoch_{epoch:04d}.png"
    elif img_type == 'embedding':
        filename = f"embedding_params_epoch_{epoch:04d}.png"
    else:
        filename = f"generated_images_epoch_{epoch:04d}.png"

    path = state.training_plots_dir / filename

    # Check If-None-Match header for caching
    if_none_match = request.headers.get('If-None-Match')

    try:
        data, etag = get_image_cached(path)
    except web.HTTPNotFound:
        raise

    if if_none_match == etag:
        return web.Response(status=304)

    return web.Response(
        body=data,
        content_type='image/png',
        headers={
            'ETag': etag,
            'Cache-Control': 'public, max-age=31536000, immutable',
        }
    )


async def handle_switch_run(request: web.Request) -> web.Response:
    """Switch to a different run."""
    data = await request.json()
    direction = data.get('direction', 0)

    new_index = state.current_run_index + direction
    if 0 <= new_index < len(state.all_runs):
        state.current_run_index = new_index
        state.training_plots_dir = state.all_runs[new_index]
        state.epochs, state.has_embedding = get_available_epochs(state.training_plots_dir)
        state.image_cache.clear()

    run_path = str(state.training_plots_dir.parent) if state.training_plots_dir else None
    mlflow_run_id = get_mlflow_run_name(run_path) if run_path else None
    return web.json_response({
        'epochs': state.epochs,
        'max_epoch': max(state.epochs) if state.epochs else 0,
        'run_index': state.current_run_index,
        'total_runs': len(state.all_runs),
        'run_path': run_path,
        'mlflow_run_id': mlflow_run_id,
        'has_embedding': state.has_embedding,
    })


async def handle_monitor(request: web.Request) -> web.Response:
    """Toggle monitoring mode."""
    data = await request.json()
    enable = data.get('enable', not state.monitoring)

    if enable and not state.monitoring:
        state.monitoring = True
        state.monitor_task = asyncio.create_task(monitor_loop())
    elif not enable and state.monitoring:
        state.monitoring = False
        if state.monitor_task:
            state.monitor_task.cancel()
            try:
                await state.monitor_task
            except asyncio.CancelledError:
                pass

    return web.json_response({'monitoring': state.monitoring})


async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
    """WebSocket endpoint for real-time updates."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    state.websockets.add(ws)

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                # Client can send commands via WebSocket too
                data = json.loads(msg.data)
                if data.get('type') == 'ping':
                    await ws.send_str(json.dumps({'type': 'pong'}))
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        state.websockets.discard(ws)

    return ws


async def handle_preload(request: web.Request) -> web.Response:
    """Preload images for a range of epochs into cache."""
    data = await request.json()
    start = data.get('start', 0)
    end = data.get('end', 0)

    loaded = 0
    for epoch in range(start, end + 1):
        if epoch not in state.epochs:
            continue
        for img_type in ['training', 'generated', 'embedding']:
            if img_type == 'training':
                filename = f"training_epoch_{epoch:04d}.png"
            elif img_type == 'embedding':
                filename = f"embedding_params_epoch_{epoch:04d}.png"
            else:
                filename = f"generated_images_epoch_{epoch:04d}.png"
            path = state.training_plots_dir / filename
            try:
                get_image_cached(path)
                loaded += 1
            except:
                pass

    return web.json_response({'loaded': loaded})


HTML_PAGE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Monitor</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;700&display=swap');

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Fira Code', monospace;
            background: #282a36;
            color: #f8f8f2;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            background: #44475a;
            padding: 8px 16px;
            text-align: center;
            font-weight: bold;
            font-size: 14px;
        }

        .header .run-info { color: #bd93f9; }
        .header .run-info.monitoring { color: #50fa7b; }
        .header .run-info.playing { color: #ffb86c; }

        #mlflow-run-id:hover { color: #ff92d0 !important; }
        #mlflow-run-id .copy-icon { opacity: 0; transition: opacity 0.15s; margin-left: 4px; }
        #mlflow-run-id:hover .copy-icon { opacity: 1; }
        #mlflow-run-id.copied { color: #50fa7b !important; }
        .header .path { color: #6272a4; font-size: 12px; }
        #run-path { cursor: pointer; }
        #run-path:hover { color: #8595c4 !important; }
        #run-path .copy-icon { opacity: 0; transition: opacity 0.15s; margin-left: 4px; }
        #run-path:hover .copy-icon { opacity: 1; }
        #run-path.copied { color: #50fa7b !important; }

        .image-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
            cursor: grab;
        }

        .image-container.dragging {
            cursor: grabbing;
        }

        .image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            transform-origin: center center;
            transition: none;
            user-select: none;
            -webkit-user-drag: none;
        }

        .zoom-indicator {
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(68, 71, 90, 0.8);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            color: #6272a4;
        }

        .footer {
            background: #282a36;
            padding: 6px 16px;
            text-align: center;
            color: #6272a4;
            font-size: 12px;
        }

        .footer kbd {
            background: #44475a;
            padding: 2px 6px;
            border-radius: 3px;
            margin: 0 2px;
        }

        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #6272a4;
        }

        /* Preload images offscreen */
        .preload-container {
            position: absolute;
            left: -9999px;
            top: -9999px;
            width: 1px;
            height: 1px;
            overflow: hidden;
        }
    </style>
</head>
<body>
    <div class="header">
        <span id="mode">Training Plot</span> |
        <span id="epoch-info">Epoch 0/0</span> |
        <span id="run-info" class="run-info">Run 0/0</span>
        <span id="status"></span>
        <br>
        <span id="run-path" class="path" onclick="copyRunPath()" title="Click to copy run path"></span>
        <span id="mlflow-run-id" class="path" style="color: #ff79c6; margin-left: 8px; cursor: pointer;" onclick="copyRunName()" title="Click to copy run name">
        </span>
    </div>

    <div class="image-container" id="image-container">
        <img id="main-image" src="" alt="Training plot" draggable="false">
        <div id="loading" class="loading" style="display: none;">Loading...</div>
        <div id="zoom-indicator" class="zoom-indicator">100%</div>
    </div>

    <div class="footer">
        <kbd>c</kbd> monitor |
        <kbd>j</kbd>/<kbd>k</kbd> epoch |
        <kbd>h</kbd>/<kbd>l</kbd> run |
        <kbd>t</kbd> toggle |
        <kbd>space</kbd> play |
        <kbd>r</kbd> refresh |
        <kbd>+</kbd>/<kbd>-</kbd> speed |
        <kbd>scroll</kbd> zoom |
        <kbd>drag</kbd> pan |
        <kbd>0</kbd> reset zoom
    </div>

    <div class="preload-container" id="preload"></div>

    <script>
        const state = {
            epochs: [],
            currentEpoch: 0,
            viewMode: 'training',  // 'training', 'generated', 'embedding'
            monitoring: false,
            playing: false,
            playInterval: 100,  // ms between frames
            runIndex: 0,
            totalRuns: 0,
            runPath: '',
            mlflowRunId: null,
            hasEmbedding: false,
            preloadedImages: new Map(),
            ws: null,
            playTimer: null,
            // Zoom/pan state
            zoom: 1,
            panX: 0,
            panY: 0,
            isDragging: false,
            dragStartX: 0,
            dragStartY: 0,
            dragStartPanX: 0,
            dragStartPanY: 0,
        };

        const img = document.getElementById('main-image');
        const loading = document.getElementById('loading');
        const preloadContainer = document.getElementById('preload');
        const imageContainer = document.getElementById('image-container');
        const zoomIndicator = document.getElementById('zoom-indicator');

        // Zoom/Pan functions
        function updateImageTransform() {
            img.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
            zoomIndicator.textContent = `${Math.round(state.zoom * 100)}%`;
            zoomIndicator.style.display = state.zoom === 1 && state.panX === 0 && state.panY === 0 ? 'none' : 'block';
        }

        function resetZoom() {
            state.zoom = 1;
            state.panX = 0;
            state.panY = 0;
            updateImageTransform();
        }

        function handleWheel(e) {
            e.preventDefault();
            const rect = imageContainer.getBoundingClientRect();
            const mouseX = e.clientX - rect.left - rect.width / 2;
            const mouseY = e.clientY - rect.top - rect.height / 2;

            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const newZoom = Math.max(0.5, Math.min(10, state.zoom * zoomFactor));

            // Adjust pan to zoom towards mouse position
            const zoomChange = newZoom / state.zoom;
            state.panX = mouseX - (mouseX - state.panX) * zoomChange;
            state.panY = mouseY - (mouseY - state.panY) * zoomChange;
            state.zoom = newZoom;

            updateImageTransform();
        }

        function handleMouseDown(e) {
            if (e.button !== 0) return; // Left click only
            e.preventDefault();
            state.isDragging = true;
            state.dragStartX = e.clientX;
            state.dragStartY = e.clientY;
            state.dragStartPanX = state.panX;
            state.dragStartPanY = state.panY;
            imageContainer.classList.add('dragging');
        }

        function handleMouseMove(e) {
            if (!state.isDragging) return;
            state.panX = state.dragStartPanX + (e.clientX - state.dragStartX);
            state.panY = state.dragStartPanY + (e.clientY - state.dragStartY);
            updateImageTransform();
        }

        function handleMouseUp() {
            state.isDragging = false;
            imageContainer.classList.remove('dragging');
        }

        function handleDoubleClick() {
            resetZoom();
        }

        // Attach zoom/pan event listeners
        imageContainer.addEventListener('wheel', handleWheel, { passive: false });
        imageContainer.addEventListener('mousedown', handleMouseDown);
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        imageContainer.addEventListener('dblclick', handleDoubleClick);
        img.addEventListener('dragstart', (e) => e.preventDefault());

        // Preload images around current epoch
        function preloadImages() {
            const idx = state.epochs.indexOf(state.currentEpoch);
            if (idx === -1) return;

            // Preload 10 epochs ahead and behind
            const start = Math.max(0, idx - 10);
            const end = Math.min(state.epochs.length - 1, idx + 10);

            for (let i = start; i <= end; i++) {
                const epoch = state.epochs[i];
                for (const type of state.hasEmbedding ? ['training', 'generated', 'embedding'] : ['training', 'generated']) {
                    // Include run index in key to avoid cross-run cache hits
                    const key = `${state.runIndex}-${epoch}-${type}`;
                    if (!state.preloadedImages.has(key)) {
                        const preloadImg = new Image();
                        preloadImg.src = `/image/${epoch}/${type}?run=${state.runIndex}`;
                        state.preloadedImages.set(key, preloadImg);
                    }
                }
            }

            // Clean old preloaded images (from other runs or far epochs)
            if (state.preloadedImages.size > 50) {
                const currentIdx = state.epochs.indexOf(state.currentEpoch);
                for (const [key, _] of state.preloadedImages) {
                    const parts = key.split('-');
                    const runIdx = parseInt(parts[0]);
                    const epoch = parseInt(parts[1]);
                    const epochIdx = state.epochs.indexOf(epoch);
                    // Remove if different run or too far from current epoch
                    if (runIdx !== state.runIndex || Math.abs(epochIdx - currentIdx) > 20) {
                        state.preloadedImages.delete(key);
                    }
                }
            }
        }

        function updateDisplay() {
            const type = state.viewMode;
            const url = `/image/${state.currentEpoch}/${type}?run=${state.runIndex}`;

            // Use preloaded image if available (key includes run index)
            const key = `${state.runIndex}-${state.currentEpoch}-${type}`;
            const preloaded = state.preloadedImages.get(key);

            if (preloaded && preloaded.complete && preloaded.naturalWidth > 0) {
                img.src = preloaded.src;
            } else {
                loading.style.display = 'block';
                img.onload = () => { loading.style.display = 'none'; };
                img.src = url;
            }

            // Update header
            const modeLabels = {training: 'Training Plot', generated: 'Generated Images', embedding: 'Embedding Params'};
            document.getElementById('mode').textContent = modeLabels[state.viewMode];
            document.getElementById('epoch-info').textContent =
                `Epoch ${state.currentEpoch}/${state.epochs.length > 0 ? Math.max(...state.epochs) : '?'}`;

            const runInfo = document.getElementById('run-info');
            runInfo.textContent = `Run ${state.runIndex + 1}/${state.totalRuns}`;
            runInfo.className = 'run-info' +
                (state.monitoring ? ' monitoring' : '') +
                (state.playing ? ' playing' : '');

            let statusText = '';
            if (state.monitoring) statusText = ' [MONITORING]';
            else if (state.playing) statusText = ` [PLAYING ${Math.round(1000/state.playInterval)}fps]`;
            document.getElementById('status').textContent = statusText;

            const pathEl = document.getElementById('run-path');
            if (state.runPath) {
                pathEl.innerHTML = `${state.runPath}<span class="copy-icon">📋</span>`;
            } else {
                pathEl.innerHTML = '';
            }
            const mlflowEl = document.getElementById('mlflow-run-id');
            if (state.mlflowRunId) {
                mlflowEl.innerHTML = `mlflow: ${state.mlflowRunId}<span class="copy-icon">📋</span>`;
            } else {
                mlflowEl.innerHTML = '';
            }

            // Preload nearby images
            preloadImages();
        }

        async function fetchState() {
            const resp = await fetch('/state');
            const data = await resp.json();
            state.epochs = data.epochs;
            state.runIndex = data.run_index;
            state.totalRuns = data.total_runs;
            state.runPath = data.run_path;
            state.mlflowRunId = data.mlflow_run_id;
            state.monitoring = data.monitoring;
            state.hasEmbedding = data.has_embedding;
            if (state.hasEmbedding === false && state.viewMode === 'embedding') {
                state.viewMode = 'training';
            }
            if (state.epochs.length > 0 && !state.epochs.includes(state.currentEpoch)) {
                state.currentEpoch = Math.max(...state.epochs);
            }
            updateDisplay();
        }

        async function switchRun(direction) {
            const resp = await fetch('/switch_run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({direction})
            });
            const data = await resp.json();
            state.epochs = data.epochs;
            state.runIndex = data.run_index;
            state.totalRuns = data.total_runs;
            state.runPath = data.run_path;
            state.mlflowRunId = data.mlflow_run_id;
            state.hasEmbedding = data.has_embedding;
            if (state.hasEmbedding === false && state.viewMode === 'embedding') {
                state.viewMode = 'training';
            }
            state.preloadedImages.clear();
            if (state.epochs.length > 0) {
                state.currentEpoch = Math.max(...state.epochs);
            }
            updateDisplay();
        }

        async function toggleMonitoring() {
            const enabling = !state.monitoring;
            const resp = await fetch('/monitor', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enable: enabling})
            });
            const data = await resp.json();
            state.monitoring = data.monitoring;

            if (state.monitoring && state.playing) {
                togglePlayback();
            }

            // When enabling monitoring, refresh state to jump to latest run/epoch
            if (enabling) {
                state.preloadedImages.clear();
                await fetchState();
                // Always jump to latest epoch when enabling monitoring
                if (state.epochs.length > 0) {
                    state.currentEpoch = Math.max(...state.epochs);
                    updateDisplay();
                }
            } else {
                updateDisplay();
            }
        }

        function togglePlayback() {
            state.playing = !state.playing;

            if (state.playing) {
                if (state.monitoring) {
                    toggleMonitoring();
                }
                state.playTimer = setInterval(playNextFrame, state.playInterval);
            } else {
                if (state.playTimer) {
                    clearInterval(state.playTimer);
                    state.playTimer = null;
                }
            }
            updateDisplay();
        }

        function playNextFrame() {
            if (!state.playing || state.epochs.length === 0) return;

            const idx = state.epochs.indexOf(state.currentEpoch);
            if (idx < state.epochs.length - 1) {
                state.currentEpoch = state.epochs[idx + 1];
            } else {
                state.currentEpoch = state.epochs[0];
            }
            updateDisplay();
        }

        function adjustSpeed(factor) {
            state.playInterval = Math.max(16, Math.min(1000, state.playInterval * factor));
            if (state.playing && state.playTimer) {
                clearInterval(state.playTimer);
                state.playTimer = setInterval(playNextFrame, state.playInterval);
            }
            updateDisplay();
        }

        // WebSocket for real-time updates
        function connectWebSocket() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            state.ws = new WebSocket(`${protocol}//${location.host}/ws`);

            state.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'update') {
                    const runChanged = data.run_index !== state.runIndex;
                    state.epochs = data.epochs;
                    state.runIndex = data.run_index;
                    state.totalRuns = data.total_runs;
                    state.runPath = data.run_path;
                    state.mlflowRunId = data.mlflow_run_id;
                    state.hasEmbedding = data.has_embedding;
                    if (state.hasEmbedding === false && state.viewMode === 'embedding') {
                        state.viewMode = 'training';
                    }
                    if (state.monitoring && (runChanged || data.max_epoch > state.currentEpoch)) {
                        state.currentEpoch = data.max_epoch;
                    }
                    if (state.epochs.length > 0 && !state.epochs.includes(state.currentEpoch)) {
                        state.currentEpoch = Math.max(...state.epochs);
                    }
                    updateDisplay();
                }
            };

            state.ws.onclose = () => {
                setTimeout(connectWebSocket, 1000);
            };
        }

        // Keyboard handling
        document.addEventListener('keydown', (e) => {
            const key = e.key.toLowerCase();

            switch(key) {
                case 'c':
                    toggleMonitoring();
                    break;
                case 'j':
                    if (state.monitoring) toggleMonitoring();
                    if (state.playing) togglePlayback();
                    if (state.epochs.length > 0) {
                        const idx = state.epochs.indexOf(state.currentEpoch);
                        if (idx > 0) {
                            state.currentEpoch = state.epochs[idx - 1];
                            updateDisplay();
                        }
                    }
                    break;
                case 'k':
                    if (state.monitoring) toggleMonitoring();
                    if (state.playing) togglePlayback();
                    if (state.epochs.length > 0) {
                        const idx = state.epochs.indexOf(state.currentEpoch);
                        if (idx < state.epochs.length - 1) {
                            state.currentEpoch = state.epochs[idx + 1];
                            updateDisplay();
                        }
                    }
                    break;
                case 't': {
                    const modes = state.hasEmbedding
                        ? ['training', 'generated', 'embedding']
                        : ['training', 'generated'];
                    state.viewMode = modes[(modes.indexOf(state.viewMode) + 1) % modes.length];
                    resetZoom();
                    updateDisplay();
                    break;
                }
                case 'h':
                    if (state.monitoring) toggleMonitoring();
                    if (state.playing) togglePlayback();
                    switchRun(-1);
                    break;
                case 'l':
                    if (state.monitoring) toggleMonitoring();
                    if (state.playing) togglePlayback();
                    switchRun(1);
                    break;
                case 'r':
                    fetchState();
                    break;
                case ' ':
                    e.preventDefault();
                    togglePlayback();
                    break;
                case '+':
                case '=':
                    adjustSpeed(0.7);  // Faster
                    break;
                case '-':
                    adjustSpeed(1.4);  // Slower
                    break;
                case '0':
                    resetZoom();
                    break;
            }
        });

        function copyRunName() {
            if (!state.mlflowRunId) return;
            navigator.clipboard.writeText(state.mlflowRunId).then(() => {
                const el = document.getElementById('mlflow-run-id');
                el.classList.add('copied');
                setTimeout(() => { el.classList.remove('copied'); }, 1000);
            });
        }

        function copyRunPath() {
            if (!state.runPath) return;
            navigator.clipboard.writeText(state.runPath).then(() => {
                const el = document.getElementById('run-path');
                el.classList.add('copied');
                setTimeout(() => { el.classList.remove('copied'); }, 1000);
            });
        }

        // Initialize
        fetchState();
        connectWebSocket();
    </script>
</body>
</html>
'''


def create_app(experiments_dir: Path) -> web.Application:
    global state
    database_dir = Path(__file__).resolve().parent.parent
    mlflow.set_tracking_uri(f"sqlite:///{database_dir}/mlflow.db")
    state = AppState(experiments_dir)
    refresh_state()

    app = web.Application()
    app.router.add_get('/', handle_index)
    app.router.add_get('/state', handle_state)
    app.router.add_get('/image/{epoch}/{type}', handle_image)
    app.router.add_post('/switch_run', handle_switch_run)
    app.router.add_post('/monitor', handle_monitor)
    app.router.add_post('/preload', handle_preload)
    app.router.add_get('/ws', handle_websocket)

    return app


def main():
    parser = argparse.ArgumentParser(description="Web-based training monitor")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path(__file__).parent.parent / "experiments",
        help="Path to experiments directory"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port (default: 8080)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)"
    )
    args = parser.parse_args()

    print(f"Starting Training Monitor Web Server")
    print(f"=" * 40)
    print(f"URL: http://localhost:{args.port}")
    print(f"Experiments dir: {args.experiments_dir}")
    print(f"=" * 40)

    app = create_app(args.experiments_dir)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
