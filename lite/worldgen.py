"""Deterministic first-person raycaster world -- the ground-truth simulator.

PROTECTED: this defines the "real" world dynamics and the held-out evaluation
episodes. Experiments learn to *predict* this world; they must never edit it,
or the benchmark would be gameable.

A grid map with colored walls is rendered first-person (Wolfenstein-style ray
casting). Actions move / turn a camera. Because the world is fully procedural
and deterministic, we can produce exact ground-truth future frames for any
action sequence -- enabling honest fidelity and action-following scoring.
"""
from __future__ import annotations

import numpy as np
from .common import RES, N_ACTIONS

GRID = 10                 # map is GRID x GRID cells
FOV = np.deg2rad(66.0)
MOVE = 0.35               # units per forward/back step
TURN = np.deg2rad(18.0)   # radians per turn step
MAX_DIST = 12.0
MARCH_STEP = 0.04

# palette for wall colors (deterministic per cell)
_PALETTE = np.array([
    [0.85, 0.24, 0.24], [0.24, 0.55, 0.85], [0.30, 0.75, 0.35],
    [0.85, 0.70, 0.20], [0.70, 0.35, 0.80], [0.25, 0.75, 0.75],
    [0.90, 0.50, 0.25], [0.60, 0.60, 0.65],
], dtype=np.float32)


def _make_map(seed: int):
    rng = np.random.default_rng(seed)
    m = np.zeros((GRID, GRID), dtype=np.int32)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 1     # border walls
    # scatter a few interior wall blocks (kept sparse so there's room to move)
    n_blocks = rng.integers(4, 8)
    for _ in range(n_blocks):
        cx = rng.integers(2, GRID - 2)
        cy = rng.integers(2, GRID - 2)
        m[cy, cx] = 1
    # per-cell color index (used when a cell is a wall)
    color_idx = rng.integers(0, len(_PALETTE), size=(GRID, GRID))
    return m, color_idx


def _free_pose(m, seed: int):
    rng = np.random.default_rng(seed + 777)
    while True:
        x = rng.uniform(1.5, GRID - 1.5)
        y = rng.uniform(1.5, GRID - 1.5)
        if m[int(y), int(x)] == 0:
            ang = rng.uniform(0, 2 * np.pi)
            return np.array([x, y, ang], dtype=np.float32)


def _render(state, m, color_idx) -> np.ndarray:
    """Render an RES x RES x3 float32 [0,1] first-person view from `state`."""
    x, y, ang = float(state[0]), float(state[1]), float(state[2])
    W = H = RES
    cols = np.arange(W)
    ray_ang = ang + (cols / (W - 1) - 0.5) * FOV        # (W,)
    dx = np.cos(ray_ang)
    dy = np.sin(ray_ang)

    steps = int(MAX_DIST / MARCH_STEP)
    dists = (np.arange(1, steps + 1) * MARCH_STEP)      # (S,)
    px = x + np.outer(dx, dists)                        # (W,S)
    py = y + np.outer(dy, dists)
    gx = np.clip(px.astype(np.int32), 0, GRID - 1)
    gy = np.clip(py.astype(np.int32), 0, GRID - 1)
    wall = m[gy, gx] == 1                               # (W,S)
    first = np.argmax(wall, axis=1)                     # first hit index
    hit = wall.any(axis=1)
    hit_d = np.where(hit, dists[first], MAX_DIST)       # (W,)
    # correct fish-eye: perpendicular distance
    perp = hit_d * np.cos(ray_ang - ang)
    perp = np.clip(perp, 0.15, MAX_DIST)
    ci = color_idx[gy[cols, first], gx[cols, first]]    # (W,)
    wall_col = _PALETTE[ci]                             # (W,3)
    shade = np.clip(1.0 / (1.0 + 0.35 * perp), 0.15, 1.0)[:, None]
    wall_col = wall_col * shade

    # column wall height in pixels
    line_h = np.clip((H / perp), 2, H).astype(np.int32) # (W,)
    img = np.zeros((H, W, 3), dtype=np.float32)
    rows = np.arange(H)[:, None]                        # (H,1)
    top = ((H - line_h) // 2)[None, :]                  # (1,W)
    bot = top + line_h[None, :]
    # sky (top) and floor (bottom) gradients
    sky = np.linspace(0.55, 0.25, H, dtype=np.float32)[:, None, None] * np.array([0.4, 0.55, 0.9], np.float32)
    floor = np.linspace(0.10, 0.30, H, dtype=np.float32)[:, None, None] * np.array([0.5, 0.45, 0.4], np.float32)
    img[:] = np.where(rows < H // 2, sky, floor)
    wall_mask = (rows >= top) & (rows < bot)            # (H,W)
    img[wall_mask] = np.repeat(wall_col[None, :, :], H, axis=0)[wall_mask]
    return img


def _step(state, action, m):
    x, y, ang = float(state[0]), float(state[1]), float(state[2])
    if action == 2:      # turn left
        ang -= TURN
    elif action == 3:    # turn right
        ang += TURN
    else:
        sgn = 1.0 if action == 0 else -1.0
        nx = x + sgn * MOVE * np.cos(ang)
        ny = y + sgn * MOVE * np.sin(ang)
        if m[int(ny), int(nx)] == 0:   # collide -> stay
            x, y = nx, ny
    return np.array([x, y, ang % (2 * np.pi)], dtype=np.float32)


def make_episode(seed: int, length: int):
    """Return frames (T+1,H,W,3), actions (T,), and per-step candidate GT
    next-frames for all N_ACTIONS (T,N_ACTIONS,H,W,3) for action-following."""
    m, color_idx = _make_map(seed)
    state = _free_pose(m, seed)
    rng = np.random.default_rng(seed + 5)
    frames = [_render(state, m, color_idx)]
    actions = []
    candidates = []
    for _ in range(length):
        a = int(rng.integers(0, N_ACTIONS))
        # candidate next frames for every possible action from this state
        cand = np.stack([_render(_step(state, ai, m), m, color_idx)
                         for ai in range(N_ACTIONS)], axis=0)
        candidates.append(cand)
        state = _step(state, a, m)
        frames.append(_render(state, m, color_idx))
        actions.append(a)
    return (np.stack(frames).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.stack(candidates).astype(np.float32))


def build_transition_dataset(seeds, length: int):
    """Flatten episodes into (frame_t, action, frame_t1) transitions."""
    F0, A, F1 = [], [], []
    for s in seeds:
        frames, actions, _ = make_episode(s, length)
        F0.append(frames[:-1]); F1.append(frames[1:]); A.append(actions)
    return (np.concatenate(F0), np.concatenate(A), np.concatenate(F1))


if __name__ == "__main__":
    f, a, c = make_episode(9000, 4)
    print("frames", f.shape, "actions", a.shape, "cands", c.shape,
          "range", float(f.min()), float(f.max()))
