"""Backend-agnostic planar push task.

The rest of the code base only talks to :class:`PushEnv`. The single implemented backend is
MuJoCo (Unity was blocked by licensing, see LAB_NOTEBOOK.md), but the public API is kept
small so another backend can be swapped in behind it:

    reset(seed) -> obs
    step(action) -> (obs, done, info)
    get_state() -> np.ndarray (STATE_DIM,)
    save_snapshot() -> int
    restore_snapshot(snapshot_id)
    render(camera="top") -> uint8 RGB frame

State vector (``STATE_DIM`` = 10), all in metres / metres-per-second, world frame:
    [px, py, pvx, pvy, bx, by, bz, bvx, bvy, bvz]
Observation = state ++ target_xy (``OBS_DIM`` = 12).
Action: 2-D in [-1, 1], scaled to a pusher velocity command of ``PUSHER_VMAX`` m/s.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import mujoco
import numpy as np

STATE_DIM = 10
OBS_DIM = 12
ACT_DIM = 2

# Index helpers into the state vector.
PXY = slice(0, 2)
PVEL = slice(2, 4)
BXY = slice(4, 6)
BZ = 6
BVEL = slice(7, 9)

TABLE_HALF = 0.5          # table top is [-0.5, 0.5]^2 at z = 0
PUSHER_REGION_HALF = 0.45  # allowed pusher region |x|,|y| <= 0.45
TARGET_RADIUS = 0.06       # block centre must be within this radius of the target
PUSHER_VMAX = 0.5          # m/s
CONTROL_DT = 0.05          # s per env step
PHYSICS_DT = 0.005
FRAME_SKIP = int(round(CONTROL_DT / PHYSICS_DT))
EPISODE_STEPS = 100        # 5 s episodes

_XML = f"""
<mujoco model="pusher_table">
  <option timestep="{PHYSICS_DT}" integrator="implicitfast" iterations="50" ls_iterations="20">
    <flag warmstart="enable"/>
  </option>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.17 0.20" rgb2="0.13 0.14 0.17"
             width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.05"/>
    <material name="table" rgba="0.78 0.72 0.62 1"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 2.5" dir="-0.1 0.1 -1" castshadow="true" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
    <camera name="top" pos="0 0 2.0" xyaxes="1 0 0 0 1 0" projection="orthographic" fovy="1.16"/>
    <camera name="persp" pos="0 -1.25 1.05" xyaxes="1 0 0 0 0.64 0.77" fovy="45"/>
    <geom name="floor" type="plane" pos="0 0 -0.4" size="3 3 0.1" material="floor"
          contype="1" conaffinity="1"/>
    <geom name="table" type="box" pos="0 0 -0.025" size="{TABLE_HALF} {TABLE_HALF} 0.025"
          material="table" contype="1" conaffinity="1" friction="0.5 0.005 0.0001"/>
    <site name="target" type="cylinder" pos="0 0 0.0005" size="{TARGET_RADIUS} 0.0005"
          rgba="0.15 0.75 0.3 0.55"/>
    <body name="pusher" pos="0 0 0.02">
      <joint name="px" type="slide" axis="1 0 0" limited="true" range="-0.6 0.6" damping="1"/>
      <joint name="py" type="slide" axis="0 1 0" limited="true" range="-0.6 0.6" damping="1"/>
      <geom name="pusher" type="cylinder" size="0.025 0.02" mass="1.0" rgba="0.2 0.45 0.95 1"
            contype="2" conaffinity="2" friction="0.3 0.005 0.0001"/>
    </body>
    <body name="block" pos="0 0 0.02">
      <freejoint name="block"/>
      <geom name="block" type="cylinder" size="0.04 0.02" mass="0.2" rgba="0.95 0.55 0.15 1"
            contype="1" conaffinity="3" friction="0.5 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="vx" joint="px" kv="60" ctrlrange="-{PUSHER_VMAX} {PUSHER_VMAX}"/>
    <velocity name="vy" joint="py" kv="60" ctrlrange="-{PUSHER_VMAX} {PUSHER_VMAX}"/>
  </actuator>
</mujoco>
"""


@dataclass
class _Snapshot:
    physics: np.ndarray
    t: int
    target: np.ndarray


class PushEnv:
    """Planar pusher + puck + target zone on a table (MuJoCo backend)."""

    _STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

    def __init__(self, episode_steps: int = EPISODE_STEPS, render_size=(720, 720)):
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = mujoco.MjData(self.model)
        self.episode_steps = episode_steps
        self._px = self.model.joint("px").qposadr[0]
        self._py = self.model.joint("py").qposadr[0]
        self._bq = self.model.joint("block").qposadr[0]
        self._bv = self.model.joint("block").dofadr[0]
        self._pvx = self.model.joint("px").dofadr[0]
        self._pvy = self.model.joint("py").dofadr[0]
        self._target_site = self.model.site("target").id
        self._state_size = mujoco.mj_stateSize(self.model, self._STATE_SPEC)
        self._snapshots: dict[int, _Snapshot] = {}
        self._snap_ids = itertools.count()
        self._renderer = None
        self._render_size = render_size
        self.target = np.zeros(2)
        self.t = 0
        self.real_steps = 0  # every physics-advancing step() call ever made on this env

    # ------------------------------------------------------------------ core API
    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        block = rng.uniform(-0.15, 0.15, size=2)
        while True:
            ang = rng.uniform(0, 2 * np.pi)
            target = block + rng.uniform(0.15, 0.30) * np.array([np.cos(ang), np.sin(ang)])
            if np.abs(target).max() <= 0.38:
                break
        while True:
            pusher = rng.uniform(-0.35, 0.35, size=2)
            if np.linalg.norm(pusher - block) >= 0.12:
                break
        self.target = target
        self.model.site_pos[self._target_site, :2] = target
        self.data.qpos[self._px] = pusher[0]
        self.data.qpos[self._py] = pusher[1]
        self.data.qpos[self._bq:self._bq + 3] = [block[0], block[1], 0.02]
        self.data.qpos[self._bq + 3:self._bq + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return self.get_obs()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = a * PUSHER_VMAX
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        self.real_steps += 1
        done = self.t >= self.episode_steps
        return self.get_obs(), done, {"t": self.t}

    def get_state(self) -> np.ndarray:
        q, v = self.data.qpos, self.data.qvel
        return np.array([
            q[self._px], q[self._py], v[self._pvx], v[self._pvy],
            q[self._bq], q[self._bq + 1], q[self._bq + 2],
            v[self._bv], v[self._bv + 1], v[self._bv + 2],
        ])

    def get_obs(self) -> np.ndarray:
        return np.concatenate([self.get_state(), self.target])

    # ------------------------------------------------------------- snapshots
    def save_snapshot(self) -> int:
        buf = np.empty(self._state_size)
        mujoco.mj_getState(self.model, self.data, buf, self._STATE_SPEC)
        sid = next(self._snap_ids)
        self._snapshots[sid] = _Snapshot(buf, self.t, self.target.copy())
        return sid

    def restore_snapshot(self, snapshot_id: int) -> None:
        snap = self._snapshots[snapshot_id]
        mujoco.mj_setState(self.model, self.data, snap.physics, self._STATE_SPEC)
        self.target = snap.target.copy()
        self.model.site_pos[self._target_site, :2] = self.target
        self.t = snap.t
        mujoco.mj_forward(self.model, self.data)

    def drop_snapshot(self, snapshot_id: int) -> None:
        self._snapshots.pop(snapshot_id, None)

    # ------------------------------------------------------------- rendering
    def render(self, camera: str = "top", size=None) -> np.ndarray:
        h, w = size or self._render_size
        if self._renderer is None or self._renderer.height != h or self._renderer.width != w:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, h, w)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def top_camera_world_to_pixel(xy: np.ndarray, size: int = 720, extent: float = 1.16) -> np.ndarray:
    """Map world (x, y) to pixel (col, row) for the orthographic 'top' camera (square image)."""
    xy = np.asarray(xy, dtype=np.float64)
    scale = size / extent
    col = size / 2 + xy[..., 0] * scale
    row = size / 2 - xy[..., 1] * scale
    return np.stack([col, row], axis=-1)
