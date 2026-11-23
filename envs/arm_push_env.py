"""Franka Panda arm pushes a puck to a target zone without disturbing a cup (MuJoCo backend).

Same public API as :class:`envs.push_env.PushEnv`:

    reset(seed) -> obs
    step(action) -> (obs, done, info)
    get_state() -> np.ndarray (STATE_DIM,)
    save_snapshot() / restore_snapshot(id) / drop_snapshot(id)
    render(camera) -> uint8 RGB frame

Robot: the Franka Emika Panda model from MuJoCo Menagerie (Apache-2.0, fetched by
``scripts/fetch_assets.sh`` into ``third_party/``). The gripper stays closed and is used as the
pushing tool. The action is a 3-D end-effector (fingertip) velocity command in [-1, 1]^3, scaled
to ``EE_VMAX`` m/s. A damped-least-squares differential-IK controller (gravity-compensated arm,
fixed "pointing down" orientation, null-space pull to a home posture) turns it into joint
position targets for the arm's position servos. The controller is a pure function of the current
physics state and the action, so the env is Markov in the full MuJoCo state, and snapshot /
restore (``mjSTATE_INTEGRATION``, which includes ``ctrl``) replays bit-exactly.

World-model state vector (``STATE_DIM`` = 21, world frame, metres, m/s, rad/s):
    [0:3]   ee position          [3:6]   ee linear velocity
    [6:9]   puck position        [9:12]  puck linear velocity
    [12:14] cup xy               [14]    cup z
    [15:17] cup up-axis xy (sin of tilt, signed)
    [17:19] cup linear velocity xy
    [19:21] cup angular velocity xy
Episode goal (``env.target``, ``GOAL_DIM`` = 4): [target_x, target_y, cup0_x, cup0_y]. The STL
verifier receives it as its "target" context.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_XML = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

STATE_DIM = 21
ACT_DIM = 3
GOAL_DIM = 4
OBS_DIM = STATE_DIM + GOAL_DIM

EE = slice(0, 3)
EE_VEL = slice(3, 6)
PUCK = slice(6, 9)
PUCK_XY = slice(6, 8)
PUCK_Z = 8
PUCK_VEL = slice(9, 12)
CUP_XY = slice(12, 14)
CUP_Z = 14
CUP_UP_XY = slice(15, 17)
CUP_VEL_XY = slice(17, 19)
CUP_ANGVEL_XY = slice(19, 21)

# Indices used for the uncertainty penalty and for drawing trajectories (ee xy, puck xy, cup xy).
UNC_IDX = [0, 1, 6, 7, 12, 13]
VIS_IDX = [0, 1, 2, 6, 7, 8, 12, 13]

TABLE_X = (-0.80, 0.45)     # table top extent (z = 0)
TABLE_Y = (-0.50, 0.50)
ROBOT_BASE = (-0.55, 0.0)
WORKSPACE_HALF = 0.35       # spec: |ee_x|, |ee_y| <= 0.35
WORKSPACE_ZMAX = 0.40       # spec: ee_z <= 0.40
TARGET_RADIUS = 0.05
PUCK_RADIUS, PUCK_HALF_H = 0.03, 0.025
CUP_RADIUS, CUP_HALF_H = 0.028, 0.03

EE_VMAX = 0.25              # m/s per axis
CONTROL_DT = 0.05
PHYSICS_DT = 0.002
FRAME_SKIP = int(round(CONTROL_DT / PHYSICS_DT))  # 25
CTRL_EVERY = 5              # IK update every 5 physics steps (100 Hz)
EPISODE_STEPS = 120         # 6 s

_HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
_LEAD = 0.1                 # servo kp/kv time constant: ctrl = q + qdot_des * LEAD tracks qdot_des
_HARD_BOX = np.array([[-0.45, 0.45], [-0.45, 0.45], [0.0, 0.50]])  # controller stops pushing outward

_SCENE_XML = f"""
<mujoco model="arm_push">
  <option timestep="{PHYSICS_DT}" integrator="implicitfast" iterations="50" ls_iterations="20"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.17 0.20" rgb2="0.13 0.14 0.17"
             width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.05"/>
    <material name="table" rgba="0.78 0.72 0.62 1"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.6 2.2" dir="-0.1 0.25 -1" castshadow="true" diffuse="0.5 0.5 0.5" specular="0.05 0.05 0.05"/>
    <light pos="-0.6 0.6 1.8" dir="0.3 -0.3 -1" castshadow="false" diffuse="0.2 0.2 0.2"/>
    <camera name="top" pos="0 0 2.0" xyaxes="1 0 0 0 1 0" projection="orthographic" fovy="0.80"/>
    <camera name="front" pos="0.95 -0.85 0.75" xyaxes="0.67 0.74 0 -0.33 0.30 0.89" fovy="40"/>
    <camera name="side" pos="0.15 -1.15 0.55" xyaxes="1 0 0 0 0.42 0.91" fovy="45"/>
    <geom name="floor" type="plane" pos="0 0 -0.75" size="4 4 0.1" material="floor"/>
    <geom name="table" type="box" pos="{(TABLE_X[0] + TABLE_X[1]) / 2} 0 -0.025"
          size="{(TABLE_X[1] - TABLE_X[0]) / 2} {(TABLE_Y[1] - TABLE_Y[0]) / 2} 0.025"
          material="table" friction="0.5 0.005 0.0001"/>
    <site name="target" type="cylinder" pos="0 0 0.0005" size="{TARGET_RADIUS} 0.0005"
          rgba="0.15 0.75 0.3 0.6"/>
    <site name="workspace" type="box" pos="0 0 0.0003" size="{WORKSPACE_HALF} {WORKSPACE_HALF} 0.0003"
          rgba="0.3 0.5 0.9 0.07"/>
    <body name="puck" pos="0 0 {PUCK_HALF_H}">
      <freejoint name="puck"/>
      <geom name="puck" type="cylinder" size="{PUCK_RADIUS} {PUCK_HALF_H}" mass="0.15"
            rgba="0.95 0.55 0.15 1" friction="0.5 0.005 0.0001"/>
    </body>
    <body name="cup" pos="0.2 0.2 {CUP_HALF_H}">
      <freejoint name="cup"/>
      <geom name="cup" type="cylinder" size="{CUP_RADIUS} {CUP_HALF_H}" mass="0.06"
            rgba="0.85 0.2 0.25 1" friction="0.6 0.005 0.0001"/>
      <geom name="cup_rim" type="cylinder" pos="0 0 {CUP_HALF_H}" size="{CUP_RADIUS + 0.003} 0.004"
            mass="0.002" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_model() -> mujoco.MjModel:
    if not PANDA_XML.exists():
        raise FileNotFoundError(f"{PANDA_XML} missing: run scripts/fetch_assets.sh")
    scene = mujoco.MjSpec.from_string(_SCENE_XML)
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    for body in panda.bodies:
        if body.name and body.name != "world":
            body.gravcomp = 1.0
    panda.body("hand").add_site(name="ee", pos=[0, 0, 0.105], size=[0.006, 0, 0], rgba=[0, 0, 0, 0])
    frame = scene.worldbody.add_frame(pos=[ROBOT_BASE[0], ROBOT_BASE[1], 0.0])
    panda.option.integrator = scene.option.integrator
    scene.attach(panda, prefix="panda/", frame=frame)
    return scene.compile()


@dataclass
class _Snapshot:
    physics: np.ndarray
    t: int
    target: np.ndarray


class ArmPushEnv:
    """Panda arm + puck + cup + target zone. See module docstring."""

    _STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
    state_dim, act_dim, goal_dim = STATE_DIM, ACT_DIM, GOAL_DIM

    def __init__(self, episode_steps: int = EPISODE_STEPS, render_size=(720, 1280)):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.episode_steps = episode_steps
        self._arm_q = np.array([m.joint(f"panda/joint{i}").qposadr[0] for i in range(1, 8)])
        self._arm_v = np.array([m.joint(f"panda/joint{i}").dofadr[0] for i in range(1, 8)])
        self._fing_q = np.array([m.joint(f"panda/finger_joint{i}").qposadr[0] for i in (1, 2)])
        self._arm_u = np.array([m.actuator(f"panda/actuator{i}").id for i in range(1, 8)])
        self._grip_u = m.actuator("panda/actuator8").id
        self._ctrl_lo = m.actuator_ctrlrange[self._arm_u, 0]
        self._ctrl_hi = m.actuator_ctrlrange[self._arm_u, 1]
        self._ee = m.site("panda/ee").id
        self._puck_q = m.joint("puck").qposadr[0]
        self._puck_v = m.joint("puck").dofadr[0]
        self._cup_q = m.joint("cup").qposadr[0]
        self._cup_v = m.joint("cup").dofadr[0]
        self._target_site = m.site("target").id
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self._state_size = mujoco.mj_stateSize(m, self._STATE_SPEC)
        self._snapshots: dict[int, _Snapshot] = {}
        self._snap_ids = itertools.count()
        self._renderer = None
        self._render_size = render_size
        self.target = np.zeros(GOAL_DIM)
        self.t = 0
        self.real_steps = 0
        # Desired ee orientation = orientation at the home posture (gripper pointing down).
        self.data.qpos[self._arm_q] = _HOME_Q
        mujoco.mj_kinematics(m, self.data)
        self._R_des = self.data.site_xmat[self._ee].reshape(3, 3).copy()

    # ------------------------------------------------------------------ controller
    def _jac(self):
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self._ee)
        return np.vstack([self._jacp[:, self._arm_v], self._jacr[:, self._arm_v]])

    def _ik_dq(self, v_lin: np.ndarray, k_rot: float = 5.0, k_null: float = 1.0) -> np.ndarray:
        """Joint velocity realising ee linear velocity v_lin with orientation hold + null-space posture."""
        R = self.data.site_xmat[self._ee].reshape(3, 3)
        rot_err = 0.5 * sum(np.cross(R[:, i], self._R_des[:, i]) for i in range(3))
        twist = np.concatenate([v_lin, k_rot * rot_err])
        J = self._jac()
        JJt = J @ J.T + 1e-4 * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(JJt, np.eye(6))
        q = self.data.qpos[self._arm_q]
        dq = J_pinv @ twist
        dq += (np.eye(7) - J_pinv @ J) @ (k_null * (_HOME_Q - q))
        return dq

    def _apply_control(self, v_cmd: np.ndarray):
        # After mj_step, derived quantities (site poses, Jacobian inputs) describe the state *before*
        # the integration. Recompute them so control depends only on the current physics state;
        # otherwise a restored snapshot (followed by mj_forward) would diverge from the original run.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        ee = self.data.site_xpos[self._ee]
        v = v_cmd.copy()
        for i in range(3):  # hard stop at the controller box (the spec's workspace is smaller)
            if (ee[i] <= _HARD_BOX[i, 0] and v[i] < 0) or (ee[i] >= _HARD_BOX[i, 1] and v[i] > 0):
                v[i] = 0.0
        dq = self._ik_dq(v)
        q = self.data.qpos[self._arm_q]
        self.data.ctrl[self._arm_u] = np.clip(q + _LEAD * dq, self._ctrl_lo, self._ctrl_hi)
        self.data.ctrl[self._grip_u] = 0.0  # gripper closed

    def _solve_ik_static(self, ee_goal: np.ndarray, iters: int = 200):
        """Kinematic IK used only at reset to place the arm at a start pose."""
        m, d = self.model, self.data
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = ee_goal - d.site_xpos[self._ee]
            if np.linalg.norm(err) < 1e-5:
                break
            dq = self._ik_dq(err / CONTROL_DT, k_rot=1.0 / CONTROL_DT, k_null=0.5) * CONTROL_DT
            d.qpos[self._arm_q] += dq

    # ------------------------------------------------------------------ core API
    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        puck = rng.uniform(-0.12, 0.12, size=2)
        while True:
            ang = rng.uniform(0, 2 * np.pi)
            target = puck + rng.uniform(0.15, 0.25) * np.array([np.cos(ang), np.sin(ang)])
            if np.abs(target).max() <= 0.25:
                break
        # The cup stands near the puck->target path, offset sideways, so a careless push or a
        # careless approach knocks it.
        seg = target - puck
        normal = np.array([-seg[1], seg[0]]) / np.linalg.norm(seg)
        while True:
            cup = puck + rng.uniform(0.3, 0.9) * seg + rng.choice([-1, 1]) * rng.uniform(0.07, 0.13) * normal
            if (np.linalg.norm(cup - puck) >= 0.09 and np.linalg.norm(cup - target) >= 0.08
                    and np.abs(cup).max() <= 0.30):
                break
        ee0 = np.array([rng.uniform(-0.25, -0.05), rng.uniform(-0.2, 0.2), rng.uniform(0.15, 0.25)])
        d.qpos[self._arm_q] = _HOME_Q
        d.qpos[self._fing_q] = 0.0
        self._solve_ik_static(ee0)
        d.qpos[self._puck_q:self._puck_q + 7] = [puck[0], puck[1], PUCK_HALF_H, 1, 0, 0, 0]
        d.qpos[self._cup_q:self._cup_q + 7] = [cup[0], cup[1], CUP_HALF_H, 1, 0, 0, 0]
        d.qvel[:] = 0.0
        d.ctrl[self._arm_u] = d.qpos[self._arm_q]
        d.ctrl[self._grip_u] = 0.0
        self.target = np.array([target[0], target[1], cup[0], cup[1]])
        m.site_pos[self._target_site, :2] = target
        mujoco.mj_forward(m, d)
        self.t = 0
        return self.get_obs()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v_cmd = a * EE_VMAX
        for i in range(FRAME_SKIP):
            if i % CTRL_EVERY == 0:
                self._apply_control(v_cmd)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)  # fresh derived quantities for get_state / rendering
        self.t += 1
        self.real_steps += 1
        return self.get_obs(), self.t >= self.episode_steps, {"t": self.t}

    def get_state(self) -> np.ndarray:
        d = self.data
        mujoco.mj_jacSite(self.model, d, self._jacp, None, self._ee)
        ee = d.site_xpos[self._ee]
        ee_vel = self._jacp @ d.qvel
        pq, pv = d.qpos[self._puck_q:self._puck_q + 3], d.qvel[self._puck_v:self._puck_v + 3]
        cq = d.qpos[self._cup_q:self._cup_q + 7]
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, cq[3:7])
        R = R.reshape(3, 3)
        cv = d.qvel[self._cup_v:self._cup_v + 6]
        w_world = R @ cv[3:6]  # free-joint angular velocity is in the body frame
        return np.concatenate([ee, ee_vel, pq, pv, cq[:2], cq[2:3], R[:2, 2], cv[:2], w_world[:2]])

    def get_obs(self) -> np.ndarray:
        return np.concatenate([self.get_state(), self.target])

    # ------------------------------------------------------------------ snapshots
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
        self.model.site_pos[self._target_site, :2] = self.target[:2]
        self.t = snap.t
        mujoco.mj_forward(self.model, self.data)

    def drop_snapshot(self, snapshot_id: int) -> None:
        self._snapshots.pop(snapshot_id, None)

    # ------------------------------------------------------------------ rendering
    def render(self, camera: str = "front", size=None, scene_option=None) -> np.ndarray:
        h, w = size or self._render_size
        if self._renderer is None or self._renderer.height != h or self._renderer.width != w:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, h, w)
        self._renderer.update_scene(self.data, camera=camera, scene_option=scene_option)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def camera_projection(model: mujoco.MjModel, data: mujoco.MjData, camera: str, height: int, width: int):
    """Return a function mapping world points (..., 3) to pixel (col, row) for a perspective camera."""
    cam = model.camera(camera)
    pos = data.cam_xpos[cam.id].copy()
    R = data.cam_xmat[cam.id].reshape(3, 3).copy()  # columns: camera x, y, z (z points backwards)
    fovy = np.deg2rad(model.cam_fovy[cam.id])
    f = 0.5 * height / np.tan(fovy / 2)

    def project(p):
        p = np.asarray(p, dtype=np.float64)
        c = (p - pos) @ R  # camera-frame coordinates
        z = -c[..., 2]
        col = width / 2 + f * c[..., 0] / z
        row = height / 2 - f * c[..., 1] / z
        return np.stack([col, row], -1)

    return project
