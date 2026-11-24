"""Render the arm demo video (results/arm_demo.mp4 + results/arm_demo.gif) from recorded traces.

Frames are rendered programmatically: MuJoCo offscreen renderer with a free 3-D camera restored
from the recorded full physics states, PIL overlays (imagined plans, verified branches, committed
plan) projected into the image, and the same HUD panel as the planar demo. Stitched with
imageio + imageio-ffmpeg.

Usage: python scripts/make_arm_video.py   (expects results/arm_demo_traces/*.pkl and
       results/arm_summary.json from experiments/analyze_arm.py)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs import arm_push_env as arm  # noqa: E402
from scripts.make_video import (AMBER, BG, BLUE, CAND_COLORS, CYAN, DIM, F_BIG, F_MED, F_SMALL, FG,  # noqa: E402
                                FPS, GREEN, H, RED, SCENE, W, WHITE, Video, clause_status, compose,
                                draw_path, hud, plot_slide, title_card)
from verifier.verifier import Spec  # noqa: E402

TRACES = ROOT / "results" / "arm_demo_traces"
T_MAX = arm.EPISODE_STEPS


# ---------------------------------------------------------------------------- 3-D scene
class ArmScene:
    """Renders recorded physics states with a free camera and projects world points to pixels."""

    def __init__(self, size=SCENE):
        self.size = size
        self.env = arm.ArmPushEnv(render_size=(size, size))
        self.renderer = mujoco.Renderer(self.env.model, size, size)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.fovy = float(self.env.model.vis.global_.fovy)
        self.set_view(np.array([-0.02, 0.0, 0.02]), 1.25, 140.0, -38.0)

    def set_view(self, lookat, distance, azimuth, elevation):
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        self._pos = np.asarray(lookat, dtype=np.float64) - distance * fwd
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        self._fwd, self._right, self._up = fwd, right, np.cross(right, fwd)
        self._f = 0.5 * self.size / np.tan(np.deg2rad(self.fovy) / 2)

    def frame_episode(self, tr, zoom=1.0):
        """Look at the region spanned by the puck, cup and target of an episode."""
        h, tgt = tr["history"], tr["target"]
        pts = np.concatenate([h[:, 6:8], h[:, 12:14], tgt[None, :2], tgt[None, 2:4]])
        c = 0.5 * (pts.min(0) + pts.max(0))
        span = float((pts.max(0) - pts.min(0)).max())
        self.set_view(np.array([c[0] - 0.10, c[1], 0.10]), zoom * max(1.05, 1.5 * span + 0.75), 140.0, -34.0)

    def px(self, p):
        d = np.asarray(p, dtype=np.float64) - self._pos
        z = d @ self._fwd
        col = self.size / 2 + self._f * (d @ self._right) / z
        row = self.size / 2 - self._f * (d @ self._up) / z
        return np.stack([col, row], -1)

    def render(self, physics, target) -> Image.Image:
        self.env.set_physics(physics, target)
        self.renderer.update_scene(self.env.data, camera=self.cam)
        return Image.fromarray(self.renderer.render())


def episode_physics(tr):
    out = [tr["physics0"]]
    for d in tr["decisions"]:
        out.extend(d["committed_physics"][1:])
    return out


def ee(states):
    return np.asarray(states)[..., 0:3]


def puck(states):
    return np.asarray(states)[..., 6:9]


def vis_ee(v):
    return np.asarray(v)[..., 0:3]


def vis_puck(v):
    return np.asarray(v)[..., 3:6]


def _steps_before(tr, upto):
    c = b = 0
    for d in tr["decisions"][:upto]:
        c += len(d["committed"]) - 1
        b += sum(len(x["real"]) - 1 for x in d.get("branches", []))
    return c, b


def _verdict(b):
    return "PASS" if b["passed"] else "FAIL " + "/".join(b["violated"])[:22]


def _callout(d, sc, p, text, offset):
    """Label with a leader line, placed at a pixel offset from the projected point."""
    q = sc.px(p)
    e = (q[0] + offset[0], q[1] + offset[1])
    d.line([tuple(q), e], fill=(255, 255, 255, 220), width=2)
    d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=(255, 255, 255, 255))
    w = d.textlength(text, font=F_SMALL)
    x = e[0] - w - 6 if offset[0] < 0 else e[0] + 6
    d.text((x, e[1] - 10), text, font=F_SMALL, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 220))


def legend(img: Image.Image, y0=H - 178):
    """Compact colour legend for the HUD panel (the planar demo's legend collides with the
    longer arm HUD)."""
    d = ImageDraw.Draw(img)
    x0 = SCENE + 28
    items = [((200, 200, 200), "imagined samples (world model)"), (CYAN, "candidates #1..#4 (#1 = model's pick)"),
             (GREEN, "real branch: PASSED"), (RED, "real branch: FAILED"), (WHITE, "committed plan")]
    for i, (col, txt) in enumerate(items):
        y = y0 + 18 * i
        d.line([(x0, y + 8), (x0 + 24, y + 8)], fill=col, width=4)
        d.text((x0 + 32, y), txt, font=F_SMALL, fill=DIM)


def _label_3d(d, sc, p, text, col, dx=12, dy=-10):
    q = sc.px(p)
    d.text((q[0] + dx, q[1] + dy), text, font=F_SMALL, fill=(*col, 255), stroke_width=2, stroke_fill=(0, 0, 0, 200))


# ---------------------------------------------------------------------------- segments
def scene_intro(v: Video, sc: ArmScene, spec: Spec, rec, seconds=7.0):
    tr = rec["trace"]
    sc.frame_episode(tr, zoom=1.0)
    s0, tgt = tr["history"][0], tr["target"]
    scene = sc.render(tr["physics0"], tgt)
    ov = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    _callout(d, sc, s0[0:3] + [0, 0, 0.10], "Franka Panda arm", (60, -70))
    _callout(d, sc, s0[6:9] + [0, 0, 0.02], "puck", (90, -60))
    _callout(d, sc, np.array([*tgt[:2], 0.0]), "target zone", (-150, 70))
    _callout(d, sc, np.array([*tgt[2:4], 0.06]), "cup: don't move or tip it", (70, 80))
    img = compose(scene, ov)
    lines = ["Task: arm-push-safely (MuJoCo, 6 s, dt = 0.05 s)",
             "Action: 3-D fingertip velocity (gripper closed)", "",
             "STL spec (T = 120 steps):",
             "  F puck_in_target                (liveness)",
             "  G cup_upright: tilt < 0.2 rad   (safety)",
             "  G cup_still: moved < 3 cm      (safety)",
             "  G puck_slow: |v| < 0.25 m/s    (safety)",
             "  G puck_on_table                 (safety)",
             "  G ee_in_workspace               (safety)", "",
             "Every real sim step is counted,",
             "including verification branches."]
    hud(img, "THE TASK", FG, lines)
    v.add(img, int(seconds * FPS))


def decision_walkthrough(v: Video, sc: ArmScene, spec: Spec, rec, k_idx, label, caption=""):
    tr = rec["trace"]
    dec = tr["decisions"][k_idx]
    tgt = tr["target"]
    phys = episode_physics(tr)
    history = tr["history"][:dec["t"] + 1]
    status = clause_status(spec, history, tgt)
    c_before, b_before = _steps_before(tr, k_idx)
    base_scene = sc.render(phys[dec["t"]], tgt)
    imagined = dec["imagined"]  # (K, H+1, 8): ee xyz, puck xyz, cup xy
    K = len(dec["shortlisted"])
    Hh = imagined.shape[1]

    def info(extra=""):
        return [label, f"decision at t = {dec['t']} / {T_MAX}", f"real steps so far this episode: {c_before + b_before}",
                f"   committed {c_before} | branch {b_before}", extra]

    def base_overlay():
        ov = Image.new("RGBA", base_scene.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        if len(history) > 1:
            draw_path(d, sc.px(puck(history)), WHITE, width=3, alpha=160)
        return ov, d

    # IMAGINE
    n1, n2 = int(1.0 * FPS), int(1.6 * FPS)
    img = None
    for f in range(n1 + n2):
        ov, d = base_overlay()
        a = min(1.0, f / n1)
        for g in dec["ghosts"]:
            draw_path(d, sc.px(vis_puck(g)), (215, 215, 215), width=1, alpha=int(80 * a))
            draw_path(d, sc.px(vis_ee(g)), (140, 170, 230), width=1, alpha=int(45 * a))
        if f >= n1:
            frac = (f - n1 + 1) / n2
            upto = max(2, int(frac * Hh))
            for r in range(K - 1, -1, -1):
                col = CAND_COLORS[r]
                draw_path(d, sc.px(vis_puck(imagined[r, :upto])), col, width=4 if r == 0 else 3, alpha=255 if r == 0 else 200)
                draw_path(d, sc.px(vis_ee(imagined[r, :upto])), col, width=2, alpha=160, dashed=True)
                if frac >= 1.0:
                    e = sc.px(vis_ee(imagined[r, -1]))
                    d.text((e[0] + 8, e[1] - 8), f"#{r + 1}", font=F_MED, fill=(*col, 255))
        img = compose(base_scene, ov)
        ex = [(f"#{r + 1} imagined score {dec['imagined_score'][r]:+.3f}  unc {dec['imagined_unc'][r]:.3f}", CAND_COLORS[r])
              for r in range(K)] if f >= n1 else None
        hud(img, "IMAGINE", BLUE, info("CEM in the learned ensemble → top-k plans"), status, caption, ex)
        legend(img)
        v.add(img)
    v.add(img, int(0.6 * FPS))

    branches = dec.get("branches", [])
    done = []
    for bi, br in enumerate(branches):
        real = br["real"]
        for fi, ph in enumerate(br["physics"]):
            scene = sc.render(ph, tgt)
            ov, d = base_overlay()
            for pb in done:
                draw_path(d, sc.px(puck(pb["real"])), GREEN if pb["passed"] else RED, width=3, alpha=150)
            upto = fi + 1
            draw_path(d, sc.px(puck(real[:upto])), WHITE, width=4, alpha=230)
            draw_path(d, sc.px(ee(real[:upto])), WHITE, width=2, alpha=150, dashed=True)
            img = compose(scene, ov)
            hud(img, f"VERIFY branch {bi + 1}/{len(branches)}", AMBER,
                info(f"restore snapshot → run candidate #{br['idx'] + 1} for real"), status, caption,
                [(f"#{pb['idx'] + 1}: {_verdict(pb)}  score {pb['real_J']:+.3f}", GREEN if pb["passed"] else RED)
                 for pb in done])
            legend(img)
            v.add(img)
        done.append(br)
        col = GREEN if br["passed"] else RED
        ov, d = base_overlay()
        for pb in done:
            c2 = GREEN if pb["passed"] else RED
            draw_path(d, sc.px(puck(pb["real"])), c2, width=4, alpha=230)
            draw_path(d, sc.px(ee(pb["real"])), c2, width=2, alpha=150, dashed=True)
        e = sc.px(ee(real[-1]))
        verdict = "PASS" if br["passed"] else "FAIL: " + ", ".join(br["violated"])
        d.text((min(e[0] + 10, SCENE - 300), e[1] + 6), f"#{br['idx'] + 1} {verdict}", font=F_SMALL, fill=(*col, 255),
               stroke_width=2, stroke_fill=(0, 0, 0, 200))
        img = compose(sc.render(br["physics"][-1], tgt), ov)
        hud(img, f"VERIFY branch {bi + 1}/{len(branches)}", AMBER,
            info(f"safety ρ = {br['safety_rob']:+.3f}   reach ρ = {br['reach_rob']:+.3f}"), status, caption,
            [(f"#{pb['idx'] + 1}: {_verdict(pb)}  score {pb['real_J']:+.3f}", GREEN if pb["passed"] else RED)
             for pb in done])
        legend(img)
        v.add(img, int(0.7 * FPS))

    # ACT
    chosen = dec["chosen"]
    seg, seg_phys = dec["committed"], dec["committed_physics"]
    msg = (f"model's #1 overturned → commit #{chosen + 1}" if chosen != 0 and branches
           else f"commit #{chosen + 1} (model's pick{'' if not branches else ', verified'})")
    for fi, ph in enumerate(seg_phys):
        for _ in range(2):
            scene = sc.render(ph, tgt)
            ov, d = base_overlay()
            for pb in branches:
                draw_path(d, sc.px(puck(pb["real"])), GREEN if pb["passed"] else RED, width=2, alpha=90)
            draw_path(d, sc.px(vis_puck(imagined[chosen])), WHITE, width=2, alpha=160, dashed=True)
            draw_path(d, sc.px(puck(seg[:fi + 1])), WHITE, width=5, alpha=255)
            img = compose(scene, ov)
            hist_now = np.concatenate([history, seg[1:fi + 1]]) if fi > 0 else history
            hud(img, "ACT", GREEN, info(msg), clause_status(spec, hist_now, tgt), caption)
            legend(img)
            v.add(img)
    v.add(img, int(0.7 * FPS))


def full_episode(v: Video, sc: ArmScene, spec: Spec, rec, label, caption=""):
    tr = rec["trace"]
    tgt, hist = tr["target"], tr["history"]
    phys = episode_physics(tr)
    dec_at = {}
    for d in tr["decisions"]:
        for t in range(d["t"], d["t"] + len(d["committed"]) - 1):
            dec_at[t] = d
    img = None
    for t, ph in enumerate(phys):
        reps = 2 if t % 2 == 0 else 1  # 1.5 frames per 50 ms step = real time at 30 fps
        scene = sc.render(ph, tgt)
        ov = Image.new("RGBA", scene.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        draw_path(d, sc.px(puck(hist[:t + 1])), WHITE, width=4, alpha=220)
        dd = dec_at.get(t)
        if dd is not None:
            draw_path(d, sc.px(vis_puck(dd["imagined"][dd["chosen"]])), CYAN, width=2, alpha=160, dashed=True)
            for pb in dd.get("branches", []):
                draw_path(d, sc.px(puck(pb["real"])), GREEN if pb["passed"] else RED, width=1, alpha=120)
        c, b = _steps_before(tr, sum(1 for x in tr["decisions"] if x["t"] <= t))
        img = compose(scene, ov)
        hud(img, "EPISODE (real time)", FG,
            [label, f"t = {t} / {T_MAX}", f"real steps used (incl. branches): {c + b}", f"   committed {c} | branch {b}"],
            clause_status(spec, hist[:t + 1], tgt, final=t >= len(hist) - 1), caption)
        v.add(img, reps)
    verdict = "SUCCESS: spec satisfied" if rec["success"] else "FAILED: " + ", ".join(k for k, x in rec["violated"].items() if x)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, SCENE, 60], fill=(0, 0, 0))
    d.text((20, 14), verdict, font=F_BIG, fill=GREEN if rec["success"] else RED)
    v.add(img, int(1.5 * FPS))


def side_by_side(v: Video, spec: Spec, rec_b, rec_c):
    size = 600
    sc = ArmScene(size)
    sc.frame_episode(rec_c["trace"], zoom=1.05)
    tgt = rec_c["trace"]["target"]
    pb, pc = episode_physics(rec_b["trace"]), episode_physics(rec_c["trace"])
    n = max(len(pb), len(pc))
    img = None
    for i in range(n):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for rec, phys, title, x0 in [(rec_b, pb, "B: world-model planning, no verification", 20),
                                     (rec_c, pc, "C: Imagine → Verify → Act (k = 4)", 660)]:
            hist = rec["trace"]["history"]
            t = min(i, len(phys) - 1)
            scene = sc.render(phys[t], tgt).convert("RGBA")
            ov = Image.new("RGBA", scene.size, (0, 0, 0, 0))
            draw_path(ImageDraw.Draw(ov), sc.px(puck(hist[:t + 1])), WHITE, width=4, alpha=220)
            img.paste(Image.alpha_composite(scene, ov).convert("RGB"), (x0, 70))
            d.text((x0, 20), title, font=F_MED, fill=FG)
            final = t >= len(hist) - 1
            speed = float(np.linalg.norm(hist[min(t, len(hist) - 1), 9:11]))
            peak = float(np.linalg.norm(hist[:t + 1, 9:11], axis=-1).max())
            d.text((x0, 676), f"puck speed {speed:.2f} m/s   (peak so far {peak:.2f}, limit 0.25)", font=F_SMALL,
                   fill=RED if peak >= 0.25 else DIM)
            st = clause_status(spec, hist[:t + 1], tgt, final=final)
            bad = [s[0] for s in st if s[1] == "VIOLATED"]
            if final:
                msg = "SUCCESS" if rec["success"] else "FAILED: " + ", ".join(bad)
                steps = rec["committed_steps"] + rec["branch_steps"]
                d.text((x0, 697), f"{msg}   (real steps this episode: {steps})", font=F_SMALL, fill=GREEN if rec["success"] else RED)
            else:
                d.text((x0, 697), f"t = {t}/{T_MAX}   " + ("violated: " + ", ".join(bad) if bad else "no violations yet"),
                       font=F_SMALL, fill=RED if bad else DIM)
        v.add(img, 2)  # slow motion: 2 frames per 50 ms step
    v.add(img, int(2.5 * FPS))


def end_card(v: Video, summary: dict | None, seconds=10.0):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((60, 40), "Arm results (30 held-out episodes per seed, mean ± std over seeds)", font=F_MED, fill=FG)
    if not summary:
        d.text((60, 120), "results/arm_summary.json missing", font=F_MED, fill=RED)
    else:
        cols = ["method", "seeds", "budget", "success", "safety viol.", "real steps / eval ep", "overturned #1"]
        xs = [50, 420, 490, 600, 760, 920, 1120]
        y = 110
        for x, c in zip(xs, cols):
            d.text((x, y), c, font=F_SMALL, fill=DIM)
        y += 34
        short = {"arm_A_ppo": "A: PPO (1M steps)", "arm_A2_sac": "A2: SAC",
                 "arm_B_plan_k1": "B: plan in model, no verify", "arm_C_verify_k4": "C: verify top-4 (k=4)",
                 "arm_C_verify_k4_noverify": "   C's model acting without verify",
                 "arm_E_verify_k2": "E: verify top-2 (k=2)", "arm_G_verify_k4_stop": "G: C + hold still if none pass",
                 "arm_B_plan_k1_long": "B, 2x budget", "arm_C_verify_k4_long": "C, 2x budget"}
        rows = [dict(r, method=short[r["key"]]) for r in summary["table"] if r["key"] in short]
        for r in rows:
            vals = [r["method"], r.get("seeds", "-"), r["budget"], r["success"], r["safety"], r["steps_per_ep"], r.get("overturned", "-")]
            for x, val in zip(xs, vals):
                d.text((x, y), str(val), font=F_SMALL, fill=FG)
            y += 29
        y += 14
        for ln in summary.get("notes", []):
            d.text((60, y), ln, font=F_SMALL, fill=AMBER)
            y += 26
    v.add(img, int(seconds * FPS))


def pick_decisions(tr, n=2):
    """Prefer decisions with mixed pass/fail branches where the puck moves."""
    scored = []
    for i, d in enumerate(tr["decisions"]):
        br = d.get("branches", [])
        mixed = any(b["passed"] for b in br) and any(not b["passed"] for b in br)
        moved = float(np.linalg.norm(d["committed"][-1][6:8] - d["committed"][0][6:8]))
        scored.append((1.0 * mixed + min(moved / 0.04, 1.0), -i, i))
    return sorted(i for *_, i in sorted(scored, reverse=True)[:n])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(TRACES))
    ap.add_argument("--out", default=str(ROOT / "results" / "arm_demo.mp4"))
    args = ap.parse_args()
    traces = Path(args.traces)
    spec = Spec.load("arm_push_safely")
    idx = json.loads((traces / "index.json").read_text())
    load = lambda n: pickle.load(open(traces / n, "rb"))  # noqa: E731
    c_succ = load("c_success.pkl")
    c_ov = load("c_overturn.pkl") if (traces / "c_overturn.pkl").exists() else None
    summary_path = ROOT / "results" / "arm_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    sc = ArmScene()
    label = f"C (k=4), eval seed {c_succ['seed']}"

    v = Video()
    title_card(v, "Imagine → Verify → Act on a robot arm",
               "Franka Panda pushes a puck to a target without disturbing the cup",
               lines=["MuJoCo  •  learned ensemble world model + CEM  •  exact snapshot / restore verification",
                      "the robot imagines plans, tries the best ones for real from a saved state, rewinds, then acts"])
    scene_intro(v, sc, spec, c_succ)
    sc.frame_episode(c_succ["trace"])
    for n_, k in enumerate(pick_decisions(c_succ["trace"], n=2)):
        decision_walkthrough(v, sc, spec, c_succ, k, label,
                             caption=f"Walkthrough {n_ + 1}: imagine in the model, try the top-4 for real from a snapshot, rewind, act.")
    full_episode(v, sc, spec, c_succ, label,
                 caption="Full episode in real time. Faint lines: branches tried at each decision; dashed cyan: chosen imagined plan.")
    if c_ov is not None and idx.get("c_overturn_decision") is not None:
        title_card(v, "When the model is wrong",
                   "the world model's favourite plan breaks a safety rule when tried for real; a verified alternative is used",
                   seconds=2.5)
        sc.frame_episode(c_ov["trace"])
        decision_walkthrough(v, sc, spec, c_ov, idx["c_overturn_decision"], f"C (k=4), eval seed {c_ov['seed']}",
                             caption="Overturn: candidate #1 (the model's pick) violates the spec when actually executed.")
    if (traces / "contrast_c.pkl").exists():
        rb, rc = load("contrast_b.pkl"), load("contrast_c.pkl")
        title_card(v, "Same scene: without vs with verification",
                   f"eval seed {rc['seed']}  (B and C each use their own trained world model; slow motion)", seconds=2.5)
        side_by_side(v, spec, rb, rc)
    plot_slide(v, ROOT / "results" / "plots" / "arm_learning_and_safety.png",
               "Arm: learning curves vs total real steps (incl. branches) and committed safety violations")
    end_card(v, summary)

    out = Path(args.out)
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=16, pixelformat="yuv420p") as w:
        for fr in v.frames:
            w.append_data(fr)
    print(f"wrote {out}: {len(v.frames)} frames, {len(v.frames) / FPS:.1f}s")
    start = int(10 * FPS)
    gif = [np.asarray(Image.fromarray(f).resize((480, 270))) for f in v.frames[start:start + 20 * FPS:3]]
    imageio.mimsave(out.with_suffix(".gif"), gif, duration=0.1, loop=0)
    print(f"wrote arm_demo.gif: {len(gif)} frames")


if __name__ == "__main__":
    main()
