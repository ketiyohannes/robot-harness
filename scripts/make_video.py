"""Render the demo video (results/demo.mp4 + results/demo.gif) from recorded decision traces.

Frames are rendered programmatically: MuJoCo offscreen top-down camera for the scene, PIL for
overlays (imagined ghosts, shortlisted candidates, verified branches, committed plan) and the
HUD panel. Stitched with imageio + imageio-ffmpeg (bundled ffmpeg).

Usage: python scripts/make_video.py   (expects results/demo_traces/*.pkl, results/summary.json)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.push_env import PUSHER_REGION_HALF, PushEnv, top_camera_world_to_pixel  # noqa: E402
from verifier.verifier import Spec  # noqa: E402

W, H = 1280, 720
SCENE = 720
FPS = 30
TRACES = ROOT / "results" / "demo_traces"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")

BG = (18, 20, 26)
FG = (235, 237, 242)
DIM = (150, 156, 170)
BLUE = (90, 170, 255)
AMBER = (255, 190, 70)
GREEN = (70, 220, 120)
RED = (255, 80, 80)
CYAN = (90, 230, 240)
WHITE = (255, 255, 255)
CAND_COLORS = [(90, 230, 240), (190, 140, 255), (255, 150, 210), (170, 200, 120)]


def font(size, bold=False, mono=False):
    try:
        if mono:
            return ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size)
        return ImageFont.truetype(str(FONT_DIR / ("Arial Bold.ttf" if bold else "Arial.ttf")), size)
    except OSError:
        return ImageFont.load_default(size=size)


F_TITLE, F_BIG, F_MED, F_SMALL, F_MONO = font(46, True), font(34, True), font(22), font(17), font(16, mono=True)


# ---------------------------------------------------------------------------- scene rendering
class SceneRenderer:
    def __init__(self, size=SCENE):
        self.env = PushEnv(render_size=(size, size))
        self.env.reset(0)
        self.size = size

    def render(self, state, target):
        e = self.env
        q = e.data.qpos
        q[e._px], q[e._py] = state[0], state[1]
        q[e._bq:e._bq + 3] = state[4:7]
        q[e._bq + 3:e._bq + 7] = [1, 0, 0, 0]
        e.data.qvel[:] = 0
        e.model.site_pos[e._target_site, :2] = target
        mujoco.mj_forward(e.model, e.data)
        return Image.fromarray(e.render())

    def px(self, xy):
        return top_camera_world_to_pixel(np.asarray(xy)[..., :2], size=self.size)


def interp_states(states, sub):
    """Linear interpolation between consecutive states, `sub` frames per step."""
    states = np.asarray(states)
    out = []
    for i in range(len(states) - 1):
        for j in range(sub):
            a = j / sub
            out.append((1 - a) * states[i] + a * states[i + 1])
    out.append(states[-1])
    return out


def draw_path(d: ImageDraw.ImageDraw, pts, color, width=3, alpha=255, dashed=False):
    pts = [tuple(map(float, p)) for p in pts]
    if len(pts) < 2:
        return
    c = (*color, alpha)
    if not dashed:
        d.line(pts, fill=c, width=width, joint="curve")
        return
    for i in range(0, len(pts) - 1, 2):
        d.line(pts[i:i + 2], fill=c, width=width)


def draw_region(d, sr: SceneRenderer):
    a = sr.px(np.array([-PUSHER_REGION_HALF, PUSHER_REGION_HALF]))
    b = sr.px(np.array([PUSHER_REGION_HALF, -PUSHER_REGION_HALF]))
    x0, y0, x1, y1 = a[0], a[1], b[0], b[1]
    n = 40
    for (p, q) in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        for i in range(0, n, 2):
            s = (p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n)
            e = (p[0] + (q[0] - p[0]) * (i + 1) / n, p[1] + (q[1] - p[1]) * (i + 1) / n)
            d.line([s, e], fill=(255, 255, 255, 90), width=2)


# ---------------------------------------------------------------------------- HUD
def clause_status(spec: Spec, history, target, final=False):
    """Status of each clause on the committed trajectory so far."""
    per = spec.batch_clause_robustness(np.asarray(history), target)
    out = []
    for c in spec.clauses:
        r = float(per[c.name])
        if c.kind == "liveness":
            st = "SATISFIED" if r > 0 else ("VIOLATED" if final else "pending")
        else:
            st = "ok" if r > 0 else "VIOLATED"
        out.append((c.name, st, r))
    return out


def hud(img: Image.Image, phase: str, phase_color, lines: list[str], status=None, caption: str = "",
        extra: list[tuple[str, tuple]] | None = None, show_legend: bool = False):
    d = ImageDraw.Draw(img)
    x = SCENE + 28
    d.rectangle([SCENE, 0, W, H], fill=BG)
    d.text((x, 22), "Imagine → Verify → Act", font=F_MED, fill=DIM)
    d.text((x, 56), phase, font=F_BIG, fill=phase_color)
    y = 110
    for ln in lines:
        d.text((x, y), ln, font=F_SMALL, fill=FG)
        y += 25
    if status:
        y += 10
        d.text((x, y), "Spec status (committed trajectory):", font=F_SMALL, fill=DIM)
        y += 26
        for name, st, r in status:
            col = GREEN if st in ("ok", "SATISFIED") else (RED if st == "VIOLATED" else AMBER)
            mark = "✓" if col == GREEN else ("✗" if col == RED else "…")
            d.text((x, y), f"{mark} {name:<17s} {st:<9s} ρ={r:+.3f}", font=F_MONO, fill=col)
            y += 24
    if extra:
        y += 10
        for ln, col in extra:
            d.text((x, y), ln, font=F_MONO, fill=col)
            y += 22
    if show_legend:
    if caption:
        # Wrap caption at the bottom of the panel.
        words, cur, rows = caption.split(), "", []
        for w_ in words:
            t = (cur + " " + w_).strip()
            if d.textlength(t, font=F_SMALL) > W - SCENE - 56:
                rows.append(cur)
                cur = w_
            else:
                cur = t
        rows.append(cur)
        yy = H - 30 - 24 * len(rows)
        for r_ in rows:
            d.text((x, yy), r_, font=F_SMALL, fill=AMBER)
            yy += 24
    return img


def legend(d, y0=H - 250, x0=SCENE + 28):
    """Colour legend, drawn in the HUD panel (above the caption)."""
    items = [((200, 200, 200), "imagined samples (world model)"), (CYAN, "shortlisted candidates (#1 = model's pick)"),
             (GREEN, "real branch: spec PASSED"), (RED, "real branch: spec FAILED"), (WHITE, "committed (executed) plan")]
    y = y0
    for col, txt in items:
        d.line([(x0, y + 9), (x0 + 28, y + 9)], fill=col, width=4)
        d.text((x0 + 38, y), txt, font=F_SMALL, fill=DIM)
        y += 22


def compose(scene: Image.Image, overlay: Image.Image | None):
    base = Image.new("RGB", (W, H), BG)
    sc = scene.convert("RGBA")
    if overlay is not None:
        sc = Image.alpha_composite(sc, overlay)
    base.paste(sc.convert("RGB"), (0, 0))
    return base


# ---------------------------------------------------------------------------- segments
class Video:
    def __init__(self):
        self.frames = []

    def add(self, img, n=1):
        arr = np.asarray(img.convert("RGB"))
        for _ in range(n):
            self.frames.append(arr)


def title_card(v: Video, title, subtitle, seconds=3.0, lines=()):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    tw = d.textlength(title, font=F_TITLE)
    d.text(((W - tw) / 2, 250), title, font=F_TITLE, fill=FG)
    sw = d.textlength(subtitle, font=F_MED)
    d.text(((W - sw) / 2, 320), subtitle, font=F_MED, fill=DIM)
    y = 390
    for ln in lines:
        lw = d.textlength(ln, font=F_SMALL)
        d.text(((W - lw) / 2, y), ln, font=F_SMALL, fill=DIM)
        y += 26
    v.add(img, int(seconds * FPS))


def scene_intro(v: Video, sr, spec, rec, seconds=5.0):
    tr = rec["trace"]
    s0, tgt = tr["history"][0], tr["target"]
    scene = sr.render(s0, tgt)
    ov = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    draw_region(d, sr)
    p, b, t = sr.px(s0[0:2]), sr.px(s0[4:6]), sr.px(tgt)
    d.text((p[0] + 18, p[1] - 10), "pusher", font=F_SMALL, fill=(*WHITE, 255))
    d.text((b[0] + 24, b[1] - 10), "block", font=F_SMALL, fill=(*WHITE, 255))
    d.text((t[0] + 30, t[1] - 10), "target zone", font=F_SMALL, fill=(*WHITE, 255))
    d.text((20, 18), "dashed: allowed pusher region   table edge = block must stay on", font=F_SMALL, fill=(*WHITE, 230))
    img = compose(scene, ov)
    lines = ["Task: push-safely (MuJoCo, 5 s episodes, dt=0.05 s)", "",
             "STL spec (T = 100 steps):",
             "  F[0,T] block_in_target          (liveness)",
             "  G[0,T] pusher_in_region         (safety)",
             "  G[0,T] block_slow: |v|<0.30 m/s (safety)",
             "  G[0,T] block_on_table           (safety)", "",
             "Pass = every clause robustness ρ > 0",
             "Every real sim step is counted,",
             "including verification branches."]
    hud(img, "THE TASK", FG, lines)
    v.add(img, int(seconds * FPS))


def _cum_steps_before(tr, upto):
    c = b = 0
    for d in tr["decisions"][:upto]:
        c += len(d["committed"]) - 1
        b += sum(len(x["real"]) - 1 for x in d.get("branches", []))
    return c, b


def decision_walkthrough(v: Video, sr, spec, rec, k_idx, label, caption=""):
    tr = rec["trace"]
    dec = tr["decisions"][k_idx]
    tgt = tr["target"]
    s0 = dec["state"]
    hist_len = dec["t"] + 1
    history = tr["history"][:hist_len]
    status = clause_status(spec, history, tgt)
    c_before, b_before = _cum_steps_before(tr, k_idx)
    base_scene = sr.render(s0, tgt)
    imagined = dec["imagined"]  # (K, H+1, 4) [px,py,bx,by]
    shortlisted = dec["shortlisted"]
    K = len(shortlisted)
    Hh = imagined.shape[1]

    def info_lines(extra=""):
        return [f"{label}", f"decision at t = {dec['t']} / 100", f"real steps so far this episode: {c_before + b_before}",
                f"   committed {c_before} | branch {b_before}", extra]

    hist_px = sr.px(history[:, 0:2]), sr.px(history[:, 4:6])

    def base_overlay():
        ov = Image.new("RGBA", base_scene.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        draw_region(d, sr)
        if len(history) > 1:
            draw_path(d, hist_px[1], WHITE, width=3, alpha=160)
        return ov, d

    # IMAGINE: ghosts fade in, then candidates grow.
    n1, n2 = int(1.0 * FPS), int(1.6 * FPS)
    for f in range(n1 + n2):
        ov, d = base_overlay()
        a = min(1.0, f / n1)
        for g in dec["ghosts"]:
            draw_path(d, sr.px(g[:, 2:4]), (210, 210, 210), width=1, alpha=int(70 * a))
            draw_path(d, sr.px(g[:, 0:2]), (140, 170, 230), width=1, alpha=int(40 * a))
        if f >= n1:
            frac = (f - n1 + 1) / n2
            upto = max(2, int(frac * Hh))
            for r in range(K - 1, -1, -1):
                col = CAND_COLORS[r]
                draw_path(d, sr.px(imagined[r, :upto, 2:4]), col, width=4 if r == 0 else 3, alpha=255 if r == 0 else 200)
                draw_path(d, sr.px(imagined[r, :upto, 0:2]), col, width=2, alpha=150, dashed=True)
                if frac >= 1.0:
                    e = sr.px(imagined[r, -1, 2:4])
                    d.text((e[0] + 8, e[1] - 8), f"#{r + 1}", font=F_MED, fill=(*col, 255))
        img = compose(base_scene, ov)
        ex = [(f"#{r + 1} imagined score {dec['imagined_score'][r]:+.3f}  unc {dec['imagined_unc'][r]:.3f}", CAND_COLORS[r])
              for r in range(K)] if f >= n1 else None
        hud(img, "IMAGINE", BLUE, info_lines("CEM in the learned ensemble → top-k plans"), status, caption, ex, show_legend=True)
        v.add(img)
    v.add(img, int(0.6 * FPS))

    branches = dec.get("branches", [])
    done = []
    for bi, br in enumerate(branches):
        real = br["real"]
        frames = interp_states(real, 2)
        for fi, st in enumerate(frames):
            scene = sr.render(st, tgt)
            ov, d = base_overlay()
            for r in range(K):
                draw_path(d, sr.px(imagined[r, :, 2:4]), CAND_COLORS[r], width=2, alpha=70)
            for pb in done:
                col = GREEN if pb["passed"] else RED
                draw_path(d, sr.px(pb["real"][:, 4:6]), col, width=3, alpha=150)
            upto = min(len(real), fi // 2 + 1)
            draw_path(d, sr.px(real[:upto, 4:6]), WHITE, width=4, alpha=230)
            draw_path(d, sr.px(real[:upto, 0:2]), WHITE, width=2, alpha=140, dashed=True)
                img = compose(scene, ov)
            hud(img, f"VERIFY branch {bi + 1}/{len(branches)}", AMBER,
                info_lines(f"restore snapshot → run candidate #{br['idx'] + 1} for real"), status, caption,
                [(f"#{pb['idx'] + 1}: {'PASS' if pb['passed'] else 'FAIL'}  real score {pb['real_J']:+.3f}", GREEN if pb['passed'] else RED)
                 for pb in done], show_legend=True)
            v.add(img)
        done.append(br)
        col = GREEN if br["passed"] else RED
        ov, d = base_overlay()
        for pb in done:
            c2 = GREEN if pb["passed"] else RED
            draw_path(d, sr.px(pb["real"][:, 4:6]), c2, width=4, alpha=230)
            draw_path(d, sr.px(pb["real"][:, 0:2]), c2, width=2, alpha=140, dashed=True)
        e = sr.px(real[-1, 4:6])
        verdict = "PASS" if br["passed"] else "FAIL: " + ", ".join(br["violated"])
        d.text((min(e[0] + 10, SCENE - 260), e[1] + 6), f"#{br['idx'] + 1} {verdict}", font=F_SMALL, fill=(*col, 255))
        img = compose(sr.render(real[-1], tgt), ov)
        hud(img, f"VERIFY branch {bi + 1}/{len(branches)}", AMBER,
            info_lines(f"safety ρ = {br['safety_rob']:+.3f}   reach ρ = {br['reach_rob']:+.3f}"), status, caption,
            [(f"#{pb['idx'] + 1}: {'PASS' if pb['passed'] else 'FAIL'}  real score {pb['real_J']:+.3f}", GREEN if pb['passed'] else RED)
             for pb in done], show_legend=True)
        v.add(img, int(0.5 * FPS))

    # ACT: highlight the chosen plan, then execute the committed segment.
    chosen = dec["chosen"]
    seg = dec["committed"]
    msg = (f"model's #1 overturned → commit #{chosen + 1}" if chosen != 0 and branches
           else f"commit #{chosen + 1} (model's pick{'' if not branches else ', verified'})")
    frames = interp_states(seg, 3)
    for fi, st in enumerate(frames):
        scene = sr.render(st, tgt)
        ov, d = base_overlay()
        for pb in branches:
            c2 = GREEN if pb["passed"] else RED
            draw_path(d, sr.px(pb["real"][:, 4:6]), c2, width=2, alpha=90)
        draw_path(d, sr.px(imagined[chosen, :, 2:4]), WHITE, width=2, alpha=160, dashed=True)
        upto = min(len(seg), fi // 3 + 1)
        draw_path(d, sr.px(seg[:upto, 4:6]), WHITE, width=5, alpha=255)
        img = compose(scene, ov)
        hist_now = np.concatenate([history, seg[1:upto]]) if upto > 1 else history
        hud(img, "ACT", GREEN, info_lines(msg), clause_status(spec, hist_now, tgt), caption, show_legend=True)
        v.add(img)
    v.add(img, int(0.7 * FPS))


def full_episode(v: Video, sr, spec, rec, label, caption="", speed=1.0):
    tr = rec["trace"]
    tgt = tr["target"]
    hist = tr["history"]
    # Map step -> decision (for drawing the current imagined chosen plan).
    dec_at = {}
    for d in tr["decisions"]:
        for t in range(d["t"], d["t"] + len(d["committed"]) - 1):
            dec_at[t] = d
    sub = max(1, int(round(1.5 / speed)))
    frames = interp_states(hist, sub)
    real_used = 0
    for fi, st in enumerate(frames):
        t = min(fi // sub, len(hist) - 1)
        scene = sr.render(st, tgt)
        ov = Image.new("RGBA", scene.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        draw_region(d, sr)
        draw_path(d, sr.px(hist[:t + 1, 4:6]), WHITE, width=4, alpha=220)
        dd = dec_at.get(t)
        if dd is not None:
            draw_path(d, sr.px(dd["imagined"][dd["chosen"], :, 2:4]), CYAN, width=2, alpha=160, dashed=True)
            for pb in dd.get("branches", []):
                draw_path(d, sr.px(pb["real"][:, 4:6]), GREEN if pb["passed"] else RED, width=1, alpha=110)
        c, b = _cum_steps_before(tr, sum(1 for x in tr["decisions"] if x["t"] <= t))
        img = compose(scene, ov)
        final = t >= len(hist) - 1
        hud(img, "EPISODE (1x speed)" if speed == 1.0 else "EPISODE", FG,
            [label, f"t = {t} / 100", f"real steps used (incl. branches): {c + b}", f"   committed {c} | branch {b}"],
            clause_status(spec, hist[:t + 1], tgt, final=final), caption)
        v.add(img)
    verdict = "SUCCESS: spec satisfied" if rec["success"] else "FAILED: " + ", ".join(k for k, x in rec["violated"].items() if x)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, SCENE, 60], fill=(0, 0, 0))
    d.text((20, 14), verdict, font=F_BIG, fill=GREEN if rec["success"] else RED)
    v.add(img, int(1.5 * FPS))


def side_by_side(v: Video, spec, rec_b, rec_c, sub=2):
    size = 600
    srs = SceneRenderer(size)
    trb, trc = rec_b["trace"], rec_c["trace"]
    tgt = trc["target"]
    hb, hc = trb["history"], trc["history"]
    fb, fc = interp_states(hb, sub), interp_states(hc, sub)
    n = max(len(fb), len(fc))
    for i in range(n + int(2.0 * FPS)):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for j, (rec, frames, hist, title, x0) in enumerate([
                (rec_b, fb, hb, "B: world-model planning, no verification", 20),
                (rec_c, fc, hc, "C: Imagine → Verify → Act (k = 4)", 660)]):
            fi = min(i, len(frames) - 1)
            t = min(fi // sub, len(hist) - 1)
            sc = srs.render(frames[fi], tgt).convert("RGBA")
            ov = Image.new("RGBA", sc.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            draw_path(od, srs.px(hist[:t + 1, 4:6]), WHITE, width=4, alpha=220)
            img.paste(Image.alpha_composite(sc, ov).convert("RGB"), (x0, 70))
            d.text((x0, 20), title, font=F_MED, fill=FG)
            st = clause_status(spec, hist[:t + 1], tgt, final=t >= len(hist) - 1)
            bad = [s for s in st if s[1] == "VIOLATED"]
            steps = rec["committed_steps"] + rec["branch_steps"] if t >= len(hist) - 1 else None
            if t >= len(hist) - 1:
                msg = "SUCCESS" if rec["success"] else "FAILED: " + ", ".join(s[0] for s in bad)
                col = GREEN if rec["success"] else RED
                d.text((x0, 680), f"{msg}   (real steps this episode: {steps})", font=F_SMALL, fill=col)
            else:
                d.text((x0, 680), f"t = {t}/100   " + ("violations: " + ", ".join(s[0] for s in bad) if bad else "no violations yet"),
                       font=F_SMALL, fill=RED if bad else DIM)
        v.add(img)


def end_card(v: Video, summary: dict | None, seconds=7.0):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((60, 40), "Results (push-safely, 30 held-out episodes, mean ± std over seeds)", font=F_MED, fill=FG)
    if not summary:
        d.text((60, 120), "results/summary.json missing", font=F_MED, fill=RED)
    else:
        rows = summary["table"]
        cols = ["method", "train budget", "success", "safety viol.", "real steps / eval ep", "overturned #1"]
        xs = [60, 420, 600, 760, 930, 1130]
        y = 110
        for x, c in zip(xs, cols):
            d.text((x, y), c, font=F_SMALL, fill=DIM)
        y += 34
        for r in rows:
            vals = [r["method"], r["budget"], r["success"], r["safety"], r["steps_per_ep"], r.get("overturned", "-")]
            for x, val in zip(xs, vals):
                d.text((x, y), str(val), font=F_SMALL, fill=FG)
            y += 32
        y += 20
        for ln in summary.get("notes", []):
            d.text((60, y), ln, font=F_SMALL, fill=AMBER)
            y += 26
    v.add(img, int(seconds * FPS))


def main():
    spec = Spec.load("push_safely")
    sr = SceneRenderer()
    idx = json.loads((TRACES / "index.json").read_text())
    load = lambda n: pickle.load(open(TRACES / n, "rb"))  # noqa: E731
    c_succ = load("c_success.pkl")
    b_same = load("b_same_seed.pkl")
    c_ov = load("c_overturn.pkl") if (TRACES / "c_overturn.pkl").exists() else None
    summary_path = ROOT / "results" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

    v = Video()
    title_card(v, "Imagine → Verify → Act", "world-model search + STL-verified branching",
               lines=["MuJoCo planar push task  •  ensemble world model + CEM  •  exact snapshot/restore",
                      "overnight MVP: small models, short budgets, few seeds"])
    scene_intro(v, sr, spec, c_succ)
    ndec = len(c_succ["trace"]["decisions"])
    picks = [0, min(2, ndec - 1)]
    for n_, k in enumerate(picks):
        decision_walkthrough(v, sr, spec, c_succ, k, f"C (k=4), eval seed {c_succ['seed']}",
                             caption=f"Decision {n_ + 1} of the walkthrough: imagine in the model, verify the top-4 for real from a snapshot, act.")
    full_episode(v, sr, spec, c_succ, f"C (k=4), eval seed {c_succ['seed']}",
                 caption="Full episode at real-time speed. Faint lines: branches tried at each decision; dashed cyan: chosen imagined plan.")
    if c_ov is not None and idx.get("c_overturn_decision") is not None:
        title_card(v, "When the model is wrong", "the world model's favourite plan fails verification; a verified alternative is committed",
                   seconds=2.5)
        decision_walkthrough(v, sr, spec, c_ov, idx["c_overturn_decision"], f"C (k=4), eval seed {c_ov['seed']}",
                             caption="Overturn: candidate #1 (the model's pick) violates the spec when actually executed.")
    contrast = (TRACES / "c_vs_b_contrast_c.pkl").exists()
    rb, rc = (load("c_vs_b_contrast_b.pkl"), load("c_vs_b_contrast_c.pkl")) if contrast else (b_same, c_succ)
    title_card(v, "Same seed: without vs with verification", f"eval seed {rc['seed']}  (B and C each use their own trained world model)", seconds=2.5)
    side_by_side(v, spec, rb, rc)
    end_card(v, summary)

    out = ROOT / "results" / "demo.mp4"
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=16,
                            pixelformat="yuv420p") as w:
        for fr in v.frames:
            w.append_data(fr)
    print(f"wrote {out}: {len(v.frames)} frames, {len(v.frames) / FPS:.1f}s")
    # GIF backup: the first decision walkthrough region, downscaled, 10 fps.
    start = int(8 * FPS)
    gif_frames = [np.asarray(Image.fromarray(f).resize((480, 270))) for f in v.frames[start:start + 20 * FPS:3]]
    imageio.mimsave(ROOT / "results" / "demo.gif", gif_frames, duration=0.1, loop=0)
    print(f"wrote demo.gif: {len(gif_frames)} frames")


if __name__ == "__main__":
    main()
