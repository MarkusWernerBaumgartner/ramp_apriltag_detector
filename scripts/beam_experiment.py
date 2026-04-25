#!/usr/bin/env python3
"""
Beam detection accuracy experiment.

Records raw AprilTag detections and pre-filter beam poses to characterise
visual detection quality. Designed to compare tag families (e.g. tag25h9
vs tag16h5) by re-running after updating config/settings.yaml.

Results are saved to <package>/experiment_results/<family>_<timestamp>.yaml.
"""

import os
import yaml
from datetime import datetime

import numpy as np
import rospy
import rospkg
import tf2_ros
from scipy.spatial.transform import Rotation as R


# ─────────────────────────────────────────────────────────────────────────────
# TF helpers
# ─────────────────────────────────────────────────────────────────────────────

def lookup_tf(buf, parent, child, timeout=0.3):
    """Return (pos np.array[3], quat np.array[4]) or None."""
    try:
        t = buf.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(timeout))
        tr, ro = t.transform.translation, t.transform.rotation
        return np.array([tr.x, tr.y, tr.z]), np.array([ro.x, ro.y, ro.z, ro.w])
    except Exception:
        return None


def step_rotation_deg(q1, q2):
    """Minimum rotation angle in degrees from orientation q1 to q2."""
    return float(np.degrees((R.from_quat(q2) * R.from_quat(q1).inv()).magnitude()))


def euler_deg(q):
    return R.from_quat(q).as_euler('xyz', degrees=True)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot
# ─────────────────────────────────────────────────────────────────────────────

def take_snapshot(buf, parent, tag1_name, tag2_name, beam_name):
    """Single TF lookup for both raw tags and derived beam pose."""
    t1 = lookup_tf(buf, parent, tag1_name)
    t2 = lookup_tf(buf, parent, tag2_name)
    bm = lookup_tf(buf, parent, beam_name)
    return {
        'tag1': {'pos': t1[0].tolist(), 'quat': t1[1].tolist()} if t1 else None,
        'tag2': {'pos': t2[0].tolist(), 'quat': t2[1].tolist()} if t2 else None,
        'beam': {'pos': bm[0].tolist(), 'quat': bm[1].tolist()} if bm else None,
        'inter_tag_m': float(np.linalg.norm(t1[0] - t2[0])) if (t1 and t2) else None,
    }


def print_snapshot(s, d_truth_m):
    if s['inter_tag_m'] is not None:
        d_mm = s['inter_tag_m'] * 1000
        print(f"    inter-tag : {d_mm:.2f} mm  (err {d_mm - d_truth_m * 1000:+.2f} mm vs {d_truth_m*1000:.1f} mm truth)")
    else:
        print("    inter-tag : N/A  (one or both tags not visible)")

    if s['beam'] is not None:
        p = s['beam']['pos']
        e = euler_deg(s['beam']['quat'])
        print(f"    beam pos  : x={p[0]*1000:.1f}  y={p[1]*1000:.1f}  z={p[2]*1000:.1f} mm")
        print(f"    beam rot  : r={e[0]:.1f}  p={e[1]:.1f}  y={e[2]:.1f} deg")
    else:
        print("    beam pos  : NOT VISIBLE")


def pf(value, target):
    return "PASS" if value <= target else "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# Experiment class
# ─────────────────────────────────────────────────────────────────────────────

class BeamExperiment:

    TARGET_POS_MM  = 3.0
    TARGET_ROT_DEG = 3.0

    def __init__(self, buf, tag_family, beam_name, tag1_name, tag2_name, d_truth_m):
        self.buf        = buf
        self.tag_family = tag_family
        self.beam       = beam_name
        self.tag1       = tag1_name
        self.tag2       = tag2_name
        self.d_truth_m  = d_truth_m
        self.parent     = 'base_link'

        self.results = {
            'tag_family'          : tag_family,
            'inter_tag_truth_mm'  : d_truth_m * 1000,
            'experiments'         : {},
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    def _snap(self):
        return take_snapshot(self.buf, self.parent, self.tag1, self.tag2, self.beam)

    def _beam_pos(self, s):
        return np.array(s['beam']['pos']) if s['beam'] else None

    def _beam_quat(self, s):
        return np.array(s['beam']['quat']) if s['beam'] else None

    @staticmethod
    def _prompt(msg="  Press Enter to record..."):
        try:
            input(msg)
        except (EOFError, KeyboardInterrupt):
            rospy.signal_shutdown("user exit")

    @staticmethod
    def _sep(title=""):
        w = 58
        if title:
            print(f"\n{'─'*3} {title} {'─'*(w - len(title) - 5)}")
        else:
            print("─" * w)

    # ── Experiment 1: Repeatability ──────────────────────────────────────────

    def run_repeatability(self, n=20):
        self._sep("EXPERIMENT 1: REPEATABILITY")
        print(f"  Return the beam to reference position P0 {n} times.")
        print("  Press Enter each time to record.\n")
        self._prompt("  Place beam at P0, press Enter for first recording...")

        snapshots = []
        for i in range(n):
            s = self._snap()
            print(f"\n  [{i+1}/{n}]")
            print_snapshot(s, self.d_truth_m)
            snapshots.append(s)
            if i < n - 1:
                self._prompt(f"\n  Move away, return to P0, press Enter [{i+2}/{n}]...")

        metrics = self._repeatability_metrics(snapshots)
        self._print_repeatability(metrics)
        self.results['experiments']['repeatability'] = {
            'snapshots': snapshots,
            'metrics'  : metrics,
        }

    def _repeatability_metrics(self, snaps):
        pos   = np.array([s['beam']['pos']  for s in snaps if s['beam']])
        quats = np.array([s['beam']['quat'] for s in snaps if s['beam']])
        dists = np.array([s['inter_tag_m'] * 1000 for s in snaps if s['inter_tag_m'] is not None])
        m     = {'n_valid': len(pos), 'n_total': len(snaps)}
        if len(pos) >= 2:
            m['pos_std_mm']     = (np.std(pos, axis=0) * 1000).tolist()
            m['pos_rmse_mm']    = float(np.sqrt(np.mean(np.var(pos, axis=0))) * 1000)
            m['rot_std_deg']    = np.std(np.array([euler_deg(q) for q in quats]), axis=0).tolist()
            m['rot_max_std_deg']= float(max(m['rot_std_deg']))
        if len(dists):
            m['inter_tag_mean_mm'] = float(np.mean(dists))
            m['inter_tag_std_mm']  = float(np.std(dists))
            m['inter_tag_err_mm']  = float(np.mean(dists) - self.d_truth_m * 1000)
        return m

    def _print_repeatability(self, m):
        self._sep("Results")
        n_v, n_t = m['n_valid'], m['n_total']
        print(f"  Valid recordings : {n_v}/{n_t}")
        if 'pos_std_mm' in m:
            s = m['pos_std_mm']
            rmse = m['pos_rmse_mm']
            print(f"  Position std     : x={s[0]:.2f}  y={s[1]:.2f}  z={s[2]:.2f} mm")
            print(f"  Position RMSE    : {rmse:.2f} mm  →  {pf(rmse, self.TARGET_POS_MM)}")
        if 'rot_std_deg' in m:
            s = m['rot_std_deg']
            mx = m['rot_max_std_deg']
            print(f"  Rotation std     : r={s[0]:.2f}  p={s[1]:.2f}  y={s[2]:.2f} deg")
            print(f"  Rotation max std : {mx:.2f} deg  →  {pf(mx, self.TARGET_ROT_DEG)}")
        if 'inter_tag_mean_mm' in m:
            print(f"  Inter-tag dist   : mean={m['inter_tag_mean_mm']:.2f}  "
                  f"std={m['inter_tag_std_mm']:.2f}  "
                  f"err={m['inter_tag_err_mm']:+.2f} mm")

    # ── Experiment 2: Translation ─────────────────────────────────────────────

    def run_translation(self, axis='x'):
        ax_idx   = {'x': 0, 'y': 1}[axis]
        steps_mm = [5, 10, 20, 30, 50, 100]
        self._sep(f"EXPERIMENT 2: TRANSLATION ({axis.upper()})")
        print(f"  Steps (mm): {steps_mm}")
        print("  Return to P0 between each step.\n")
        self._prompt("  Place beam at P0, press Enter for reference...")

        ref = self._snap()
        print("\n  Reference:")
        print_snapshot(ref, self.d_truth_m)

        if self._beam_pos(ref) is None:
            print("\n  ERROR: beam not visible at reference. Aborting.")
            return

        records = []
        for d_mm in steps_mm:
            self._prompt(f"\n  Translate +{d_mm} mm along {axis.upper()}, press Enter...")
            s = self._snap()
            print(f"\n  At +{d_mm} mm:")
            print_snapshot(s, self.d_truth_m)

            self._prompt("  Return to P0, press Enter...")
            s_ret = self._snap()
            print("\n  P0 return:")
            print_snapshot(s_ret, self.d_truth_m)
            records.append({'commanded_mm': d_mm, 'snap': s, 'snap_p0_return': s_ret})

        metrics = self._translation_metrics(ref, records, ax_idx)
        self._print_translation(metrics, axis)
        self.results['experiments'][f'translation_{axis}'] = {
            'reference': ref,
            'records'  : records,
            'metrics'  : metrics,
        }

    def _translation_metrics(self, ref, records, ax_idx):
        ref_pos = self._beam_pos(ref)
        rows = []
        for r in records:
            pos = self._beam_pos(r['snap'])
            if pos is None or ref_pos is None:
                rows.append({'commanded_mm': r['commanded_mm'], 'visible': False})
                continue
            disp_mm = (pos - ref_pos) * 1000
            cross_mm = float(np.max(np.abs(np.delete(disp_mm, ax_idx))))
            rows.append({
                'commanded_mm'    : float(r['commanded_mm']),
                'measured_mm'     : float(disp_mm[ax_idx]),
                'error_mm'        : float(disp_mm[ax_idx] - r['commanded_mm']),
                'cross_axis_mm'   : cross_mm,
                'inter_tag_mm'    : r['snap']['inter_tag_m'] * 1000 if r['snap']['inter_tag_m'] else None,
                'visible'         : True,
            })
        errors = [abs(r['error_mm']) for r in rows if r.get('visible')]
        rmse   = float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else None
        return {'per_step': rows, 'rmse_mm': rmse}

    def _print_translation(self, m, axis):
        self._sep("Results")
        print(f"  {'Commanded':>11}  {'Measured':>10}  {'Error':>8}  {'Cross-axis':>12}")
        for r in m['per_step']:
            if not r.get('visible'):
                print(f"  {r['commanded_mm']:>9.0f} mm   {'N/A':>10}  {'N/A':>8}  {'N/A':>12}")
            else:
                print(f"  {r['commanded_mm']:>9.0f} mm  {r['measured_mm']:>9.1f} mm"
                      f"  {r['error_mm']:>+7.2f} mm  {r['cross_axis_mm']:>10.2f} mm")
        if m['rmse_mm'] is not None:
            print(f"\n  Translation RMSE : {m['rmse_mm']:.2f} mm  →  {pf(m['rmse_mm'], self.TARGET_POS_MM)}")

    # ── Experiment 3: Rotation ────────────────────────────────────────────────

    def run_rotation(self):
        self._sep("EXPERIMENT 3: ROTATION (90° STEPS)")
        print("  Rotate: 0° → 90° → 180° → 270° → back to 0° (360°)")
        print("  Each step is a 90° rotation. Press Enter at each position.\n")

        angles  = [0, 90, 180, 270, 360]
        entries = []
        for angle in angles:
            if   angle ==   0: self._prompt("  Set beam to reference orientation (0°), press Enter...")
            elif angle == 360: self._prompt("\n  Rotate back to 0° (full 360°), press Enter...")
            else:              self._prompt(f"\n  Rotate to {angle}°, press Enter...")
            s = self._snap()
            print(f"\n  At {angle}°:")
            print_snapshot(s, self.d_truth_m)
            entries.append({'angle': angle, 'snap': s})

        metrics = self._rotation_metrics(entries)
        self._print_rotation(metrics)
        self.results['experiments']['rotation'] = {
            'entries': entries,
            'metrics': metrics,
        }

    def _rotation_metrics(self, entries):
        rows   = []
        q_prev = None
        for e in entries:
            q     = self._beam_quat(e['snap'])
            angle = e['angle']
            if q is None:
                rows.append({'angle': angle, 'visible': False})
                q_prev = None
                continue
            if q_prev is None:
                rows.append({'angle': angle, 'step_deg': None, 'step_err_deg': None, 'visible': True})
            else:
                step = step_rotation_deg(q_prev, q)
                rows.append({
                    'angle'        : angle,
                    'step_deg'     : float(step),
                    'step_err_deg' : float(step - 90.0),
                    'visible'      : True,
                })
            q_prev = q

        q0     = self._beam_quat(entries[0]['snap'])
        q_last = self._beam_quat(entries[-1]['snap'])
        closure = step_rotation_deg(q0, q_last) if (q0 is not None and q_last is not None) else None

        step_errors = [abs(r['step_err_deg']) for r in rows if r.get('visible') and r['step_err_deg'] is not None]
        rmse = float(np.sqrt(np.mean(np.array(step_errors) ** 2))) if step_errors else None

        return {'per_angle': rows, 'closure_err_deg': closure, 'rmse_deg': rmse}

    def _print_rotation(self, m):
        self._sep("Results")
        print(f"  {'Target':>8}  {'Step':>8}  {'Step error':>11}")
        for r in m['per_angle']:
            if not r.get('visible'):
                print(f"  {r['angle']:>6}°    {'N/A':>8}  {'N/A':>11}")
            elif r['step_deg'] is None:
                print(f"  {r['angle']:>6}°    {'ref':>8}  {'—':>11}")
            else:
                print(f"  {r['angle']:>6}°   {r['step_deg']:>7.2f}°  {r['step_err_deg']:>+10.2f}°")
        if m['closure_err_deg'] is not None:
            print(f"\n  Closure error  : {m['closure_err_deg']:.2f}°  (360° return to 0°)")
        if m['rmse_deg'] is not None:
            print(f"  Rotation RMSE  : {m['rmse_deg']:.2f}°  →  {pf(m['rmse_deg'], self.TARGET_ROT_DEG)}")

    # ── Inter-tag summary across all experiments ──────────────────────────────

    def print_inter_tag_summary(self):
        all_dists = []
        for exp in self.results['experiments'].values():
            for key in ('snapshots', 'entries'):
                for item in exp.get(key, []):
                    s = item if 'inter_tag_m' in item else item.get('snap')
                    if s and s.get('inter_tag_m') is not None:
                        all_dists.append(s['inter_tag_m'] * 1000)
            for r in exp.get('records', []):
                s = r.get('snap')
                if s and s.get('inter_tag_m') is not None:
                    all_dists.append(s['inter_tag_m'] * 1000)

        if not all_dists:
            return
        arr = np.array(all_dists)
        self._sep("Inter-tag distance summary (all experiments)")
        print(f"  N        : {len(arr)}")
        print(f"  Mean     : {np.mean(arr):.2f} mm  (truth {self.d_truth_m*1000:.1f} mm)")
        print(f"  Std      : {np.std(arr):.2f} mm")
        print(f"  Bias     : {np.mean(arr) - self.d_truth_m*1000:+.2f} mm")
        print(f"  Max err  : {np.max(np.abs(arr - self.d_truth_m*1000)):.2f} mm")

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(self):
        pkg = rospkg.RosPack().get_path('ramp_apriltag_detector')
        out = os.path.join(pkg, 'experiment_results')
        os.makedirs(out, exist_ok=True)
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(out, f'{self.tag_family}_{ts}.yaml')
        with open(path, 'w') as f:
            yaml.dump(self.results, f, default_flow_style=False)
        print(f"\n  Results saved → {path}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rospy.init_node('beam_experiment', anonymous=False)

    tag_family  = rospy.get_param('~tag_family',            'tag25h9')
    beam_name   = rospy.get_param('~beam_name',             'beam1')
    tag1_name   = rospy.get_param('~tag1_name',             't1_beam1')
    tag2_name   = rospy.get_param('~tag2_name',             't2_beam1')
    d_truth_mm  = rospy.get_param('~inter_tag_distance_mm', 35.0)
    n_reps      = rospy.get_param('~n_repeatability',       20)

    w = 58
    print("\n" + "=" * w)
    print("  BEAM DETECTION ACCURACY EXPERIMENT")
    print("=" * w)
    print(f"  Tag family        : {tag_family}")
    print(f"  Beam frame        : {beam_name}")
    print(f"  Tag frames        : {tag1_name},  {tag2_name}")
    print(f"  Inter-tag truth   : {d_truth_mm} mm  (physical measurement)")
    print(f"  Repeatability N   : {n_reps}")
    print("=" * w)
    print("\n  Waiting for TF...")

    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf)
    rospy.sleep(2.0)

    # Quick visibility check
    for name in [tag1_name, tag2_name, beam_name]:
        r = lookup_tf(buf, 'base_link', name, timeout=0.5)
        status = "visible" if r is not None else "NOT VISIBLE"
        print(f"  {name:<30} {status}")
    print()

    exp = BeamExperiment(buf, tag_family, beam_name, tag1_name, tag2_name, d_truth_mm / 1000.0)

    menu = (
        "\nSelect experiment:\n"
        "  1   Repeatability\n"
        "  2   Translation X\n"
        "  3   Translation Y\n"
        "  4   Rotation (90° steps)\n"
        "  5   Run all\n"
        "  s   Save + quit\n"
        "  q   Quit\n"
    )

    while not rospy.is_shutdown():
        print(menu)
        try:
            c = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if   c == '1': exp.run_repeatability(n_reps)
        elif c == '2': exp.run_translation('x')
        elif c == '3': exp.run_translation('y')
        elif c == '4': exp.run_rotation()
        elif c == '5':
            exp.run_repeatability(n_reps)
            exp.run_translation('x')
            exp.run_translation('y')
            exp.run_rotation()
            exp.print_inter_tag_summary()
            exp.save()
            break
        elif c == 's':
            exp.print_inter_tag_summary()
            exp.save()
            break
        elif c == 'q':
            break
        else:
            print("  Unknown option.")

    print("\nDone.")


if __name__ == '__main__':
    main()
