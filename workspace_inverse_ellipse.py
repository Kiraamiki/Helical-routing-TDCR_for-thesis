#!/usr/bin/env python3
# workspace_inverse_ellipse.py
"""
8-point discrete ellipse inverse trajectory generator.

This script uses sim_only.py only. It samples the simulated workspace, selects a
cable0-biased 8-point ellipse, inversely solves tendon commands for each waypoint,
and exports a CSV for discrete step-hold validation.

Run in your project folder:
    cd ~/桌面/'thesis project'/twin_vs_real_1
    python3 workspace_inverse_ellipse.py --cable0-bias 0.65 --major-mm 60 --minor-mm 35
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import sim_only as SIM
except Exception as e:
    print('[ERROR] Could not import sim_only.py from current folder.')
    print('        Put this script in the same folder as sim_only.py.')
    print('        Import error:', e)
    sys.exit(1)


def get_params():
    if not hasattr(SIM, 'RobotParams'):
        raise RuntimeError('sim_only.py does not provide RobotParams()')
    return SIM.RobotParams()


def default_delta_list(params, seed=0):
    if hasattr(SIM, 'random_delta_list'):
        try:
            total_angle = float(getattr(params, 'helix_angle', 6.0 * math.pi))
            return SIM.random_delta_list(params.num_disks, total_angle=total_angle,
                                         deg_min=30.0, deg_max=60.0, seed=int(seed))
        except Exception:
            pass
    n_disks = int(getattr(params, 'num_disks', 6))
    total_angle = float(getattr(params, 'helix_angle', 6.0 * math.pi))
    return [total_angle / max(1, n_disks - 1)] * max(1, n_disks - 1)


def solve_tip(cmd_mm, params, delta_list, enforce_no_twist=True, mu_friction=0.8):
    dL_m = [float(x) * 1e-3 for x in cmd_mm]
    try:
        out = SIM.solve_shape_from_dL(dL_m, params, delta_list=delta_list,
                                      enforce_no_twist=enforce_no_twist,
                                      mu_friction=mu_friction)
    except TypeError:
        out = SIM.solve_shape_from_dL(dL_m, params, delta_list)

    if isinstance(out, dict):
        P = out.get('P') or out.get('p') or out.get('centerline')
    elif isinstance(out, (list, tuple)) and len(out) > 0:
        P = out[0]
    else:
        P = None
    if P is None or len(P) == 0:
        raise RuntimeError('solve_shape_from_dL returned no centerline P')
    p0 = np.asarray(P[0], dtype=float)
    return np.asarray(P[-1], dtype=float) - p0


def sample_workspace(params, delta_list, n, lo, hi, seed, enforce_no_twist=True, mu_friction=0.8):
    rng = np.random.default_rng(seed)
    cmds, tips = [], []
    for _ in range(n):
        cmd = rng.uniform(lo, hi, size=3)
        try:
            tip = solve_tip(cmd, params, delta_list, enforce_no_twist, mu_friction)
        except Exception:
            continue
        cmds.append(cmd); tips.append(tip)

    # Extra samples around cable0-dominant movement.
    for a in np.linspace(lo, hi, max(12, n // 25)):
        for base in [-18.0, -22.0, -26.0, -30.0]:
            cmd = np.clip(np.array([a, base, base], dtype=float), lo, hi)
            try:
                tip = solve_tip(cmd, params, delta_list, enforce_no_twist, mu_friction)
            except Exception:
                continue
            cmds.append(cmd); tips.append(tip)
    return np.asarray(cmds), np.asarray(tips)


def select_cable0_biased_center(cmds, tips, cable0_bias):
    c0_pull = -cmds[:, 0]
    other_pull = -0.5 * (cmds[:, 1] + cmds[:, 2])
    dominance = c0_pull - other_pull
    q = 0.50 + 0.40 * np.clip(cable0_bias, 0.0, 1.0)
    mask = dominance >= np.quantile(dominance, q)
    pts = tips[mask] if np.sum(mask) >= 10 else tips
    local_cmds = cmds[mask] if np.sum(mask) >= 10 else cmds

    xy = pts[:, :2]
    z = pts[:, 2]
    xy_med = np.median(xy, axis=0)
    z_med = np.median(z)
    score = np.linalg.norm(xy - xy_med, axis=1) + 0.5 * np.abs(z - z_med) + 0.002 * np.linalg.norm(local_cmds + 25.0, axis=1)
    idx = int(np.argmin(score))
    return pts[idx], local_cmds[idx]


def estimate_cable0_direction(params, delta_list, base_cmd, lo, hi, enforce_no_twist=True, mu_friction=0.8):
    c1 = np.array(base_cmd, dtype=float)
    c2 = np.array(base_cmd, dtype=float)
    c1[0] = np.clip(c1[0] + 5.0, lo, hi)
    c2[0] = np.clip(c2[0] - 5.0, lo, hi)
    try:
        return solve_tip(c2, params, delta_list, enforce_no_twist, mu_friction) - solve_tip(c1, params, delta_list, enforce_no_twist, mu_friction)
    except Exception:
        return np.array([1.0, 0.0, 0.0])


def build_target_ellipse(center, major_mm, minor_mm, cable0_direction):
    v = np.asarray(cable0_direction[:2], dtype=float)
    if np.linalg.norm(v) < 1e-9:
        e1 = np.array([1.0, 0.0])
    else:
        e1 = v / np.linalg.norm(v)
    e2 = np.array([-e1[1], e1[0]])
    A = major_mm * 0.5e-3
    B = minor_mm * 0.5e-3
    targets = []
    for i in range(8):
        th = 2.0 * math.pi * i / 8.0
        xy = center[:2] + A * math.cos(th) * e1 + B * math.sin(th) * e2
        targets.append([xy[0], xy[1], center[2]])
    return np.asarray(targets), e1, e2


def nearest_workspace_guess(target, cmds, tips):
    d = np.linalg.norm(tips - target[None, :], axis=1)
    return np.array(cmds[int(np.argmin(d))], dtype=float)


def inverse_solve_one(target, params, delta_list, guess, prev_cmd, lo, hi, args, seed):
    rng = np.random.default_rng(seed)
    best_cmd = np.clip(np.array(guess, dtype=float), lo, hi)

    def cost_of(cmd):
        tip = solve_tip(cmd, params, delta_list, not args.allow_twist, args.mu)
        e_xy = np.linalg.norm(tip[:2] - target[:2])
        e_z = abs(tip[2] - target[2])
        smooth = np.linalg.norm((cmd - prev_cmd) / max(1e-6, hi - lo))
        cable12 = abs(cmd[1] - prev_cmd[1]) + abs(cmd[2] - prev_cmd[2])
        c0_var = abs(cmd[0] - prev_cmd[0])
        other_var = 0.5 * (abs(cmd[1] - prev_cmd[1]) + abs(cmd[2] - prev_cmd[2]))
        cable0_bonus = max(0.0, other_var - c0_var)
        c = (e_xy ** 2 + args.z_weight * e_z ** 2
             + args.smooth_weight * smooth ** 2
             + args.cable12_weight * (cable12 / 30.0) ** 2
             + args.cable0_pref_weight * (cable0_bonus / 10.0) ** 2)
        return c, tip

    best_cost, best_tip = cost_of(best_cmd)
    step = args.initial_step
    no_imp = 0
    for _ in range(args.inverse_iters):
        noise = rng.normal(0.0, step, size=3)
        noise[0] *= 1.25
        noise[1] *= 0.80
        noise[2] *= 0.80
        cand = np.clip(best_cmd + noise, lo, hi)
        try:
            c, tip = cost_of(cand)
        except Exception:
            continue
        if c < best_cost:
            best_cmd, best_cost, best_tip = cand, c, tip
            no_imp = 0
        else:
            no_imp += 1
        if no_imp > 80:
            step = max(0.12, step * 0.72)
            no_imp = 0
    return best_cmd, best_tip, best_cost


def inverse_solve_path(targets, params, delta_list, workspace_cmds, workspace_tips, args):
    solved_cmds, solved_tips, costs = [], [], []
    prev = nearest_workspace_guess(targets[0], workspace_cmds, workspace_tips)
    for i, target in enumerate(targets):
        guess = nearest_workspace_guess(target, workspace_cmds, workspace_tips)
        if i > 0:
            guess = 0.65 * guess + 0.35 * prev
        cmd, tip, cost = inverse_solve_one(target, params, delta_list, guess, prev, args.cmd_lo, args.cmd_hi, args, args.seed + i * 19)
        solved_cmds.append(cmd); solved_tips.append(tip); costs.append(cost)
        prev = cmd

    if args.second_pass:
        prev = solved_cmds[-1]
        new_cmds, new_tips, new_costs = [], [], []
        for i, target in enumerate(targets):
            cmd, tip, cost = inverse_solve_one(target, params, delta_list, solved_cmds[i], prev, args.cmd_lo, args.cmd_hi, args, args.seed + 1000 + i * 23)
            new_cmds.append(cmd); new_tips.append(tip); new_costs.append(cost)
            prev = cmd
        solved_cmds, solved_tips, costs = new_cmds, new_tips, new_costs
    return np.asarray(solved_cmds), np.asarray(solved_tips), np.asarray(costs)


def save_outputs(out_dir, targets, solved_cmds, solved_tips, workspace_cmds, workspace_tips, center, center_cmd, cable0_dir, args):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / 'inverse_ellipse_waypoints.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['waypoint_id', 'phase', 'target_x', 'target_y', 'target_z',
                    'cmd0', 'cmd1', 'cmd2', 'sim_tip_x', 'sim_tip_y', 'sim_tip_z', 'pos_error_mm'])
        for i in range(8):
            err = np.linalg.norm(solved_tips[i] - targets[i]) * 1000.0
            w.writerow([i, 2.0 * math.pi * i / 8.0,
                        targets[i,0], targets[i,1], targets[i,2],
                        solved_cmds[i,0], solved_cmds[i,1], solved_cmds[i,2],
                        solved_tips[i,0], solved_tips[i,1], solved_tips[i,2], err])

    errors = np.linalg.norm(solved_tips - targets, axis=1) * 1000.0
    jumps = np.linalg.norm(np.diff(np.vstack([solved_cmds, solved_cmds[0:1]]), axis=0), axis=1)
    summary_path = out / 'inverse_ellipse_summary.txt'
    with summary_path.open('w') as f:
        f.write('8-point discrete ellipse inverse trajectory\n')
        f.write('===========================================\n\n')
        f.write(f'ellipse major/minor diameters: {args.major_mm:.1f} / {args.minor_mm:.1f} mm\n')
        f.write(f'command bounds: [{args.cmd_lo:.1f}, {args.cmd_hi:.1f}] mm\n')
        f.write(f'cable0_bias: {args.cable0_bias:.2f}\n')
        f.write(f'selected center tip: {center.tolist()} m\n')
        f.write(f'selected center command: {center_cmd.tolist()} mm\n')
        f.write(f'estimated cable0 direction: {cable0_dir.tolist()}\n\n')
        f.write(f'inverse error mean/max: {errors.mean():.3f} / {errors.max():.3f} mm\n')
        f.write(f'closed-loop command jump mean/max: {jumps.mean():.3f} / {jumps.max():.3f} mm\n\n')
        for k in range(3):
            f.write(f'cmd{k}: min={solved_cmds[:,k].min():.3f}, max={solved_cmds[:,k].max():.3f}, span={np.ptp(solved_cmds[:,k]):.3f} mm\n')
        f.write('\nRecommended first hardware run:\n')
        f.write('python3 discrete_waypoint_player.py --csv inverse_ellipse_output/inverse_ellipse_waypoints.csv --transition 5 --hold 7 --cycles 1 --scale 0.7\n')

    if plt is not None:
        fig = plt.figure()
        plt.scatter(workspace_tips[:,0]*1000, workspace_tips[:,1]*1000, s=4, alpha=0.22)
        plt.plot(targets[:,0]*1000, targets[:,1]*1000, 'o--', label='target')
        plt.plot(solved_tips[:,0]*1000, solved_tips[:,1]*1000, 's-', label='inverse solution')
        for i in range(8):
            plt.text(targets[i,0]*1000, targets[i,1]*1000, str(i))
        plt.axis('equal'); plt.xlabel('x (mm)'); plt.ylabel('y (mm)')
        plt.title('Workspace and selected 8-point inverse ellipse')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(out / 'workspace_xy.png', dpi=180); plt.close(fig)

        fig = plt.figure()
        plt.plot(targets[:,0]*1000, targets[:,1]*1000, 'o--', label='target')
        plt.plot(solved_tips[:,0]*1000, solved_tips[:,1]*1000, 's-', label='model tip')
        plt.axis('equal'); plt.xlabel('x (mm)'); plt.ylabel('y (mm)')
        plt.title('8-point inverse ellipse, XY')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(out / 'inverse_ellipse_xy.png', dpi=180); plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(targets[:,0]*1000, targets[:,1]*1000, targets[:,2]*1000, 'o--', label='target')
        ax.plot(solved_tips[:,0]*1000, solved_tips[:,1]*1000, solved_tips[:,2]*1000, 's-', label='model tip')
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
        ax.set_title('8-point inverse ellipse, 3D'); ax.legend(); plt.tight_layout()
        plt.savefig(out / 'inverse_ellipse_xyz.png', dpi=180); plt.close(fig)

        fig = plt.figure()
        idx = np.arange(8)
        plt.plot(idx, solved_cmds[:,0], 'o-', label='cmd0')
        plt.plot(idx, solved_cmds[:,1], 'o-', label='cmd1')
        plt.plot(idx, solved_cmds[:,2], 'o-', label='cmd2')
        plt.xlabel('waypoint id'); plt.ylabel('dL command (mm)')
        plt.title('Inverse-solved tendon commands')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(out / 'inverse_ellipse_commands.png', dpi=180); plt.close(fig)

    print('\nSaved:')
    print(' ', csv_path)
    print(' ', summary_path)
    if plt is not None:
        print(' ', out / 'workspace_xy.png')
        print(' ', out / 'inverse_ellipse_xy.png')
        print(' ', out / 'inverse_ellipse_xyz.png')
        print(' ', out / 'inverse_ellipse_commands.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--major-mm', type=float, default=60.0)
    ap.add_argument('--minor-mm', type=float, default=35.0)
    ap.add_argument('--cmd-lo', type=float, default=-42.0)
    ap.add_argument('--cmd-hi', type=float, default=-8.0)
    ap.add_argument('--workspace-n', type=int, default=1800)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mu', type=float, default=0.8)
    ap.add_argument('--allow-twist', action='store_true')
    ap.add_argument('--cable0-bias', type=float, default=0.65)
    ap.add_argument('--inverse-iters', type=int, default=700)
    ap.add_argument('--initial-step', type=float, default=4.0)
    ap.add_argument('--smooth-weight', type=float, default=0.012)
    ap.add_argument('--cable12-weight', type=float, default=0.004)
    ap.add_argument('--cable0-pref-weight', type=float, default=0.0025)
    ap.add_argument('--z-weight', type=float, default=0.55)
    ap.add_argument('--second-pass', action='store_true', default=True)
    ap.add_argument('--out-dir', default='inverse_ellipse_output')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    params = get_params()
    delta_list = default_delta_list(params, args.seed)

    print('[1/4] Sampling simulated workspace...')
    ws_cmds, ws_tips = sample_workspace(params, delta_list, args.workspace_n, args.cmd_lo, args.cmd_hi,
                                        args.seed, not args.allow_twist, args.mu)
    print('      valid samples:', len(ws_tips))

    print('[2/4] Selecting cable0-biased ellipse...')
    center, center_cmd = select_cable0_biased_center(ws_cmds, ws_tips, args.cable0_bias)
    c0_dir = estimate_cable0_direction(params, delta_list, center_cmd, args.cmd_lo, args.cmd_hi,
                                       not args.allow_twist, args.mu)
    targets, _, _ = build_target_ellipse(center, args.major_mm, args.minor_mm, c0_dir)
    print('      center tip (m):', center)
    print('      center cmd (mm):', center_cmd)
    print('      cable0 direction:', c0_dir)

    print('[3/4] Inverse-solving 8 waypoints...')
    solved_cmds, solved_tips, costs = inverse_solve_path(targets, params, delta_list, ws_cmds, ws_tips, args)
    err = np.linalg.norm(solved_tips - targets, axis=1) * 1000.0
    print('      position error mm:', np.round(err, 3))
    print('      mean/max error mm:', float(err.mean()), float(err.max()))
    for k in range(3):
        print(f'      cmd{k}: {solved_cmds[:,k].min():.2f} to {solved_cmds[:,k].max():.2f} mm, span {np.ptp(solved_cmds[:,k]):.2f} mm')

    print('[4/4] Saving outputs...')
    save_outputs(args.out_dir, targets, solved_cmds, solved_tips, ws_cmds, ws_tips, center, center_cmd, c0_dir, args)


if __name__ == '__main__':
    main()
