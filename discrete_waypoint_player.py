#!/usr/bin/env python3
# discrete_waypoint_player.py
"""
Discrete step-hold waypoint player for inverse_ellipse_waypoints.csv.

Publishes:
    /robot/tendon_commands  std_msgs/Float32MultiArray [cmd0, cmd1, cmd2]

Pattern:
    waypoint -> slow transition -> hold -> next waypoint

Run:
    source /opt/ros/jazzy/setup.bash
    python3 discrete_waypoint_player.py --csv inverse_ellipse_output/inverse_ellipse_waypoints.csv --transition 5 --hold 7 --cycles 1 --scale 0.7
"""

import argparse
import csv
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


def load_waypoints(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'waypoint_id': int(float(r['waypoint_id'])),
                'target': [float(r['target_x']), float(r['target_y']), float(r['target_z'])],
                'cmd': [float(r['cmd0']), float(r['cmd1']), float(r['cmd2'])],
                'sim_tip': [float(r['sim_tip_x']), float(r['sim_tip_y']), float(r['sim_tip_z'])],
            })
    rows.sort(key=lambda x: x['waypoint_id'])
    if len(rows) < 2:
        raise RuntimeError('Need at least two waypoints')
    return rows


def scaled_rows(rows, scale):
    cmds = [r['cmd'] for r in rows]
    mean = [sum(c[k] for c in cmds) / len(cmds) for k in range(3)]
    out = []
    for r in rows:
        rr = dict(r)
        rr['cmd'] = [mean[k] + scale * (r['cmd'][k] - mean[k]) for k in range(3)]
        out.append(rr)
    return out


def lerp(a, b, s):
    return [(1.0 - s) * a[k] + s * b[k] for k in range(3)]


class WaypointPlayer(Node):
    def __init__(self, topic):
        super().__init__('discrete_waypoint_player')
        self.pub = self.create_publisher(Float32MultiArray, topic, 10)

    def publish_cmd(self, cmd):
        msg = Float32MultiArray()
        msg.data = [float(cmd[0]), float(cmd[1]), float(cmd[2])]
        self.pub.publish(msg)


def run_for(node, cmd_func, duration, rate_hz):
    dt = 1.0 / rate_hz
    t0 = time.time()
    while rclpy.ok():
        t = time.time() - t0
        if t >= duration:
            break
        cmd = cmd_func(t / max(1e-9, duration))
        node.publish_cmd(cmd)
        rclpy.spin_once(node, timeout_sec=0)
        time.sleep(dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='inverse_ellipse_output/inverse_ellipse_waypoints.csv')
    ap.add_argument('--topic', default='/robot/tendon_commands')
    ap.add_argument('--transition', type=float, default=5.0)
    ap.add_argument('--hold', type=float, default=7.0)
    ap.add_argument('--cycles', type=int, default=1)
    ap.add_argument('--rate-hz', type=float, default=30.0)
    ap.add_argument('--scale', type=float, default=0.7)
    ap.add_argument('--no-return-first', action='store_true')
    args = ap.parse_args()

    rows = scaled_rows(load_waypoints(args.csv), args.scale)
    rclpy.init()
    node = WaypointPlayer(args.topic)

    print(f'[INFO] Loaded {len(rows)} waypoints from {args.csv}')
    print(f'[INFO] topic={args.topic}, transition={args.transition}s, hold={args.hold}s, scale={args.scale}')
    for r in rows:
        c = r['cmd']
        print(f"  wp{r['waypoint_id']}: [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]")

    current = rows[0]['cmd']
    print('[INFO] Initial hold at wp0')
    run_for(node, lambda s, c=current: c, args.hold, args.rate_hz)

    for cyc in range(args.cycles):
        print(f'[INFO] Cycle {cyc + 1}/{args.cycles}')
        for i in range(1, len(rows)):
            target = rows[i]['cmd']
            print(f"[INFO] Transition to wp{rows[i]['waypoint_id']}")
            start = list(current)
            run_for(node, lambda s, a=start, b=target: lerp(a, b, min(1.0, max(0.0, s))), args.transition, args.rate_hz)
            current = list(target)
            print(f"[INFO] Hold wp{rows[i]['waypoint_id']}")
            run_for(node, lambda s, c=current: c, args.hold, args.rate_hz)

        if not args.no_return_first:
            target = rows[0]['cmd']
            print('[INFO] Transition back to wp0')
            start = list(current)
            run_for(node, lambda s, a=start, b=target: lerp(a, b, min(1.0, max(0.0, s))), args.transition, args.rate_hz)
            current = list(target)
            print('[INFO] Hold wp0')
            run_for(node, lambda s, c=current: c, args.hold, args.rate_hz)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
