#!/usr/bin/env python3
"""
validation_monitor.py

Live validation GUI and logger for cyclic trajectory tests.

It does NOT command the robot. It only monitors ROS topics and logs:
    - tendon command
    - SOFA tip Euler
    - IMU Euler
    - optional hybrid tip Euler
    - optional SOFA tip position if your SOFA publishes /robot/tip_position

Required topics:
    /robot/tendon_commands
    /robot/tip_euler
    /robot/imu_euler

Optional topics:
    /robot/tip_euler_hybrid
    /robot/tip_euler_hybrid_error
    /robot/tip_position
"""

import csv
import os
import time
import argparse
from collections import deque

try:
    from PySide6 import QtCore, QtWidgets
    QtLib = "PySide6"
except Exception:
    try:
        from PyQt5 import QtCore, QtWidgets
        QtLib = "PyQt5"
    except Exception as e:
        raise RuntimeError("Need PySide6 or PyQt5: pip install PySide6 or PyQt5") from e

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class ValidationRos(Node):
    def __init__(
        self,
        cmd_topic="/robot/tendon_commands",
        sofa_topic="/robot/tip_euler",
        imu_topic="/robot/imu_euler",
        hybrid_topic="/robot/tip_euler_hybrid",
        hybrid_error_topic="/robot/tip_euler_hybrid_error",
        tip_pos_topic="/robot/tip_position",
    ):
        super().__init__("validation_monitor_node")

        self.cmd = [0.0, 0.0, 0.0]
        self.sofa = [0.0, 0.0]
        self.imu = [0.0, 0.0]
        self.hybrid = [0.0, 0.0]
        self.hybrid_error = [0.0, 0.0]
        self.tip_pos = [0.0, 0.0, 0.0]
        self.has_tip_pos = False

        self.cmd_stamp = 0.0
        self.sofa_stamp = 0.0
        self.imu_stamp = 0.0
        self.hybrid_stamp = 0.0
        self.hybrid_error_stamp = 0.0
        self.tip_pos_stamp = 0.0

        self.create_subscription(Float32MultiArray, cmd_topic, self.on_cmd, 10)
        self.create_subscription(Float32MultiArray, sofa_topic, self.on_sofa, 10)
        self.create_subscription(Float32MultiArray, imu_topic, self.on_imu, 10)
        self.create_subscription(Float32MultiArray, hybrid_topic, self.on_hybrid, 10)
        self.create_subscription(Float32MultiArray, hybrid_error_topic, self.on_hybrid_error, 10)
        self.create_subscription(Float32MultiArray, tip_pos_topic, self.on_tip_pos, 10)

    def on_cmd(self, msg):
        if len(msg.data) >= 3:
            self.cmd = [float(msg.data[0]), float(msg.data[1]), float(msg.data[2])]
            self.cmd_stamp = time.time()

    def on_sofa(self, msg):
        if len(msg.data) >= 2:
            self.sofa = [float(msg.data[0]), float(msg.data[1])]
            self.sofa_stamp = time.time()

    def on_imu(self, msg):
        if len(msg.data) >= 2:
            self.imu = [float(msg.data[0]), float(msg.data[1])]
            self.imu_stamp = time.time()

    def on_hybrid(self, msg):
        if len(msg.data) >= 2:
            self.hybrid = [float(msg.data[0]), float(msg.data[1])]
            self.hybrid_stamp = time.time()

    def on_hybrid_error(self, msg):
        if len(msg.data) >= 2:
            self.hybrid_error = [float(msg.data[0]), float(msg.data[1])]
            self.hybrid_error_stamp = time.time()

    def on_tip_pos(self, msg):
        if len(msg.data) >= 3:
            self.tip_pos = [float(msg.data[0]), float(msg.data[1]), float(msg.data[2])]
            self.tip_pos_stamp = time.time()
            self.has_tip_pos = True


class ValidationMonitor(QtWidgets.QWidget):
    def __init__(self, ros_node, log_path="ellipse_validation_log_v2.csv", spin_hz=100, ui_hz=20, log_hz=30):
        super().__init__()
        self.setWindowTitle(f"ellipse_validation Monitor ({QtLib})")
        self.ros_node = ros_node
        self.log_path = log_path
        self.t0 = time.time()

        self.last_log = 0.0
        self.log_period = 1.0 / float(log_hz)

        self.max_points = 500
        self.t_hist = deque(maxlen=self.max_points)
        self.sofa_roll_hist = deque(maxlen=self.max_points)
        self.sofa_pitch_hist = deque(maxlen=self.max_points)
        self.imu_roll_hist = deque(maxlen=self.max_points)
        self.imu_pitch_hist = deque(maxlen=self.max_points)
        self.hybrid_roll_hist = deque(maxlen=self.max_points)
        self.hybrid_pitch_hist = deque(maxlen=self.max_points)

        self._init_logger()
        self._build_ui()

        self.spin_timer = QtCore.QTimer(self)
        self.spin_timer.timeout.connect(self._tick_ros)
        self.spin_timer.start(max(1, int(1000 / spin_hz)))

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self._tick_ui)
        self.ui_timer.start(max(1, int(1000 / ui_hz)))

    def _init_logger(self):
        # Keep one fixed header. If an old CSV with a different header already exists,
        # automatically create a new timestamped log file instead of appending mismatched rows.
        self.fieldnames = [
            "t",
            "cmd0", "cmd1", "cmd2",
            "sofa_roll", "sofa_pitch",
            "imu_roll", "imu_pitch",
            "hybrid_roll", "hybrid_pitch",
            "err_sofa_roll", "err_sofa_pitch",
            "err_hybrid_roll", "err_hybrid_pitch",
            "sofa_tip_x", "sofa_tip_y", "sofa_tip_z", "has_sofa_tip",
            "cmd_age", "sofa_age", "imu_age", "hybrid_age", "tip_pos_age",
        ]

        if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0:
            try:
                with open(self.log_path, "r", newline="") as f:
                    first = next(csv.reader(f), [])
                if first != self.fieldnames:
                    base, ext = os.path.splitext(self.log_path)
                    self.log_path = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}{ext or '.csv'}"
            except Exception:
                base, ext = os.path.splitext(self.log_path)
                self.log_path = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}{ext or '.csv'}"

        file_exists = os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0
        self.log_file = open(self.log_path, "a", newline="")
        self.log_writer = csv.writer(self.log_file)

        if not file_exists:
            self.log_writer.writerow(self.fieldnames)
            self.log_file.flush()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel(
            "ellipse_validation monitor: command / SOFA Euler / IMU Euler / Hybrid Euler"
        )
        title.setWordWrap(True)
        layout.addWidget(title)

        grid_box = QtWidgets.QGroupBox("Live values")
        grid = QtWidgets.QGridLayout(grid_box)

        self.labels = {}
        rows = [
            ("cmd", "Command dL0,dL1,dL2 (mm)"),
            ("sofa", "SOFA roll,pitch (deg)"),
            ("imu", "IMU roll,pitch (deg)"),
            ("hybrid", "Hybrid roll,pitch (deg)"),
            ("err_sofa", "SOFA - IMU error (deg)"),
            ("err_hybrid", "Hybrid - IMU error (deg)"),
            ("tip_pos", "SOFA tip position x,y,z (m)"),
            ("status", "Topic status"),
            ("log", "Log file"),
        ]

        for r, (key, name) in enumerate(rows):
            grid.addWidget(QtWidgets.QLabel(name), r, 0)
            lbl = QtWidgets.QLabel("-")
            lbl.setMinimumWidth(500)
            grid.addWidget(lbl, r, 1)
            self.labels[key] = lbl

        layout.addWidget(grid_box)

        self.phase_box = QtWidgets.QGroupBox("Roll-Pitch phase summary")
        phase_layout = QtWidgets.QVBoxLayout(self.phase_box)
        self.phase_text = QtWidgets.QPlainTextEdit()
        self.phase_text.setReadOnly(True)
        self.phase_text.setMinimumHeight(140)
        phase_layout.addWidget(self.phase_text)
        layout.addWidget(self.phase_box)

        button_row = QtWidgets.QHBoxLayout()
        self.clear_btn = QtWidgets.QPushButton("Clear live buffer")
        self.clear_btn.clicked.connect(self._clear_buffer)
        self.mark_btn = QtWidgets.QPushButton("Mark event in log")
        self.mark_btn.clicked.connect(self._mark_event)
        button_row.addWidget(self.clear_btn)
        button_row.addWidget(self.mark_btn)
        layout.addLayout(button_row)

    def _clear_buffer(self):
        self.t_hist.clear()
        self.sofa_roll_hist.clear()
        self.sofa_pitch_hist.clear()
        self.imu_roll_hist.clear()
        self.imu_pitch_hist.clear()
        self.hybrid_roll_hist.clear()
        self.hybrid_pitch_hist.clear()

    def _mark_event(self):
        now = time.time() - self.t0
        self.log_writer.writerow([f"# MARK {now:.3f}"])
        self.log_file.flush()

    def _age_text(self, stamp):
        if stamp <= 0:
            return "NO DATA"
        age = time.time() - stamp
        if age < 0.5:
            return f"OK {age:.2f}s"
        if age < 2.0:
            return f"OLD {age:.2f}s"
        return f"STALE {age:.2f}s"

    def _tick_ros(self):
        try:
            rclpy.spin_once(self.ros_node, timeout_sec=0)
        except Exception as e:
            self.labels["status"].setText(f"ROS spin failed: {e}")

    def _tick_ui(self):
        now_abs = time.time()
        t = now_abs - self.t0

        cmd = list(self.ros_node.cmd)
        sofa = list(self.ros_node.sofa)
        imu = list(self.ros_node.imu)
        hybrid = list(self.ros_node.hybrid)
        tip_pos = list(self.ros_node.tip_pos)

        err_sofa = [sofa[0] - imu[0], sofa[1] - imu[1]]
        err_hybrid = [hybrid[0] - imu[0], hybrid[1] - imu[1]]

        self.labels["cmd"].setText(f"[{cmd[0]: .3f}, {cmd[1]: .3f}, {cmd[2]: .3f}]")
        self.labels["sofa"].setText(f"[{sofa[0]: .3f}, {sofa[1]: .3f}]")
        self.labels["imu"].setText(f"[{imu[0]: .3f}, {imu[1]: .3f}]")
        self.labels["hybrid"].setText(f"[{hybrid[0]: .3f}, {hybrid[1]: .3f}]")
        self.labels["err_sofa"].setText(f"[{err_sofa[0]: .3f}, {err_sofa[1]: .3f}]")
        self.labels["err_hybrid"].setText(f"[{err_hybrid[0]: .3f}, {err_hybrid[1]: .3f}]")
        self.labels["tip_pos"].setText(
            f"[{tip_pos[0]: .5f}, {tip_pos[1]: .5f}, {tip_pos[2]: .5f}]  "
            f"has_sofa_tip={int(self.ros_node.has_tip_pos)}"
        )
        self.labels["status"].setText(
            f"cmd={self._age_text(self.ros_node.cmd_stamp)} | "
            f"SOFA={self._age_text(self.ros_node.sofa_stamp)} | "
            f"IMU={self._age_text(self.ros_node.imu_stamp)} | "
            f"hybrid={self._age_text(self.ros_node.hybrid_stamp)} | "
            f"tip_pos={self._age_text(self.ros_node.tip_pos_stamp)}"
        )
        self.labels["log"].setText(self.log_path)

        self.t_hist.append(t)
        self.sofa_roll_hist.append(sofa[0])
        self.sofa_pitch_hist.append(sofa[1])
        self.imu_roll_hist.append(imu[0])
        self.imu_pitch_hist.append(imu[1])
        self.hybrid_roll_hist.append(hybrid[0])
        self.hybrid_pitch_hist.append(hybrid[1])

        if len(self.t_hist) > 5:
            self.phase_text.setPlainText(
                "Roll-Pitch validation summary, last live buffer:\n"
                f"SOFA   roll range [{min(self.sofa_roll_hist): .2f}, {max(self.sofa_roll_hist): .2f}], "
                f"pitch range [{min(self.sofa_pitch_hist): .2f}, {max(self.sofa_pitch_hist): .2f}]\n"
                f"IMU    roll range [{min(self.imu_roll_hist): .2f}, {max(self.imu_roll_hist): .2f}], "
                f"pitch range [{min(self.imu_pitch_hist): .2f}, {max(self.imu_pitch_hist): .2f}]\n"
                f"Hybrid roll range [{min(self.hybrid_roll_hist): .2f}, {max(self.hybrid_roll_hist): .2f}], "
                f"pitch range [{min(self.hybrid_pitch_hist): .2f}, {max(self.hybrid_pitch_hist): .2f}]\n\n"
                "Use circle_validation_log.csv for final plots."
            )

        if now_abs - self.last_log >= self.log_period:
            self.last_log = now_abs
            self.log_writer.writerow([
                t,
                cmd[0], cmd[1], cmd[2],
                sofa[0], sofa[1],
                imu[0], imu[1],
                hybrid[0], hybrid[1],
                err_sofa[0], err_sofa[1],
                err_hybrid[0], err_hybrid[1],
                tip_pos[0], tip_pos[1], tip_pos[2], int(self.ros_node.has_tip_pos),
                now_abs - self.ros_node.cmd_stamp if self.ros_node.cmd_stamp > 0 else -1,
                now_abs - self.ros_node.sofa_stamp if self.ros_node.sofa_stamp > 0 else -1,
                now_abs - self.ros_node.imu_stamp if self.ros_node.imu_stamp > 0 else -1,
                now_abs - self.ros_node.hybrid_stamp if self.ros_node.hybrid_stamp > 0 else -1,
                now_abs - self.ros_node.tip_pos_stamp if self.ros_node.tip_pos_stamp > 0 else -1,
            ])
            self.log_file.flush()

    def closeEvent(self, event):
        try:
            self.log_file.flush()
            self.log_file.close()
        except Exception:
            pass
        event.accept()


def main(args=None):
    parser = argparse.ArgumentParser(description="Validation monitor for ellipse trajectory")
    parser.add_argument("--cmd-topic", default="/robot/tendon_commands")
    parser.add_argument("--sofa-topic", default="/robot/tip_euler")
    parser.add_argument("--imu-topic", default="/robot/imu_euler")
    parser.add_argument("--hybrid-topic", default="/robot/tip_euler_hybrid")
    parser.add_argument("--hybrid-error-topic", default="/robot/tip_euler_hybrid_error")
    parser.add_argument("--tip-pos-topic", default="/robot/tip_position")
    parser.add_argument("--log", default="ellipse_validation_log_v2.csv")
    parser.add_argument("--spin-hz", type=float, default=100.0)
    parser.add_argument("--ui-hz", type=float, default=20.0)
    parser.add_argument("--log-hz", type=float, default=30.0)

    parsed, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = ValidationRos(
        cmd_topic=parsed.cmd_topic,
        sofa_topic=parsed.sofa_topic,
        imu_topic=parsed.imu_topic,
        hybrid_topic=parsed.hybrid_topic,
        hybrid_error_topic=parsed.hybrid_error_topic,
        tip_pos_topic=parsed.tip_pos_topic,
    )

    app = QtWidgets.QApplication([])
    win = ValidationMonitor(
        node,
        log_path=parsed.log,
        spin_hz=parsed.spin_hz,
        ui_hz=parsed.ui_hz,
        log_hz=parsed.log_hz,
    )
    win.resize(760, 520)
    win.show()

    try:
        app.exec()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
