#!/usr/bin/env python3
import sys
import time
import csv
import os

QtLib = None
try:
    from PySide6 import QtCore, QtWidgets
    QtLib = "PySide6"
except Exception:
    try:
        from PyQt5 import QtCore, QtWidgets
        QtLib = "PyQt5"
    except Exception as e:
        raise RuntimeError(
            "Need PySide6 or PyQt5.\n"
            "Install one:\n  pip install PySide6\nor\n  pip install PyQt5\n"
        ) from e

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class RosInterface(Node):
    def __init__(self, cmd_topic='/robot/tendon_commands', state_topic='/robot/tendon_state', imu_topic='/robot/imu_euler', tip_topic='/robot/tip_euler'):
        super().__init__('tendon_qt_console')
        self.publisher_ = self.create_publisher(Float32MultiArray, cmd_topic, 10)
        self.subscription = self.create_subscription(Float32MultiArray, state_topic, self._on_state, 10)
        self.imu_subscription = self.create_subscription(Float32MultiArray, imu_topic, self._on_imu, 10)
        self.tip_subscription = self.create_subscription(Float32MultiArray, tip_topic, self._on_tip, 10)
        self.latest_state = [0.0, 0.0, 0.0]
        self.state_stamp = 0.0
        self.latest_imu = [0.0, 0.0]
        self.imu_stamp = 0.0
        self.latest_tip = [0.0, 0.0]
        self.tip_stamp = 0.0

    def send_command(self, vals):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in vals[:3]]
        self.publisher_.publish(msg)

    def _on_state(self, msg):
        if len(msg.data) >= 3:
            self.latest_state = [float(msg.data[0]), float(msg.data[1]), float(msg.data[2])]
            self.state_stamp = time.time()

    def _on_imu(self, msg):
        if len(msg.data) >= 2:
            self.latest_imu = [float(msg.data[0]), float(msg.data[1])]
            self.imu_stamp = time.time()

    def _on_tip(self, msg):
        if len(msg.data) >= 2:
            self.latest_tip = [float(msg.data[0]), float(msg.data[1])]
            self.tip_stamp = time.time()


class TendonConsole(QtWidgets.QWidget):
    def __init__(self, ros_node, dL_min=-60.0, dL_max=0.0, send_hz=30, spin_hz=100):
        super().__init__()
        self.setWindowTitle(f"Tendon Console (Qt / {QtLib}) → ROS 2 stable coordinated")
        self.ros_node = ros_node
        self.dL_min = float(dL_min)
        self.dL_max = float(dL_max)
        self.scale = 1000
        self.smin = int(round(self.dL_min * self.scale))
        self.smax = int(round(self.dL_max * self.scale))
        self._send_period = 1.0 / float(send_hz)
        self._last_send = 0.0
        self._last_applied_state_stamp = 0.0
        self._dragging = False
        self._sync_paused_until = 0.0
        self._command_dirty = True
        self._actual_state = [0.0, 0.0, 0.0]
        self._imu_state = [0.0, 0.0]
        self._tip_state = [0.0, 0.0]
        self._last_imu_stamp = 0.0
        self._last_tip_stamp = 0.0

        # ==== Physics-informed dynamic residual learning logger ====
        # This logger turns the SOFA soft sensor + IMU comparison into a
        # time-history dataset for lightweight dynamic residual learning.
        self.log_path = "dynamic_residual_log.csv"
        self._log_file = None
        self._log_writer = None

        self._prev_t = None
        self._prev_cmd = [0.0, 0.0, 0.0]
        self._prev_actual = [0.0, 0.0, 0.0]
        self._prev_imu = [0.0, 0.0]
        self._prev_tip = [0.0, 0.0]
        self._prev_err = [0.0, 0.0]

        # ==== IMU based reset parameters ====
        # Positive command means releasing extra cable beyond the nominal zero point.
        # Keep this small to avoid the cable coming off the winch.
        self.reset_extra_release_mm = 6.0
        self.reset_max_release_mm = 8.0
        self.reset_release_hold_s = 1.5
        self.reset_zero_hold_s = 0.6
        self.reset_imu_tol_deg = 1.0
        self.reset_stable_required = 8
        self.reset_tune_period_s = 0.08
        self.reset_tune_gain_mm_per_deg = 0.55
        self.reset_max_pull_mm = 18.0
        self.reset_max_tune_s = 12.0
        self.reset_roll_sign = -1.0
        self.reset_pitch_sign = -1.0
        self._reset_active = False
        self._reset_phase = "idle"
        self._reset_phase_t0 = 0.0
        self._reset_last_tune = 0.0
        self._reset_stable_count = 0
        self._reset_cmd = [0.0, 0.0, 0.0]
        self._reset_release_cmd = [0.0, 0.0, 0.0]
        self._reset_last_publish_vals = [0.0, 0.0, 0.0]
        self._reset_publish_count = 0
        # The command that actually makes the physical robot straight according to IMU.
        # At power-up this is only a guess; every successful IMU reset updates it.
        # This is necessary because [0,0,0] can correspond to a bent robot when the
        # Teflon backbone has hysteresis or the tendons were not mechanically homed.
        self._home_cmd = [0.0, 0.0, 0.0]
        self._home_valid = False

        self._build_ui()
        self._init_dynamic_logger()

        self.send_timer = QtCore.QTimer(self)
        self.send_timer.timeout.connect(self._tick_send)
        self.send_timer.start(max(1, int(1000 / send_hz)))

        self.spin_timer = QtCore.QTimer(self)
        self.spin_timer.timeout.connect(self._tick_ros)
        self.spin_timer.start(max(1, int(1000 / spin_hz)))

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(
            "拖动滑块发送意图 ΔL(mm)。SOFA 回传 actual 只用于显示和作为下一次控制的基线；"
            "不会再被 Qt 自动回发成新命令，避免闭环自激。"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        grp = QtWidgets.QGroupBox("Tendons (command / actual)")
        gl = QtWidgets.QGridLayout(grp)
        self.sliders = []
        self.cmd_labels = []
        self.actual_labels = []

        for i in range(3):
            name = QtWidgets.QLabel(f"dL{i}_mm")
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(self.smin, self.smax)
            slider.setValue(0)
            slider.setSingleStep(10)
            slider.setPageStep(100)
            slider.valueChanged.connect(self._on_slider)
            slider.sliderPressed.connect(self._on_slider_pressed)
            slider.sliderReleased.connect(self._on_slider_released)

            cmd = QtWidgets.QLabel("0.000")
            cmd.setMinimumWidth(70)
            actual = QtWidgets.QLabel("0.000")
            actual.setMinimumWidth(70)
            gl.addWidget(name, i, 0)
            gl.addWidget(slider, i, 1)
            gl.addWidget(QtWidgets.QLabel("cmd:"), i, 2)
            gl.addWidget(cmd, i, 3)
            gl.addWidget(QtWidgets.QLabel("actual:"), i, 4)
            gl.addWidget(actual, i, 5)
            self.sliders.append(slider)
            self.cmd_labels.append(cmd)
            self.actual_labels.append(actual)

        layout.addWidget(grp)

        pose_grp = QtWidgets.QGroupBox("Pose compare (SOFA tip Euler vs IMU Euler)")
        pose_gl = QtWidgets.QGridLayout(pose_grp)
        self.pose_value_labels = {}
        self.pose_err_labels = {}
        pose_rows = [
            ("imu_a", "IMU A (deg)"),
            ("imu_b", "IMU B (deg)"),
            ("sofa_roll", "SOFA Roll (deg)"),
            ("sofa_pitch", "SOFA Pitch (deg)"),
            ("err_a", "Err A = Roll - IMU A"),
            ("err_b", "Err B = Pitch - IMU B"),
        ]
        for row, (key, title) in enumerate(pose_rows):
            pose_gl.addWidget(QtWidgets.QLabel(title), row, 0)
            lbl = QtWidgets.QLabel("0.000")
            lbl.setMinimumWidth(90)
            pose_gl.addWidget(lbl, row, 1)
            self.pose_value_labels[key] = lbl
        layout.addWidget(pose_grp)

        btn_row = QtWidgets.QHBoxLayout()
        zero_btn = QtWidgets.QPushButton("IMU Reset / Straighten")
        zero_btn.clicked.connect(self._zero)
        cancel_reset_btn = QtWidgets.QPushButton("Cancel reset")
        cancel_reset_btn.clicked.connect(self._cancel_reset)
        adopt_btn = QtWidgets.QPushButton("Adopt actual as command")
        adopt_btn.clicked.connect(self._adopt_actual)
        set_home_btn = QtWidgets.QPushButton("Set current as IMU home")
        set_home_btn.clicked.connect(self._set_current_as_home)
        preset1 = QtWidgets.QPushButton("Preset (-40,0,0)")
        preset1.clicked.connect(lambda: self._set_vals([-40.0, 0.0, 0.0], from_user=True, emit_now=True))
        preset2 = QtWidgets.QPushButton("Preset (-40,-40,-40)")
        preset2.clicked.connect(lambda: self._set_vals([-40.0, -40.0, -40.0], from_user=True, emit_now=True))
        btn_row.addWidget(zero_btn)
        btn_row.addWidget(cancel_reset_btn)
        btn_row.addWidget(adopt_btn)
        btn_row.addWidget(set_home_btn)
        btn_row.addWidget(preset1)
        btn_row.addWidget(preset2)
        layout.addLayout(btn_row)

        self.status = QtWidgets.QLabel("Ready. User edits publish ROS 2 commands; SOFA feedback updates actual only. IMU home is not calibrated yet.")
        layout.addWidget(self.status)

        self._refresh_cmd_labels()
        self._refresh_actual_labels([0.0, 0.0, 0.0])
        self._refresh_pose_labels()

    def _mm(self, sval_int):
        return float(sval_int) / self.scale

    def _current_vals(self):
        return [self._mm(s.value()) for s in self.sliders]

    def _refresh_cmd_labels(self):
        vals = self._current_vals()
        for i, v in enumerate(vals):
            self.cmd_labels[i].setText(f"{v:.3f}")


    def _show_cmd_labels_direct(self, vals):
        # Display reset commands even when they include small positive over-release beyond normal slider max.
        for i, v in enumerate(vals[:3]):
            self.cmd_labels[i].setText(f"{float(v):.3f}")

    def _refresh_actual_labels(self, vals):
        for i, v in enumerate(vals[:3]):
            self.actual_labels[i].setText(f"{v:.3f}")


    def _refresh_pose_labels(self):
        imu_a, imu_b = self._imu_state[:2]
        sofa_roll, sofa_pitch = self._tip_state[:2]
        self.pose_value_labels["imu_a"].setText(f"{imu_a:.3f}")
        self.pose_value_labels["imu_b"].setText(f"{imu_b:.3f}")
        self.pose_value_labels["sofa_roll"].setText(f"{sofa_roll:.3f}")
        self.pose_value_labels["sofa_pitch"].setText(f"{sofa_pitch:.3f}")
        self.pose_value_labels["err_a"].setText(f"{(sofa_roll - imu_a):.3f}")
        self.pose_value_labels["err_b"].setText(f"{(sofa_pitch - imu_b):.3f}")

    def _init_dynamic_logger(self):
        """
        Create CSV log for physics-informed dynamic residual learning.

        Residual definition:
            residual = IMU - SOFA

        Corrected soft-sensor output:
            SOFA_corrected = SOFA + learned_residual

        The logged features include previous command, command velocity,
        previous SOFA prediction, and previous residual, so this is a
        lightweight dynamic residual model rather than a purely static fit.
        """
        try:
            file_exists = os.path.exists(self.log_path)
            self._log_file = open(self.log_path, "a", newline="")
            self._log_writer = csv.writer(self._log_file)

            if not file_exists:
                self._log_writer.writerow([
                    "t",

                    "cmd0", "cmd1", "cmd2",
                    "actual0", "actual1", "actual2",

                    "cmd0_prev", "cmd1_prev", "cmd2_prev",
                    "actual0_prev", "actual1_prev", "actual2_prev",

                    "dcmd0_dt", "dcmd1_dt", "dcmd2_dt",
                    "dactual0_dt", "dactual1_dt", "dactual2_dt",

                    "sofa_roll", "sofa_pitch",
                    "imu_roll", "imu_pitch",

                    "sofa_roll_prev", "sofa_pitch_prev",
                    "imu_roll_prev", "imu_pitch_prev",

                    "err_roll", "err_pitch",
                    "err_roll_prev", "err_pitch_prev",
                ])
                self._log_file.flush()

            self.status.setText(f"Dynamic residual logger active: {self.log_path}")
        except Exception as e:
            self._log_file = None
            self._log_writer = None
            try:
                self.status.setText(f"Logger init failed: {e}")
            except Exception:
                pass

    def _log_dynamic_sample(self):
        """
        Log one synchronized-ish sample whenever SOFA tip Euler or IMU Euler updates.
        This is intentionally lightweight and does not change the controller behavior.
        """
        if self._log_writer is None:
            return

        now = time.time()

        cmd = list(self._current_vals())
        actual = list(self._actual_state)
        imu = list(self._imu_state[:2])
        tip = list(self._tip_state[:2])

        sofa_roll, sofa_pitch = tip
        imu_roll, imu_pitch = imu

        # Residual is IMU - SOFA.
        # Later correction is: corrected_SOFA = SOFA + predicted_residual.
        err = [
            imu_roll - sofa_roll,
            imu_pitch - sofa_pitch,
        ]

        if self._prev_t is None:
            dt = 1e-3
        else:
            dt = max(1e-3, now - self._prev_t)

        dcmd_dt = [(cmd[i] - self._prev_cmd[i]) / dt for i in range(3)]
        dactual_dt = [(actual[i] - self._prev_actual[i]) / dt for i in range(3)]

        try:
            self._log_writer.writerow([
                now,

                cmd[0], cmd[1], cmd[2],
                actual[0], actual[1], actual[2],

                self._prev_cmd[0], self._prev_cmd[1], self._prev_cmd[2],
                self._prev_actual[0], self._prev_actual[1], self._prev_actual[2],

                dcmd_dt[0], dcmd_dt[1], dcmd_dt[2],
                dactual_dt[0], dactual_dt[1], dactual_dt[2],

                sofa_roll, sofa_pitch,
                imu_roll, imu_pitch,

                self._prev_tip[0], self._prev_tip[1],
                self._prev_imu[0], self._prev_imu[1],

                err[0], err[1],
                self._prev_err[0], self._prev_err[1],
            ])
            self._log_file.flush()
        except Exception as e:
            self.status.setText(f"Logging failed: {e}")

        self._prev_t = now
        self._prev_cmd = cmd
        self._prev_actual = actual
        self._prev_imu = imu
        self._prev_tip = tip
        self._prev_err = err

    def _mark_dirty(self):
        self._command_dirty = True
        self._sync_paused_until = time.time() + 0.25

    def _on_slider_pressed(self):
        self._dragging = True
        # 开始拖动前，把 command 基线对齐到当前 actual，保证连续控制从真实状态出发
        self._set_vals(self._actual_state, from_user=False, block_signals=True)
        sender = self.sender()
        if sender is not None:
            sender.blockSignals(True)
            sender.setValue(sender.value())
            sender.blockSignals(False)
        self._sync_paused_until = time.time() + 0.5

    def _on_slider_released(self):
        self._dragging = False
        self._mark_dirty()

    def _on_slider(self, _):
        self._refresh_cmd_labels()
        self._mark_dirty()

    def _publish_command_direct(self, vals, status_prefix="Published intent"):
        vals = [float(v) for v in vals[:3]]
        try:
            self.ros_node.send_command(vals)
            self._reset_last_publish_vals = list(vals)
            self._reset_publish_count += 1
            self._show_cmd_labels_direct(vals)
            self.status.setText(
                f"{status_prefix}: {vals[0]:.3f} {vals[1]:.3f} {vals[2]:.3f} mm"
            )
            self._last_send = time.time()
            return True
        except Exception as e:
            self.status.setText(f"ROS 2 publish failed: {e}")
            return False

    def _tick_send(self):
        if self._reset_active:
            return
        now = time.time()
        if (not self._command_dirty) or (now - self._last_send < self._send_period):
            return
        vals = self._current_vals()
        if self._publish_command_direct(vals):
            self._command_dirty = False

    def _tick_ros(self):
        try:
            rclpy.spin_once(self.ros_node, timeout_sec=0)
        except Exception as e:
            self.status.setText(f"ROS 2 spin failed: {e}")
            return

        if self.ros_node.state_stamp > self._last_applied_state_stamp:
            vals = list(self.ros_node.latest_state)
            self._actual_state = vals
            self._last_applied_state_stamp = self.ros_node.state_stamp
            self._refresh_actual_labels(vals)

            now = time.time()
            should_sync_command = (not self._reset_active) and (not self._dragging) and (not self._command_dirty) and (now >= self._sync_paused_until)
            if should_sync_command:
                # 只在用户没有继续编辑时，静默把 command 跟到 actual，作为下一次控制的基线
                self._set_vals(vals, from_user=False, block_signals=True)
                self.status.setText(
                    f"Synced baseline from SOFA actual: {vals[0]:.3f} {vals[1]:.3f} {vals[2]:.3f} mm"
                )
            else:
                if not self._reset_active:
                    self.status.setText(
                        f"SOFA actual: {vals[0]:.3f} {vals[1]:.3f} {vals[2]:.3f} mm"
                    )

        pose_updated = False
        if self.ros_node.imu_stamp > self._last_imu_stamp:
            self._imu_state = list(self.ros_node.latest_imu)
            self._last_imu_stamp = self.ros_node.imu_stamp
            pose_updated = True
        if self.ros_node.tip_stamp > self._last_tip_stamp:
            self._tip_state = list(self.ros_node.latest_tip)
            self._last_tip_stamp = self.ros_node.tip_stamp
            pose_updated = True
        if pose_updated:
            self._refresh_pose_labels()
            self._log_dynamic_sample()

        if self._reset_active:
            self._tick_imu_reset()

    def _set_vals(self, vals, from_user=False, emit_now=False, block_signals=False):
        for i in range(3):
            v = max(self.dL_min, min(self.dL_max, float(vals[i])))
            if block_signals:
                self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(int(round(v * self.scale)))
            if block_signals:
                self.sliders[i].blockSignals(False)
        self._refresh_cmd_labels()
        if from_user:
            self._mark_dirty()
        if emit_now:
            self._last_send = 0.0
            self._tick_send()

    def _clamp_cmd_for_reset(self, v):
        # Reset is allowed to command a small positive release, unlike the normal sliders.
        lo = float(self.dL_min)
        hi = float(self.reset_max_release_mm)
        return max(lo, min(hi, float(v)))

    def _start_phase(self, phase):
        self._reset_phase = phase
        self._reset_phase_t0 = time.time()
        self._reset_last_tune = 0.0
        self._reset_stable_count = 0

    def _zero(self):
        # New reset policy for Teflon-backbone hysteresis:
        # 1) over-release relative to the current IMU-defined home command,
        # 2) return to that home command,
        # 3) use actual IMU roll/pitch to tune until straight.
        #
        # Important: [0,0,0] is NOT assumed to be geometrically straight forever.
        # On the first run _home_cmd is only a guess ([0,0,0]); after a successful
        # IMU reset, _home_cmd becomes the final command that made roll/pitch ~ 0.
        self._dragging = False
        self._command_dirty = False
        self._sync_paused_until = time.time() + 30.0
        self._reset_active = True

        # Over-release is absolute command = IMU-home + common release margin.
        # First run: home=[0,0,0] -> [+6,+6,+6].
        # Later runs: home=[h0,h1,h2] -> [h0+6,h1+6,h2+6], capped by reset_max_release_mm.
        self._reset_release_cmd = [
            self._clamp_cmd_for_reset(float(self._home_cmd[i]) + float(self.reset_extra_release_mm))
            for i in range(3)
        ]
        self._reset_cmd = list(self._home_cmd)
        self._start_phase("release")
        home_note = "calibrated-home" if self._home_valid else "initial zero guess"
        self._publish_command_direct(
            self._reset_release_cmd,
            status_prefix=f"Reset phase 1/3 over-release from {home_note}"
        )

    def _cancel_reset(self):
        self._reset_active = False
        self._reset_phase = "idle"
        self._sync_paused_until = time.time() + 0.5
        self.status.setText("IMU reset cancelled. Current command is left unchanged.")

    def _tick_imu_reset(self):
        now = time.time()

        if self._reset_phase == "release":
            if now - self._reset_phase_t0 >= self.reset_release_hold_s:
                # Keep the robot at the over-released command. Do not return to [0,0,0],
                # because that can immediately re-introduce the Teflon backbone hysteresis.
                self._reset_cmd = list(self._reset_release_cmd)
                self._publish_command_direct(self._reset_cmd, status_prefix="Reset phase 2/2 IMU tune starts from over-release")
                self._start_phase("imu_tune")
            return

        if self._reset_phase != "imu_tune":
            return

        if now - self._reset_phase_t0 > self.reset_max_tune_s:
            self._finish_imu_reset(success=False, reason="timeout")
            return

        if now - self._reset_last_tune < self.reset_tune_period_s:
            return
        self._reset_last_tune = now

        if now - self._last_imu_stamp > 1.0:
            self.status.setText("Reset phase 2/2 waiting for fresh IMU data...")
            return

        roll = float(self._imu_state[0]) * float(self.reset_roll_sign)
        pitch = float(self._imu_state[1]) * float(self.reset_pitch_sign)
        err_abs = max(abs(roll), abs(pitch))

        if err_abs <= self.reset_imu_tol_deg:
            self._reset_stable_count += 1
            if self._reset_stable_count >= self.reset_stable_required:
                self._finish_imu_reset(success=True, reason="imu tolerance reached")
            else:
                self.status.setText(
                    f"Reset stabilizing IMU: A={float(self._imu_state[0]):.2f}, B={float(self._imu_state[1]):.2f} deg"
                )
            return
        self._reset_stable_count = 0

        # Three-cable differential correction. Cable angles are 0/120/240 deg.
        # Coordinate convention for the physical robot:
        #   cable 0 lies on +X axis; roll is rotation about X; pitch is rotation about Y.
        # A tendon on +X mainly changes pitch (Y-axis rotation), while tendons with
        # +/-Y components mainly change roll (X-axis rotation).
        # Positive projection means that cable should be pulled; command is negative dL.
        import math
        phis = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
        raw_pull = []
        for phi in phis:
            projection = -(pitch * math.cos(phi) + roll * math.sin(phi))
            raw_pull.append(max(0.0, projection * self.reset_tune_gain_mm_per_deg))

        # Remove common-mode pull so at least one cable stays fully released.
        common = min(raw_pull)
        raw_pull = [p - common for p in raw_pull]
        max_pull = max(raw_pull) if raw_pull else 0.0
        if max_pull > self.reset_max_pull_mm and max_pull > 1e-9:
            raw_pull = [p * self.reset_max_pull_mm / max_pull for p in raw_pull]

        target_cmd = [self._reset_release_cmd[i] - raw_pull[i] for i in range(3)]
        # Low-pass the correction to avoid fighting IMU noise and mechanical hysteresis.
        alpha = 0.35
        self._reset_cmd = [
            self._clamp_cmd_for_reset((1.0 - alpha) * self._reset_cmd[i] + alpha * target_cmd[i])
            for i in range(3)
        ]
        self._set_vals(self._reset_cmd, from_user=False, block_signals=True)
        self._publish_command_direct(
            self._reset_cmd,
            status_prefix=f"Reset phase 3/3 IMU tune roll={roll:.2f}, pitch={pitch:.2f}"
        )

    def _finish_imu_reset(self, success, reason):
        self._reset_active = False
        self._reset_phase = "idle"
        self._set_vals(self._reset_cmd, from_user=False, block_signals=True)
        self._sync_paused_until = time.time() + 0.5
        roll, pitch = self._imu_state[:2]
        if success:
            # This final command is now the real straight-home command for later resets.
            self._home_cmd = list(self._reset_cmd)
            self._home_valid = True
            self.status.setText(
                f"IMU reset complete: roll={roll:.2f}, pitch={pitch:.2f} deg; calibrated IMU home = {self._home_cmd[0]:.3f}, {self._home_cmd[1]:.3f}, {self._home_cmd[2]:.3f} mm"
            )
        else:
            self.status.setText(
                f"IMU reset stopped by {reason}: roll={roll:.2f}, pitch={pitch:.2f} deg; kept best command {self._reset_cmd[0]:.3f}, {self._reset_cmd[1]:.3f}, {self._reset_cmd[2]:.3f} mm"
            )

    def _set_current_as_home(self):
        # Use this only after you have physically confirmed that the robot is straight,
        # or immediately after a successful IMU reset. It changes the reference used by
        # future reset over-release steps.
        self._home_cmd = list(self._current_vals())
        self._home_valid = True
        self.status.setText(
            f"Set current command as IMU home: {self._home_cmd[0]:.3f}, {self._home_cmd[1]:.3f}, {self._home_cmd[2]:.3f} mm"
        )

    def _adopt_actual(self):
        self._set_vals(self._actual_state, from_user=False, block_signals=True)
        self.status.setText("Adopted latest SOFA actual as command baseline.")


def main():
    rclpy.init(args=sys.argv)
    ros_node = RosInterface(
        cmd_topic='/robot/tendon_commands',
        state_topic='/robot/tendon_state',
        imu_topic='/robot/imu_euler',
        tip_topic='/robot/tip_euler',
    )
    app = QtWidgets.QApplication(sys.argv)
    w = TendonConsole(ros_node=ros_node, dL_min=-60.0, dL_max=0.0, send_hz=30, spin_hz=100)
    w.resize(920, 420)
    w.show()
    exit_code = app.exec()
    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
