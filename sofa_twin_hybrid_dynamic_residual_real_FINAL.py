"""
SOFA scene: Digital Twin (ROS 2 Subscriber + Interpolation)
SCRIPTED (no keyboard).
Ultra-compatible: avoids strict RequiredPlugin lists, avoids damping attributes.
Subscribes to ROS 2 topic: /robot/tendon_commands (std_msgs/Float32MultiArray)
"""
import math
import sys
import types
import time

try:
    import socket
except Exception:
    socket = None

try:
    import serial
except Exception:
    serial = None

# ---- ROS 2 Imports ----
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# ---- Optional: stub matplotlib so importing sim_only won't fail in SOFA python
if "matplotlib" not in sys.modules:
    mpl = types.ModuleType("matplotlib")
    mpl_pyplot = types.ModuleType("matplotlib.pyplot")
    def _noop(*a, **k): return None
    for n in ["plot","figure","show","subplots","subplot","title","xlabel","ylabel",
              "legend","grid","tight_layout","savefig","close"]:
        setattr(mpl_pyplot, n, _noop)
    mpl.pyplot = mpl_pyplot
    sys.modules["matplotlib"] = mpl
    sys.modules["matplotlib.pyplot"] = mpl_pyplot

# ---- Import solver module (user-provided)
SIM = None
try:
    import sim_only as SIM  # must be in same folder / python path
except Exception as e:
    SIM = None
    print("[WARN] Could not import sim_only:", e)

import Sofa
import Sofa.Core

try:
    from dynamic_residual_model import DynamicResidualModel
except Exception as e:
    DynamicResidualModel = None
    print("[WARN] Could not import DynamicResidualModel:", e)


def _try_required_plugins(root, names):
    for nm in names:
        try:
            root.addObject('RequiredPlugin', name=nm)
        except Exception as e:
            print(f"[WARN] RequiredPlugin missing/failed: {nm} ({e})")

def _safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)

def _clamp_deg_range(lo, hi, default_lo=30.0, default_hi=60.0):
    lo = _safe_float(lo, default_lo)
    hi = _safe_float(hi, default_hi)
    if hi < lo:
        lo, hi = hi, lo
    if abs(hi - lo) < 1e-6:
        hi = lo + 1.0
    return lo, hi

def _uniform_delta_list(n_disks, helix_total_rad):
    if n_disks <= 1:
        return [0.0]
    d = helix_total_rad / (n_disks - 1)
    return [d] * (n_disks - 1)

def _cum_theta_at_disks(delta_list, n_disks):
    thetas = [0.0] * n_disks
    for j in range(1, n_disks):
        dj = delta_list[j - 1] if j - 1 < len(delta_list) else 0.0
        thetas[j] = thetas[j - 1] + float(dj)
    return thetas

def _ensure_R_list(R_list, n):
    I = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if not isinstance(R_list, (list, tuple)) or len(R_list) == 0:
        return [I for _ in range(n)]
    out = []
    for i in range(n):
        if i < len(R_list) and isinstance(R_list[i], (list, tuple)) and len(R_list[i]) == 3:
            out.append(R_list[i])
        else:
            out.append(I)
    return out


class SolverDrivenController(Sofa.Core.Controller):
    """
    ROS 2 Digital Twin Controller.
    Subscribes to /robot/tendon_commands and smoothly interpolates tendon lengths.
    """

    def _log(self, msg, level="info"):
        tag = getattr(self, "name", None) or "TwinCtrl"
        try:
            if level == "info" and hasattr(Sofa, "msg_info"):
                Sofa.msg_info(tag, msg)
                return
            if level == "warning" and hasattr(Sofa, "msg_warning"):
                Sofa.msg_warning(tag, msg)
                return
            if level == "error" and hasattr(Sofa, "msg_error"):
                Sofa.msg_error(tag, msg)
                return
        except Exception:
            pass
        print(f"[{level.upper()}] {tag}: {msg}")

    def __init__(self, root, vis, **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.vis = vis

        # ---- Parameters
        self.n_disks = int(kwargs.get("n_disks", 6))
        self.length_m = float(kwargs.get("length_m", 0.30))
        self.hole_r_m = float(kwargs.get("hole_r_m", 0.020))
        self.helix_total_rad = float(kwargs.get("helix_total_rad", 6.0 * math.pi))
        self.mu_friction = float(kwargs.get("mu_friction", 0.8))
        self.enforce_no_twist = bool(kwargs.get("enforce_no_twist", True))
        self.update_every_s = float(kwargs.get("update_every_s", 0.05)) # 更快的更新频率换取丝滑渲染

        # ==== 数字孪生状态变量 ====
        self.dL_mm = [0.0, 0.0, 0.0]           # 当前真正送入求解器的输入位移
        self.target_dL_mm = [0.0, 0.0, 0.0]    # 用户/Qt/ROS 期望的“真实目标位移”
        self.comp_target_dL_mm = [0.0, 0.0, 0.0]  # 经过补偿后，真正想送入求解器的目标
        self.comp_bias_mm = [0.0, 0.0, 0.0]    # 为抵消模型误差而自动学习的补偿偏置
        self.max_speed_mm_s = 2.0             # 限制位移最大速度 (mm/s)，彻底消除跳变
        self.comp_gain = float(kwargs.get("comp_gain", 0.35))
        self.comp_deadband_mm = float(kwargs.get("comp_deadband_mm", 0.15))
        self.max_comp_mm = float(kwargs.get("max_comp_mm", 25.0))
        self.max_target_mm = float(kwargs.get("max_target_mm", 60.0))

        # ==== 协同控制增量功能（最小侵入添加） ====
        self.initial_cable_lengths = [0.0, 0.0, 0.0]   # 保留：视觉几何长度(m)
        self.initial_solver_lengths = [0.0, 0.0, 0.0]  # 新增：求解器内部定义下的基准长度(m)
        self.real_dL_mm = [0.0, 0.0, 0.0]              # 基于求解器内部长度定义得到的真实位移(mm)
        self.mm_to_steps = float(kwargs.get("mm_to_steps", 80.0))
        self._last_solver_P = None
        self._last_solver_R = None

        self.use_udp = bool(kwargs.get("use_udp", False))
        self.use_serial = bool(kwargs.get("use_serial", False))
        self.udp_bind_ip = str(kwargs.get("udp_bind_ip", "127.0.0.1"))
        self.udp_cmd_port = int(kwargs.get("udp_cmd_port", 9999))
        self.udp_feedback_ip = str(kwargs.get("udp_feedback_ip", "127.0.0.1"))
        self.udp_feedback_port = int(kwargs.get("udp_feedback_port", 9998))
        self.serial_port = str(kwargs.get("serial_port", "COM3"))
        self.serial_baud = int(kwargs.get("serial_baud", 115200))

        self.seed = int(kwargs.get("seed", 0))
        self.delta_list = None
        self.theta_at_disk = None

        self._t = 0.0
        self._accum = 0.0

        # ==== 初始化协同控制通信（默认关闭，避免影响原有稳定流程） ====
        self.ser = None
        if self.use_serial:
            if serial is None:
                self._log("pyserial not available, serial disabled", level="warning")
            else:
                try:
                    self.ser = serial.Serial(self.serial_port, self.serial_baud, timeout=0.01)
                    self._log(f"Serial connected: {self.serial_port}@{self.serial_baud}")
                except Exception as e:
                    self.ser = None
                    self._log(f"Serial init failed: {e}", level="warning")

        self.sock_in = None
        self.sock_out = None
        self.qt_addr = (self.udp_feedback_ip, self.udp_feedback_port)
        if self.use_udp:
            if socket is None:
                self._log("socket module unavailable, UDP disabled", level="warning")
            else:
                try:
                    self.sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    self.sock_in.bind((self.udp_bind_ip, self.udp_cmd_port))
                    self.sock_in.setblocking(False)
                    self.sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    self._log(f"UDP ready: recv {self.udp_bind_ip}:{self.udp_cmd_port} -> send {self.udp_feedback_ip}:{self.udp_feedback_port}")
                except Exception as e:
                    self.sock_in = None
                    self.sock_out = None
                    self._log(f"UDP init failed: {e}", level="warning")

        # ==== 初始化 ROS 2 节点 ====
        self.ros_node = None
        self.ros_cmd_topic = str(kwargs.get("ros_cmd_topic", "/robot/tendon_commands"))
        self.ros_state_topic = str(kwargs.get("ros_state_topic", "/robot/tendon_state"))
        self.ros_feedback_topic = str(kwargs.get("ros_feedback_topic", self.ros_state_topic))
        self.ros_tip_topic = str(kwargs.get("ros_tip_topic", "/robot/tip_euler"))
        # Tip position topic used by validation_monitor.py.
        # Unit: metre, message: std_msgs/Float32MultiArray, data = [x, y, z]
        self.ros_tip_position_topic = str(kwargs.get("ros_tip_position_topic", "/robot/tip_position"))
        self._last_published_state = None
        self._last_published_tip_euler = None
        self._last_published_tip_position = None
        self.tip_euler_deg = [0.0, 0.0]
        self.tip_position_m = [0.0, 0.0, 0.0]

        # ==== Hybrid physics-learning dynamic residual model ====
        # Original SOFA/mechanics model remains unchanged.
        # The learning module only predicts residual = IMU - SOFA.
        self.latest_imu_euler_deg = [0.0, 0.0]
        self.has_imu_feedback = False
        self.hybrid_tip_euler_deg = [0.0, 0.0]
        self.hybrid_residual_deg = [0.0, 0.0]
        self._last_published_hybrid_tip_euler = None
        self._last_published_hybrid_error = None

        self.use_dynamic_residual = bool(kwargs.get("use_dynamic_residual", True))
        self.dynamic_model_path = str(kwargs.get("dynamic_model_path", "dynamic_residual_model.json"))
        self.dynamic_model = None
        if self.use_dynamic_residual and DynamicResidualModel is not None:
            try:
                self.dynamic_model = DynamicResidualModel(self.dynamic_model_path)
                self._log(f"Dynamic residual model loaded: {self.dynamic_model_path}")
            except Exception as e:
                self.dynamic_model = None
                self._log(f"Dynamic residual model disabled: {e}", level="warning")
        try:
            if not rclpy.ok():
                rclpy.init()

            self.ros_node = rclpy.create_node('sofa_twin_node')
            self.sub = self.ros_node.create_subscription(
                Float32MultiArray,
                self.ros_cmd_topic,
                self._ros_cmd_callback,
                10
            )
            self.state_pub = self.ros_node.create_publisher(
                Float32MultiArray,
                self.ros_state_topic,
                10
            )
            if self.ros_feedback_topic != self.ros_state_topic:
                self.feedback_pub = self.ros_node.create_publisher(
                    Float32MultiArray,
                    self.ros_feedback_topic,
                    10
                )
            else:
                self.feedback_pub = None
            self.tip_pub = self.ros_node.create_publisher(
                Float32MultiArray,
                self.ros_tip_topic,
                10
            )
            self.tip_position_pub = self.ros_node.create_publisher(
                Float32MultiArray,
                self.ros_tip_position_topic,
                10
            )
            self.ros_imu_topic = str(kwargs.get("ros_imu_topic", "/robot/imu_euler"))
            self.ros_hybrid_tip_topic = str(kwargs.get("ros_hybrid_tip_topic", "/robot/tip_euler_hybrid"))
            self.ros_hybrid_error_topic = str(kwargs.get("ros_hybrid_error_topic", "/robot/tip_euler_hybrid_error"))

            self.imu_sub = self.ros_node.create_subscription(
                Float32MultiArray,
                self.ros_imu_topic,
                self._ros_imu_callback,
                10
            )

            self.hybrid_tip_pub = self.ros_node.create_publisher(
                Float32MultiArray,
                self.ros_hybrid_tip_topic,
                10
            )

            self.hybrid_error_pub = self.ros_node.create_publisher(
                Float32MultiArray,
                self.ros_hybrid_error_topic,
                10
            )
            self._log(f"ROS 2 Node Initialized! cmd: {self.ros_cmd_topic} | state: {self.ros_state_topic} | tip: {self.ros_tip_topic}")
        except Exception as e:
            self._log(f"Failed to initialize ROS 2: {e}", level="error")
            self.state_pub = None
            self.feedback_pub = None
            self.tip_pub = None
            self.tip_position_pub = None

        # build initial assembly
        self._build_random_assembly()
        # initial solve+update
        self._update_from_solver()

        try:
            self.initial_cable_lengths = self._measure_cable_lengths()
            if self._last_solver_P is not None and self._last_solver_R is not None:
                solver_lengths = self._solver_calc_lengths(self._last_solver_P, self._last_solver_R)
                if solver_lengths is not None:
                    self.initial_solver_lengths = list(solver_lengths)
        except Exception as e:
            self._log(f"Initial cable length measurement failed: {e}", level="warning")

        self._refresh_comp_target()

        try:
            self._update_from_solver()
            self._publish_real_state(force=True)
            self._publish_tip_euler(force=True)
            self._publish_tip_position(force=True)
            self._publish_hybrid_tip_euler(force=True)
            self._publish_hybrid_error(force=True)
        except Exception:
            pass

    def _ros_cmd_callback(self, msg):
        """ROS 2 Callback: 仅更新目标位置"""
        if len(msg.data) >= 3:
            self.target_dL_mm[0] = self._clamp_mm(float(msg.data[0]))
            self.target_dL_mm[1] = self._clamp_mm(float(msg.data[1]))
            self.target_dL_mm[2] = self._clamp_mm(float(msg.data[2]))
            self._refresh_comp_target()

    def _ros_imu_callback(self, msg):
        """Read IMU Euler feedback for hybrid residual compensation."""
        if len(msg.data) >= 2:
            try:
                self.latest_imu_euler_deg[0] = float(msg.data[0])
                self.latest_imu_euler_deg[1] = float(msg.data[1])
                self.has_imu_feedback = True
            except Exception:
                pass

    def _measure_cable_lengths(self):
        """测量当前 SOFA 里三根线缆的 3D 几何长度。"""
        lengths = [0.0, 0.0, 0.0]
        if not hasattr(self.vis, "cable_mo"):
            return lengths

        for k in range(min(3, len(self.vis.cable_mo))):
            try:
                pts = self.vis.cable_mo[k].position.value
            except Exception:
                pts = []
            total_len = 0.0
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                dx = float(p1[0]) - float(p2[0])
                dy = float(p1[1]) - float(p2[1])
                dz = float(p1[2]) - float(p2[2])
                total_len += math.sqrt(dx * dx + dy * dy + dz * dz)
            lengths[k] = total_len
        return lengths

    def _solver_calc_lengths(self, p_list, R_list):
        """严格复用 sim_only.solve_shape_from_dL 里的长度定义，而不是用可视化折线长度代替。"""
        if SIM is None or not hasattr(SIM, "RobotParams") or not hasattr(SIM, "build_alpha_disks"):
            return None
        try:
            params = SIM.RobotParams()
            alpha_disks = SIM.build_alpha_disks(params, self.delta_list)
            ls = [0.0, 0.0, 0.0]
            prev = [[0.0, 0.0, 0.0] for _ in range(3)]
            for k in range(3):
                a0 = float(alpha_disks[0, k])
                r0 = [params.r_disk * math.cos(a0), params.r_disk * math.sin(a0), 0.0]
                R0 = R_list[0]
                prev[k] = [
                    float(p_list[0][0] + R0[0][0] * r0[0] + R0[0][1] * r0[1] + R0[0][2] * r0[2]),
                    float(p_list[0][1] + R0[1][0] * r0[0] + R0[1][1] * r0[1] + R0[1][2] * r0[2]),
                    float(p_list[0][2] + R0[2][0] * r0[0] + R0[2][1] * r0[1] + R0[2][2] * r0[2]),
                ]
            Nn = len(p_list)
            if Nn <= 1:
                return ls
            for i in range(1, Nn):
                s = params.L * float(i) / float(Nn - 1)
                j = int(math.floor(s / params.disk_spacing))
                j = max(0, min(params.num_disks - 1, j))
                for k in range(3):
                    a0 = float(alpha_disks[j, k])
                    r0 = [params.r_disk * math.cos(a0), params.r_disk * math.sin(a0), 0.0]
                    Ri = R_list[i]
                    curr = [
                        float(p_list[i][0] + Ri[0][0] * r0[0] + Ri[0][1] * r0[1] + Ri[0][2] * r0[2]),
                        float(p_list[i][1] + Ri[1][0] * r0[0] + Ri[1][1] * r0[1] + Ri[1][2] * r0[2]),
                        float(p_list[i][2] + Ri[2][0] * r0[0] + Ri[2][1] * r0[1] + Ri[2][2] * r0[2]),
                    ]
                    dx = curr[0] - prev[k][0]
                    dy = curr[1] - prev[k][1]
                    dz = curr[2] - prev[k][2]
                    ls[k] += math.sqrt(dx * dx + dy * dy + dz * dz)
                    prev[k] = curr
            return ls
        except Exception as e:
            self._log(f"solver length calc failed: {e}", level="warning")
            return None

    def _clamp_mm(self, x):
        lim = abs(float(self.max_target_mm))
        return max(-lim, min(lim, float(x)))

    def _is_active_cmd(self, v):
        return float(v) < -1e-6

    def _refresh_comp_target(self):
        for i in range(3):
            if self._is_active_cmd(self.target_dL_mm[i]):
                self.comp_bias_mm[i] = max(-self.max_comp_mm, min(self.max_comp_mm, float(self.comp_bias_mm[i])))
                self.comp_target_dL_mm[i] = self._clamp_mm(self.target_dL_mm[i] + self.comp_bias_mm[i])
            else:
                # 非激活通道保持原命令，不做 outer-loop 补偿
                # 否则会把 sim_only 里“被动通道=零张力”的假设破坏掉
                self.comp_bias_mm[i] = 0.0
                self.comp_target_dL_mm[i] = self._clamp_mm(self.target_dL_mm[i])

    def _update_compensation_from_error(self):
        # 只对“主动拉动通道”补偿；被动通道允许自然协同变化，不追 real==target==0
        changed = False
        for i in range(3):
            if not self._is_active_cmd(self.target_dL_mm[i]):
                continue
            err = float(self.target_dL_mm[i]) - float(self.real_dL_mm[i])
            if abs(err) <= self.comp_deadband_mm:
                continue
            self.comp_bias_mm[i] += self.comp_gain * err
            self.comp_bias_mm[i] = max(-self.max_comp_mm, min(self.max_comp_mm, self.comp_bias_mm[i]))
            changed = True
        if changed:
            self._refresh_comp_target()
        return changed


    def _publish_real_state(self, force=False):
        if self.ros_node is None or self.state_pub is None:
            return
        vals = [float(self.real_dL_mm[0]), float(self.real_dL_mm[1]), float(self.real_dL_mm[2])]
        rounded = tuple(round(v, 3) for v in vals)
        if (not force) and self._last_published_state == rounded:
            return
        msg = Float32MultiArray()
        msg.data = vals
        try:
            self.state_pub.publish(msg)
            if self.feedback_pub is not None:
                self.feedback_pub.publish(msg)
            self._last_published_state = rounded
        except Exception as e:
            self._log(f"Failed to publish real tendon state: {e}", level="warning")


    def _publish_tip_euler(self, force=False):
        if self.ros_node is None or self.tip_pub is None:
            return
        vals = [float(self.tip_euler_deg[0]), float(self.tip_euler_deg[1])]
        rounded = tuple(round(v, 3) for v in vals)
        if (not force) and self._last_published_tip_euler == rounded:
            return
        msg = Float32MultiArray()
        msg.data = vals
        try:
            self.tip_pub.publish(msg)
            self._last_published_tip_euler = rounded
        except Exception as e:
            self._log(f"Failed to publish tip euler: {e}", level="warning")

    def _publish_tip_position(self, force=False):
        """
        Publish SOFA tip centre position for trajectory validation.

        Important:
        - Always publish when this function is called. Do not suppress repeated
          values, otherwise validation_monitor.py will show stale tip_pos_age
          when the robot is stationary or when motion is small.
        - Unit: metre.
        - Message: std_msgs/Float32MultiArray, data = [x, y, z].
        """
        if self.ros_node is None:
            return
        if not hasattr(self, "tip_position_pub") or self.tip_position_pub is None:
            return

        vals = [
            float(self.tip_position_m[0]),
            float(self.tip_position_m[1]),
            float(self.tip_position_m[2]),
        ]

        msg = Float32MultiArray()
        msg.data = vals
        try:
            self.tip_position_pub.publish(msg)
            self._last_published_tip_position = tuple(round(v, 6) for v in vals)
        except Exception as e:
            self._log(f"Failed to publish tip position: {e}", level="warning")

    def _publish_hybrid_tip_euler(self, force=False):
        if self.ros_node is None:
            return
        if not hasattr(self, "hybrid_tip_pub") or self.hybrid_tip_pub is None:
            return
        vals = [float(self.hybrid_tip_euler_deg[0]), float(self.hybrid_tip_euler_deg[1])]
        rounded = tuple(round(v, 3) for v in vals)
        if (not force) and self._last_published_hybrid_tip_euler == rounded:
            return
        msg = Float32MultiArray()
        msg.data = vals
        try:
            self.hybrid_tip_pub.publish(msg)
            self._last_published_hybrid_tip_euler = rounded
        except Exception as e:
            self._log(f"Failed to publish hybrid tip euler: {e}", level="warning")

    def _publish_hybrid_error(self, force=False):
        if self.ros_node is None:
            return
        if not hasattr(self, "hybrid_error_pub") or self.hybrid_error_pub is None:
            return
        vals = [
            float(self.hybrid_tip_euler_deg[0] - self.latest_imu_euler_deg[0]),
            float(self.hybrid_tip_euler_deg[1] - self.latest_imu_euler_deg[1]),
        ]
        rounded = tuple(round(v, 3) for v in vals)
        if (not force) and self._last_published_hybrid_error == rounded:
            return
        msg = Float32MultiArray()
        msg.data = vals
        try:
            self.hybrid_error_pub.publish(msg)
            self._last_published_hybrid_error = rounded
        except Exception as e:
            self._log(f"Failed to publish hybrid error: {e}", level="warning")

    def _build_random_assembly(self):
        deg_lo, deg_hi = 30.0, 60.0
        if SIM is not None and hasattr(SIM, "RobotParams"):
            try:
                p = SIM.RobotParams()
                deg_lo, deg_hi = _clamp_deg_range(30.0, 60.0)
            except Exception:
                pass

        self.delta_list = None
        if SIM is not None and hasattr(SIM, "random_delta_list") and hasattr(SIM, "RobotParams"):
            try:
                params = SIM.RobotParams()
                self.delta_list = SIM.random_delta_list(
                    params.num_disks,
                    total_angle=float(getattr(params, "helix_angle", self.helix_total_rad)),
                    deg_min=float(deg_lo),
                    deg_max=float(deg_hi),
                    seed=int(self.seed)
                )
            except Exception as e:
                print("[WARN] random_delta_list failed:", e)

        if self.delta_list is None or (hasattr(self.delta_list, '__len__') and len(self.delta_list) == 0):
            self.delta_list = _uniform_delta_list(self.n_disks, self.helix_total_rad)

        self.theta_at_disk = _cum_theta_at_disks(self.delta_list, self.n_disks)

    def _call_solver(self):
        centers = [[0.0, 0.0, (self.length_m * (j / (self.n_disks - 1)) if self.n_disks > 1 else 0.0)]
                   for j in range(self.n_disks)]
        R_list = None
        full_P = None
        full_R = None

        if SIM is None:
            return centers, _ensure_R_list(R_list, self.n_disks), full_P, full_R

        if hasattr(SIM, "solve_shape_from_dL") and hasattr(SIM, "RobotParams"):
            try:
                params = SIM.RobotParams()
                # mm -> m
                dL_m = [x * 1e-3 for x in self.dL_mm]

                out = None
                try:
                    out = SIM.solve_shape_from_dL(
                        dL_m,
                        params,
                        delta_list=self.delta_list,
                        enforce_no_twist=self.enforce_no_twist,
                        mu_friction=self.mu_friction
                    )
                except TypeError:
                    out = SIM.solve_shape_from_dL(dL_m, params, self.delta_list)

                if isinstance(out, dict):
                    P = out.get("P") or out.get("p") or out.get("centerline")
                    R_list = out.get("R") or out.get("R_list") or out.get("rotations")
                elif isinstance(out, (list, tuple)):
                    P = out[0] if len(out) > 0 else None
                    R_list = out[1] if len(out) > 1 else None
                else:
                    P, R_list = None, None

                full_P = P
                full_R = R_list

                if P is not None and len(P) >= self.n_disks:
                    if len(P) == self.n_disks:
                        centers = [list(map(float, p)) for p in P]
                    else:
                        idxs = [round(i * (len(P) - 1) / (self.n_disks - 1)) for i in range(self.n_disks)] if self.n_disks > 1 else [0]
                        centers = [list(map(float, P[i])) for i in idxs]
            except Exception as e:
                print("[WARN] solve_shape_from_dL failed:", e)

        return centers, _ensure_R_list(R_list, self.n_disks), full_P, full_R

    def _update_from_solver(self):
        centers, R_list, full_P, full_R = self._call_solver()

        # Anchor base to origin (VISUAL ONLY)
        if centers and len(centers) > 0:
            bx, by, bz = centers[0]
            centers = [[p[0] - bx, p[1] - by, p[2] - bz] for p in centers]

        # Robustness
        try:
            for p in centers:
                if (not math.isfinite(p[0])) or (not math.isfinite(p[1])) or (not math.isfinite(p[2])):
                    return
                if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
                    return
        except Exception:
            return

        # Use the actual solver tip point when the dense centreline full_P is available.
        # This is more reliable than using the last visual disk sample, especially when
        # the solver returns more centreline points than the six disk centres.
        if full_P is not None:
            try:
                p0 = full_P[0]
                pt = full_P[-1]
                self.tip_position_m = [
                    float(pt[0]) - float(p0[0]),
                    float(pt[1]) - float(p0[1]),
                    float(pt[2]) - float(p0[2]),
                ]
            except Exception:
                if centers:
                    try:
                        self.tip_position_m = [
                            float(centers[-1][0]),
                            float(centers[-1][1]),
                            float(centers[-1][2]),
                        ]
                    except Exception:
                        self.tip_position_m = [0.0, 0.0, 0.0]
        elif centers:
            try:
                self.tip_position_m = [
                    float(centers[-1][0]),
                    float(centers[-1][1]),
                    float(centers[-1][2]),
                ]
            except Exception:
                self.tip_position_m = [0.0, 0.0, 0.0]

        self.vis.backbone_mo.position.value = centers
        self._update_disks(centers)
        self._update_cables(centers, R_list)
        self._update_tip_pose_visual(centers)
        self._update_tip_euler(centers)
        self._update_hybrid_tip_euler()

        self._last_solver_P = full_P
        self._last_solver_R = full_R

        # 视觉几何长度仅保留给调试；真实控制量按 sim_only 内部长度定义计算
        try:
            current_lengths = self._measure_cable_lengths()
            self.visual_dL_mm = [(current_lengths[k] - self.initial_cable_lengths[k]) * 1000.0 for k in range(3)]
        except Exception:
            self.visual_dL_mm = [0.0, 0.0, 0.0]

        try:
            if full_P is not None and full_R is not None and any(abs(x) > 0.0 for x in self.initial_solver_lengths):
                solver_lengths = self._solver_calc_lengths(full_P, full_R)
                if solver_lengths is not None:
                    for k in range(3):
                        self.real_dL_mm[k] = (float(solver_lengths[k]) - float(self.initial_solver_lengths[k])) * 1000.0
            elif full_P is not None and full_R is not None:
                solver_lengths = self._solver_calc_lengths(full_P, full_R)
                if solver_lengths is not None:
                    self.initial_solver_lengths = list(solver_lengths)
                    self.real_dL_mm = [0.0, 0.0, 0.0]
        except Exception as e:
            self._log(f"internal solver length update failed: {e}", level="warning")

    def _compute_frames(self, centers):
        def _dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
        def _norm(v):
            l = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            if l < 1e-12: return [0.0, 0.0, 1.0], 0.0
            return [v[0]/l, v[1]/l, v[2]/l], l
        def _cross(a, b):
            return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
        def _choose_ref(n):
            ref = [0.0, 0.0, 1.0]
            if abs(_dot(n, ref)) > 0.95: ref = [1.0, 0.0, 0.0]
            return ref

        N = self.n_disks
        if N <= 0: return [], [], []

        n_list = []
        for j in range(N):
            if N == 1: n = [0.0, 0.0, 1.0]
            elif j == 0: n = [centers[1][0] - centers[0][0], centers[1][1] - centers[0][1], centers[1][2] - centers[0][2]]
            elif j == N - 1: n = [centers[-1][0] - centers[-2][0], centers[-1][1] - centers[-2][1], centers[-1][2] - centers[-2][2]]
            else: n = [centers[j+1][0] - centers[j-1][0], centers[j+1][1] - centers[j-1][1], centers[j+1][2] - centers[j-1][2]]
            n, _ = _norm(n)
            n_list.append(n)

        ref0 = [1.0, 0.0, 0.0]
        proj0 = _dot(ref0, n_list[0])
        u0 = [ref0[0] - proj0*n_list[0][0], ref0[1] - proj0*n_list[0][1], ref0[2] - proj0*n_list[0][2]]
        u0, ln = _norm(u0)
        if ln < 1e-12:
            ref0 = [0.0, 1.0, 0.0]
            proj0 = _dot(ref0, n_list[0])
            u0 = [ref0[0] - proj0*n_list[0][0], ref0[1] - proj0*n_list[0][1], ref0[2] - proj0*n_list[0][2]]
            u0, _ = _norm(u0)
        v0 = _cross(n_list[0], u0)

        u_list, v_list = [u0], [v0]

        for j in range(1, N):
            n = n_list[j]
            u_prev = u_list[j-1]
            proj = _dot(u_prev, n)
            u = [u_prev[0] - proj*n[0], u_prev[1] - proj*n[1], u_prev[2] - proj*n[2]]
            u, ln = _norm(u)

            if ln < 1e-12:
                ref = _choose_ref(n)
                u, _ = _norm(_cross(ref, n))
            if _dot(u, u_prev) < 0.0:
                u = [-u[0], -u[1], -u[2]]
            v = _cross(n, u)
            u_list.append(u)
            v_list.append(v)

        return n_list, u_list, v_list

    def _update_disks(self, centers):
        ringN = self.vis.ringN
        pos = []
        _, u_list, v_list = self._compute_frames(centers)
        start = getattr(self.vis, 'start_disk', 0)

        for j in range(start, self.n_disks):
            cx, cy, cz = centers[j]
            if j == 0:
                u, v = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
            else:
                u, v = u_list[j], v_list[j]

            for t in range(ringN):
                ang = 2.0 * math.pi * t / (ringN - 1)
                ca, sa = math.cos(ang), math.sin(ang)
                x = cx + self.vis.disc_r_m * (ca*u[0] + sa*v[0])
                y = cy + self.vis.disc_r_m * (ca*u[1] + sa*v[1])
                z = cz + self.vis.disc_r_m * (ca*u[2] + sa*v[2])
                pos.append([x, y, z])
        self.vis.disks_mo.position.value = pos

    def _update_cables(self, centers, R_list):
        base_phis = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
        _, u_list, v_list = self._compute_frames(centers)

        for k in range(3):
            pts = []
            start = getattr(self.vis, 'start_disk', 0)
            for j in range(start, self.n_disks):
                theta = self.theta_at_disk[j] if self.theta_at_disk else 0.0
                ang = base_phis[k] + theta
                ca, sa = math.cos(ang), math.sin(ang)
                u, v = u_list[j], v_list[j]
                off = [self.hole_r_m * (ca*u[0] + sa*v[0]),
                       self.hole_r_m * (ca*u[1] + sa*v[1]),
                       self.hole_r_m * (ca*u[2] + sa*v[2])]
                pts.append([centers[j][0] + off[0], centers[j][1] + off[1], centers[j][2] + off[2]])
            self.vis.cable_mo[k].position.value = pts

    def _update_tip_pose_visual(self, centers):
        if not hasattr(self, "vis") or self.vis is None: return
        if not hasattr(self.vis, "tip_point_mo") or not hasattr(self.vis, "tip_axis_mo"): return
        if not centers: return

        tip = centers[-1]
        n_list, u_list, v_list = self._compute_frames(centers)
        if not n_list or not u_list or not v_list: return

        n, u, v = n_list[-1], u_list[-1], v_list[-1]
        L = float(getattr(self.vis, "tip_axis_len_m", 0.05))

        try:
            self.vis.tip_point_mo.position.value = [tip]
            self.vis.tip_axis_mo[0].position.value = [tip, [tip[0] + L * u[0], tip[1] + L * u[1], tip[2] + L * u[2]]]
            self.vis.tip_axis_mo[1].position.value = [tip, [tip[0] + L * v[0], tip[1] + L * v[1], tip[2] + L * v[2]]]
            self.vis.tip_axis_mo[2].position.value = [tip, [tip[0] + L * n[0], tip[1] + L * n[1], tip[2] + L * n[2]]]
        except Exception:
            pass


    def _update_tip_euler(self, centers):
        if not centers:
            return
        n_list, u_list, v_list = self._compute_frames(centers)
        if not n_list or not u_list or not v_list:
            return

        n = n_list[-1]
        u = u_list[-1]
        v = v_list[-1]

        # Rotation matrix with columns = local tip axes expressed in world frame.
        R_tip = [
            [u[0], v[0], n[0]],
            [u[1], v[1], n[1]],
            [u[2], v[2], n[2]],
        ]

        # Two-angle output for direct comparison with 2-axis IMU Euler readout.
        # roll  : rotation around local x-like bending axis
        # pitch : rotation around local y-like bending axis
        try:
            pitch = math.degrees(math.atan2(-R_tip[2][0], math.sqrt(R_tip[0][0] ** 2 + R_tip[1][0] ** 2)))
            roll = math.degrees(math.atan2(R_tip[2][1], R_tip[2][2]))
            if math.isfinite(roll) and math.isfinite(pitch):
                self.tip_euler_deg = [float(roll), float(pitch)]
        except Exception:
            pass

    def _update_hybrid_tip_euler(self):
        """
        Hybrid model:
            SOFA physics model gives tip_euler_deg.
            Dynamic residual module predicts residual = IMU - SOFA.
            Hybrid output = SOFA + predicted residual.

        This does not replace the original SOFA/mechanics model. It is only
        a learned compensation layer on top of the physics-based soft sensor.
        """
        sofa_tip = [float(self.tip_euler_deg[0]), float(self.tip_euler_deg[1])]

        if self.dynamic_model is None:
            self.hybrid_tip_euler_deg = list(sofa_tip)
            self.hybrid_residual_deg = [0.0, 0.0]
            return

        try:
            imu_tip = None
            if getattr(self, "has_imu_feedback", False):
                imu_tip = [float(self.latest_imu_euler_deg[0]), float(self.latest_imu_euler_deg[1])]

            out = self.dynamic_model.predict(
                cmd_mm=self.target_dL_mm,
                actual_mm=self.real_dL_mm,
                sofa_tip_deg=sofa_tip,
                imu_deg=imu_tip,
                update_state=True,
            )

            self.hybrid_tip_euler_deg = [
                float(out["hybrid_tip"][0]),
                float(out["hybrid_tip"][1]),
            ]
            self.hybrid_residual_deg = [
                float(out["predicted_residual"][0]),
                float(out["predicted_residual"][1]),
            ]
        except Exception as e:
            self._log(f"Hybrid residual update failed: {e}", level="warning")
            self.hybrid_tip_euler_deg = list(sofa_tip)
            self.hybrid_residual_deg = [0.0, 0.0]


    def onAnimateBeginEvent(self, e):
        """主循环：基于系统真实时间的平滑插值"""

        current_real_time = time.time()
        if not hasattr(self, 'last_real_time'):
            self.last_real_time = current_real_time
        real_dt = current_real_time - self.last_real_time
        self.last_real_time = current_real_time

        # 可选：接收来自 Qt / UDP 的控制意图
        if self.sock_in is not None and socket is not None:
            try:
                while True:
                    data, _ = self.sock_in.recvfrom(1024)
                    raw = data.decode(errors='ignore').replace(',', ' ')
                    vals = [float(x) for x in raw.split()]
                    if len(vals) >= 3:
                        self.target_dL_mm[0] = self._clamp_mm(vals[0])
                        self.target_dL_mm[1] = self._clamp_mm(vals[1])
                        self.target_dL_mm[2] = self._clamp_mm(vals[2])
                        self._refresh_comp_target()
            except (BlockingIOError, OSError, ValueError):
                pass

        # ROS 2
        if self.ros_node is not None and rclpy.ok():
            rclpy.spin_once(self.ros_node, timeout_sec=0)

        is_moving = False
        for i in range(3):
            diff = self.comp_target_dL_mm[i] - self.dL_mm[i]
            if abs(diff) > 0.05:
                is_moving = True
                step = self.max_speed_mm_s * real_dt
                if abs(diff) <= step:
                    self.dL_mm[i] = self.comp_target_dL_mm[i]
                else:
                    self.dL_mm[i] += step if diff > 0 else -step

        self._accum += real_dt
        should_refresh_state = is_moving or self._accum >= 0.2

        if should_refresh_state:
            if self._accum >= 0.2:
                self._accum = 0.0
            solve_start = time.time()
            self._update_from_solver()
            comp_changed = self._update_compensation_from_error()
            if comp_changed:
                is_moving = True
            solve_cost = time.time() - solve_start

            if self.ser is not None:
                try:
                    s0 = int(round(self.real_dL_mm[0] * self.mm_to_steps))
                    s1 = int(round(self.real_dL_mm[1] * self.mm_to_steps))
                    s2 = int(round(self.real_dL_mm[2] * self.mm_to_steps))
                    self.ser.write(f"{s0},{s1},{s2}\n".encode('utf-8'))
                except Exception as e:
                    self._log(f"Serial send failed: {e}", level="warning")

            if self.sock_out is not None:
                try:
                    msg = f"{self.real_dL_mm[0]:.3f} {self.real_dL_mm[1]:.3f} {self.real_dL_mm[2]:.3f}"
                    self.sock_out.sendto(msg.encode('utf-8'), self.qt_addr)
                except Exception:
                    pass

            self._publish_real_state(force=is_moving)
            self._publish_tip_euler(force=is_moving)
            # Always refresh /robot/tip_position at each solver update so the
            # validation monitor can distinguish "stationary but alive" from "no data".
            self._publish_tip_position(force=True)
            self._publish_hybrid_tip_euler()
            self._publish_hybrid_error()

            if is_moving:
                try:
                    Sofa.msg_info(
                        "Twin",
                        f"Cmd:[{self.target_dL_mm[0]:.1f}, {self.target_dL_mm[1]:.1f}, {self.target_dL_mm[2]:.1f}] | SolverIn:[{self.dL_mm[0]:.1f}, {self.dL_mm[1]:.1f}, {self.dL_mm[2]:.1f}] | RealInternal:[{self.real_dL_mm[0]:.1f}, {self.real_dL_mm[1]:.1f}, {self.real_dL_mm[2]:.1f}] | Bias:[{self.comp_bias_mm[0]:.1f}, {self.comp_bias_mm[1]:.1f}, {self.comp_bias_mm[2]:.1f}] | Solver Cost: {solve_cost*1000:.1f} ms"
                    )
                except Exception:
                    pass

class VisBundle:
    def __init__(self, root, n_disks, ringN=64, disc_r_m=0.025, start_disk=0):
        self.root = root
        self.n_disks = n_disks
        self.ringN = ringN
        self.disc_r_m = disc_r_m
        self.start_disk = max(0, min(int(start_disk), n_disks - 1))
        self.n_vis_disks = n_disks - self.start_disk

        # Backbone
        bb = root.addChild("Backbone")
        self.backbone_mo = bb.addObject("MechanicalObject", name="dofs", template="Vec3d",
                                        position=[[0.0, 0.0, 0.30 * (j / (n_disks - 1) if n_disks > 1 else 0.0)] for j in range(n_disks)])
        bb.addObject("EdgeSetTopologyContainer", edges=[[i, i + 1] for i in range(max(1, n_disks - 1))])
        bb.addObject("EdgeSetTopologyModifier")
        bb_vis = bb.addChild("BackboneVis")
        bb_vis.addObject("OglModel", name="ogl", primitiveType="LINES", position='@../dofs.position',
                         edges='@../EdgeSetTopologyContainer.edges', lineWidth=8, color=[1.0, 0.8, 0.2, 1])
        bb_vis.addObject("IdentityMapping", input='@../dofs', output='@ogl')

        # Disks
        disks = root.addChild("Disks")
        ring_pts, ring_edges = [], []
        for j in range(self.start_disk, n_disks):
            base = (j - self.start_disk) * ringN
            for t in range(ringN):
                ring_pts.append([0, 0, 0])
                if t < ringN - 1: ring_edges.append([base + t, base + t + 1])
        self.disks_mo = disks.addObject("MechanicalObject", name="rings", template="Vec3d", position=ring_pts)
        disks.addObject("EdgeSetTopologyContainer", edges=ring_edges)
        disks.addObject("EdgeSetTopologyModifier")
        disksVis = disks.addChild('Vis')
        disksVis.addObject("OglModel", name="ogl", primitiveType="LINES", position='@../rings.position',
                           edges='@../EdgeSetTopologyContainer.edges', lineWidth=2, color=[0.3, 0.7, 1.0, 0.6])
        disksVis.addObject("IdentityMapping", input='@../rings', output='@ogl')

        # Cables
        self.cable_mo = []
        cable_rgba = [1.0, 0.62, 0.12, 1.0]
        for k in range(3):
            node = root.addChild(f"Cable{k}")
            mo = node.addObject("MechanicalObject", name="dofs", template="Vec3d",
                                position=[[0, 0, 0] for _ in range(self.n_vis_disks)])
            node.addObject("EdgeSetTopologyContainer", edges=[[i, i + 1] for i in range(max(0, self.n_vis_disks - 1))])
            node.addObject("EdgeSetTopologyModifier")
            vis = node.addChild('Vis')
            vis.addObject("OglModel", name="ogl", primitiveType="LINES", position='@../dofs.position',
                          edges='@../EdgeSetTopologyContainer.edges', lineWidth=6, color=cable_rgba)
            vis.addObject("IdentityMapping", input='@../dofs', output='@ogl')
            self.cable_mo.append(mo)

        def _make_axis(node_name, vec, rgba, line_width=4):
            nd = root.addChild(node_name)
            mo = nd.addObject("MechanicalObject", name="dofs", template="Vec3d",
                              position=[[0.0, 0.0, 0.0], [float(vec[0]), float(vec[1]), float(vec[2])]])
            nd.addObject("EdgeSetTopologyContainer", edges=[[0, 1]])
            nd.addObject("EdgeSetTopologyModifier")
            vis = nd.addChild("Vis")
            vis.addObject("OglModel", name="ogl", primitiveType="LINES", position='@../dofs.position',
                          edges='@../EdgeSetTopologyContainer.edges', lineWidth=float(line_width), color=rgba)
            vis.addObject("IdentityMapping", input='@../dofs', output='@ogl')
            return mo

        # Ground Grid & Axes
        try:
            grid = root.addChild("GroundGrid")
            size, step, z0 = 0.25, 0.025, 0.0
            pts, edges, idx = [], [], 0
            y = -size
            while y <= size + 1e-9:
                pts.extend([[-size, y, z0], [size, y, z0]]); edges.append([idx, idx + 1]); idx += 2; y += step
            x = -size
            while x <= size + 1e-9:
                pts.extend([[x, -size, z0], [x, size, z0]]); edges.append([idx, idx + 1]); idx += 2; x += step
            grid.addObject("MechanicalObject", name="pts", template="Vec3d", position=pts)
            grid.addObject("EdgeSetTopologyContainer", edges=edges)
            grid.addObject("EdgeSetTopologyModifier")
            gvis = grid.addChild("Vis")
            gvis.addObject("OglModel", name="ogl", primitiveType="LINES", position='@../pts.position',
                           edges='@../EdgeSetTopologyContainer.edges', lineWidth=1, color=[0.25, 0.28, 0.32, 1])
            gvis.addObject("IdentityMapping", input='@../pts', output='@ogl')
            
            Lw = 0.08
            self.world_axis_mo = [
                _make_axis("AxisX", [Lw, 0, 0], [1.0, 0.2, 0.2, 1], line_width=5),
                _make_axis("AxisY", [0, Lw, 0], [0.2, 1.0, 0.2, 1], line_width=5),
                _make_axis("AxisZ", [0, 0, Lw], [0.2, 0.6, 1.0, 1], line_width=5),
            ]
        except Exception:
            pass

        # Tip visuals
        self.tip_axis_len_m = 0.05
        try:
            tipN = root.addChild("TipPoint")
            self.tip_point_mo = tipN.addObject("MechanicalObject", name="dofs", template="Vec3d", position=[[0.0, 0.0, 0.0]])
            tipV = tipN.addChild("Vis")
            tipV.addObject("OglModel", name="ogl", primitiveType="POINTS", position='@../dofs.position', pointSize=10, color=[1.0, 1.0, 1.0, 1])
            tipV.addObject("IdentityMapping", input='@../dofs', output='@ogl')
            self.tip_axis_mo = [
                _make_axis("TipAxisX", [self.tip_axis_len_m, 0, 0], [1.0, 0.2, 0.2, 1], line_width=6),
                _make_axis("TipAxisY", [0, self.tip_axis_len_m, 0], [0.2, 1.0, 0.2, 1], line_width=6),
                _make_axis("TipAxisZ", [0, 0, self.tip_axis_len_m], [0.2, 0.6, 1.0, 1], line_width=6),
            ]
        except Exception:
            pass


def createScene(root):
    root.dt = 0.01

    _try_required_plugins(root, [
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Visual",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.Setting",
        "Sofa.Component.IO.Mesh",
        "Sofa.GL.Component.Rendering3D",
        "Sofa.GL.Component.Shader",
    ])

    root.addObject("DefaultAnimationLoop")
    root.addObject("DefaultVisualManagerLoop")
    root.addObject("VisualStyle", displayFlags="showVisualModels")
    root.addObject("BackgroundSetting", color=[0.08, 0.1, 0.12, 1])
    root.addObject("InteractiveCamera", position=[0.45, -0.85, 0.35], lookAt=[0, 0, 0.15], fieldOfView=45)

    try:
        root.addObject("LightManager")
        root.addObject("DirectionalLight", name="KeyLight", direction=[0, -1, -1], color=[1, 1, 1, 1])
        root.addObject("DirectionalLight", name="FillLight", direction=[1, 0, -1], color=[0.6, 0.6, 0.7, 1])
        root.addObject("AmbientLight", name="Ambient", color=[0.15, 0.15, 0.18, 1])
    except Exception:
        pass

    BASE_STL_PATH = "/home/ubuntu/桌面/thesis project/base.STL"
    BASE_SCALE_MM_TO_M = 1e-3
    BASE_ROT_DEG = [90.0, 0.0, 0.0]
    BASE_TRANSLATION_M = [-0.08, 0.08, -0.108]

    def _rot_matrix_xyz(rx, ry, rz):
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        Rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
        Ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
        Rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
        def mmul(A, B): return [[A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j] for j in range(3)] for i in range(3)]
        return mmul(Rz, mmul(Ry, Rx))

    def _apply_R_t(p, R, t):
        return [R[0][0]*p[0] + R[0][1]*p[1] + R[0][2]*p[2] + t[0],
                R[1][0]*p[0] + R[1][1]*p[1] + R[1][2]*p[2] + t[1],
                R[2][0]*p[0] + R[2][1]*p[1] + R[2][2]*p[2] + t[2]]

    try:
        base_node = root.addChild("BaseSTL")
        base_loader = None
        try:
            base_loader = base_node.addObject("MeshSTLLoader", name="loader", filename=BASE_STL_PATH)
        except Exception as e:
            print("[WARN] MeshSTLLoader failed:", e)

        if base_loader is not None:
            pts, tris = None, None
            try: pts = [list(map(float, p)) for p in base_loader.position.value]
            except Exception:
                try: pts = [list(map(float, p)) for p in base_loader.position]
                except Exception: pass
            
            try: tris = [list(map(int, t)) for t in base_loader.triangles.value]
            except Exception:
                try: tris = [list(map(int, t)) for t in base_loader.triangles]
                except Exception: pass

            if pts:
                S = float(BASE_SCALE_MM_TO_M)
                pts_m = [[p[0]*S, p[1]*S, p[2]*S] for p in pts]
                rx, ry, rz = [math.radians(float(a)) for a in BASE_ROT_DEG]
                R = _rot_matrix_xyz(rx, ry, rz)
                t = [float(x) for x in BASE_TRANSLATION_M]
                pts_tf = [_apply_R_t(p, R, t) for p in pts_m]

                base_geom = base_node.addChild("Geom")
                base_geom.addObject("MechanicalObject", name="dofs", template="Vec3d", position=pts_tf)

                if tris:
                    base_geom.addObject("TriangleSetTopologyContainer", triangles=tris)
                    base_geom.addObject("TriangleSetTopologyModifier")
                    base_vis = base_geom.addChild("Vis")
                    base_vis.addObject("OglModel", name="ogl", primitiveType="TRIANGLES", position='@../dofs.position',
                                       triangles='@../TriangleSetTopologyContainer.triangles', color=[0.45, 0.46, 0.48, 1.0])
                    base_vis.addObject("IdentityMapping", input='@../dofs', output='@ogl')
                else:
                    base_node.addObject("MeshTopology", src="@loader")
                    base_vis = base_node.addChild("Vis")
                    base_vis.addObject("OglModel", name="ogl", src="@../loader", color=[0.45, 0.46, 0.48, 1.0])
    except Exception as e:
        print("[WARN] BaseSTL load failed:", e)

    n_disks = 6
    vis = VisBundle(root, n_disks=n_disks, ringN=64, disc_r_m=0.025, start_disk=0)

    # 现在的参数更清爽，只传必要的东西
    ctrl = SolverDrivenController(
        root, vis,
        name="TwinCtrl",
        n_disks=n_disks,
        length_m=0.30,
        hole_r_m=0.020,
        helix_total_rad=6.0 * math.pi,
        mu_friction=0.8,
        enforce_no_twist=True,
        update_every_s=0.05
    )

    try:
        root.addObject(ctrl, listening=True)
    except Exception:
        ctrl.listening = True
        root.addObject(ctrl)

    return root
