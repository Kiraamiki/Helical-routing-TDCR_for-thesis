#!/usr/bin/env python3
import json
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

MODEL_PATH = "dynamic_residual_model.json"


class DynamicResidualCorrector(Node):
    def __init__(self):
        super().__init__("dynamic_residual_corrector")

        with open(MODEL_PATH, "r") as f:
            self.model = json.load(f)

        self.features = self.model["features"]
        self.x_mean = np.array(self.model["x_mean"], dtype=float)
        self.x_std = np.array(self.model["x_std"], dtype=float)
        self.W = np.array(self.model["W"], dtype=float)

        self.cmd = [0.0, 0.0, 0.0]
        self.actual = [0.0, 0.0, 0.0]
        self.tip = [0.0, 0.0]
        self.imu = [0.0, 0.0]

        self.prev_t = None
        self.prev_cmd = [0.0, 0.0, 0.0]
        self.prev_actual = [0.0, 0.0, 0.0]
        self.prev_tip = [0.0, 0.0]
        self.prev_imu = [0.0, 0.0]
        self.prev_err = [0.0, 0.0]

        self.create_subscription(Float32MultiArray, "/robot/tendon_commands", self.on_cmd, 10)
        self.create_subscription(Float32MultiArray, "/robot/tendon_state", self.on_actual, 10)
        self.create_subscription(Float32MultiArray, "/robot/tip_euler", self.on_tip, 10)
        self.create_subscription(Float32MultiArray, "/robot/imu_euler", self.on_imu, 10)

        self.pub = self.create_publisher(Float32MultiArray, "/robot/tip_euler_corrected", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/robot/tip_euler_corrected_error", 10)
        self.timer = self.create_timer(0.03, self.tick)

        self.get_logger().info("Dynamic residual corrector running.")
        self.get_logger().info("Publishing corrected tip Euler to /robot/tip_euler_corrected")

    def on_cmd(self, msg):
        if len(msg.data) >= 3:
            self.cmd = [float(msg.data[0]), float(msg.data[1]), float(msg.data[2])]

    def on_actual(self, msg):
        if len(msg.data) >= 3:
            self.actual = [float(msg.data[0]), float(msg.data[1]), float(msg.data[2])]

    def on_tip(self, msg):
        if len(msg.data) >= 2:
            self.tip = [float(msg.data[0]), float(msg.data[1])]

    def on_imu(self, msg):
        if len(msg.data) >= 2:
            self.imu = [float(msg.data[0]), float(msg.data[1])]

    def make_feature_vector(self):
        now = time.time()
        if self.prev_t is None:
            dt = 1e-3
        else:
            dt = max(1e-3, now - self.prev_t)

        dcmd_dt = [(self.cmd[i] - self.prev_cmd[i]) / dt for i in range(3)]
        dactual_dt = [(self.actual[i] - self.prev_actual[i]) / dt for i in range(3)]

        sofa_roll, sofa_pitch = self.tip
        imu_roll, imu_pitch = self.imu
        current_err = [imu_roll - sofa_roll, imu_pitch - sofa_pitch]

        values = {
            "cmd0": self.cmd[0],
            "cmd1": self.cmd[1],
            "cmd2": self.cmd[2],
            "actual0": self.actual[0],
            "actual1": self.actual[1],
            "actual2": self.actual[2],
            "cmd0_prev": self.prev_cmd[0],
            "cmd1_prev": self.prev_cmd[1],
            "cmd2_prev": self.prev_cmd[2],
            "actual0_prev": self.prev_actual[0],
            "actual1_prev": self.prev_actual[1],
            "actual2_prev": self.prev_actual[2],
            "dcmd0_dt": dcmd_dt[0],
            "dcmd1_dt": dcmd_dt[1],
            "dcmd2_dt": dcmd_dt[2],
            "dactual0_dt": dactual_dt[0],
            "dactual1_dt": dactual_dt[1],
            "dactual2_dt": dactual_dt[2],
            "sofa_roll": sofa_roll,
            "sofa_pitch": sofa_pitch,
            "sofa_roll_prev": self.prev_tip[0],
            "sofa_pitch_prev": self.prev_tip[1],
            "err_roll_prev": self.prev_err[0],
            "err_pitch_prev": self.prev_err[1],
        }

        x = np.array([values[name] for name in self.features], dtype=float)

        self.prev_t = now
        self.prev_cmd = list(self.cmd)
        self.prev_actual = list(self.actual)
        self.prev_tip = list(self.tip)
        self.prev_imu = list(self.imu)
        self.prev_err = current_err

        return x

    def tick(self):
        x = self.make_feature_vector()
        x_n = (x - self.x_mean) / self.x_std
        x_aug = np.concatenate([x_n, [1.0]])
        pred_residual = x_aug @ self.W

        corrected_roll = self.tip[0] + float(pred_residual[0])
        corrected_pitch = self.tip[1] + float(pred_residual[1])

        msg = Float32MultiArray()
        msg.data = [corrected_roll, corrected_pitch]
        self.pub.publish(msg)

        err_msg = Float32MultiArray()
        err_msg.data = [corrected_roll - self.imu[0], corrected_pitch - self.imu[1]]
        self.err_pub.publish(err_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicResidualCorrector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
