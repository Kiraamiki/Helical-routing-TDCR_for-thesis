#!/usr/bin/env python3
import sys
import math
import argparse
import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class Mega2560RosBridge(Node):
    def __init__(
        self,
        port='/dev/ttyACM0',
        baudrate=115200,
        state_topic='/robot/tendon_state',
        imu_topic='/robot/imu_euler',
        send_hz=30.0,
        min_step_delta=20,
    ):
        super().__init__('mega2560_ros_bridge')

        self.state_topic = state_topic
        self.imu_topic = imu_topic

        # ===== 机械/驱动参数（按你当前实际配置）=====
        # 电机0: 4细分  -> 800 pulse/rev
        # 电机1: 8细分  -> 1600 pulse/rev
        # 电机2: 8细分  -> 1600 pulse/rev
        self.pulses_per_motor_rev = [800.0, 1600.0, 1600.0]

        # PG51 减速箱，卷线轮在输出轴
        self.gear_ratio = 51.0

        # 卷线轮直径 40 mm
        self.spool_diameter_mm = 40.0
        self.spool_circumference_mm = self.spool_diameter_mm * math.pi

        # 每个轴单独的 steps_per_mm
        # 输出轴一圈所需脉冲 = 电机一圈脉冲 * 减速比
        self.steps_per_mm = [
            (ppr * self.gear_ratio) / self.spool_circumference_mm
            for ppr in self.pulses_per_motor_rev
        ]

        self.get_logger().info(
            f'steps_per_mm = {[round(v, 6) for v in self.steps_per_mm]}'
        )

        self.latest_mm_vals = [0.0, 0.0, 0.0]
        self.last_sent_steps = None
        self.last_imu_vals = None
        self.min_step_delta = int(min_step_delta)

        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.02)
            self.get_logger().info(f'Connected to Mega2560: {port} @ {baudrate}')
        except Exception as e:
            self.get_logger().error(f'Cannot open serial port {port}: {e}')
            sys.exit(1)

        self.state_sub = self.create_subscription(
            Float32MultiArray,
            self.state_topic,
            self._on_state,
            10,
        )

        self.imu_pub = self.create_publisher(
            Float32MultiArray,
            self.imu_topic,
            10,
        )

        self.poll_timer = self.create_timer(0.01, self._poll_serial)
        self.send_timer = self.create_timer(1.0 / float(send_hz), self._send_latest_state)

        self.get_logger().info(f'Listening tendon state: {self.state_topic}')
        self.get_logger().info(f'Publishing IMU euler: {self.imu_topic}')
        self.get_logger().info(
            f'Merged bridge running. send_hz={send_hz}, min_step_delta={self.min_step_delta}'
        )

    def mm_to_driver_steps(self, mm_vals):
        # 分轴换算：每个电机用自己的 steps_per_mm
        return [
            int(round(-float(mm_vals[i]) * self.steps_per_mm[i]))
            for i in range(3)
        ]

    def _on_state(self, msg):
        if len(msg.data) < 3:
            return
        self.latest_mm_vals = [float(msg.data[0]), float(msg.data[1]), float(msg.data[2])]

    def _send_latest_state(self):
        steps = self.mm_to_driver_steps(self.latest_mm_vals)

        if self.last_sent_steps is not None:
            if all(abs(steps[i] - self.last_sent_steps[i]) < self.min_step_delta for i in range(3)):
                return

        cmd_str = f"{steps[0]},{steps[1]},{steps[2]}\n"

        try:
            self.ser.write(cmd_str.encode('utf-8'))
            self.last_sent_steps = list(steps)
        except Exception as e:
            self.get_logger().error(f'Serial send failed: {e}')

    def _poll_serial(self):
        try:
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                if not line.startswith('IMU,'):
                    continue

                parts = line.split(',')
                if len(parts) < 3:
                    continue

                try:
                    a = float(parts[1])
                    b = float(parts[2])
                except ValueError:
                    continue

                vals = (round(a, 4), round(b, 4))
                if vals == self.last_imu_vals:
                    continue

                msg = Float32MultiArray()
                msg.data = [a, b]
                self.imu_pub.publish(msg)
                self.last_imu_vals = vals

        except Exception as e:
            self.get_logger().warning(f'Serial read failed: {e}')

    def destroy_node(self):
        try:
            if hasattr(self, 'ser') and self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/ttyACM0')
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--state-topic', default='/robot/tendon_state')
    parser.add_argument('--imu-topic', default='/robot/imu_euler')
    parser.add_argument('--send-hz', type=float, default=30.0)
    parser.add_argument('--min-step-delta', type=int, default=20)
    parsed, ros_args = parser.parse_known_args(args=sys.argv[1:])

    rclpy.init(args=ros_args)
    node = Mega2560RosBridge(
        port=parsed.port,
        baudrate=parsed.baudrate,
        state_topic=parsed.state_topic,
        imu_topic=parsed.imu_topic,
        send_hz=parsed.send_hz,
        min_step_delta=parsed.min_step_delta,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()