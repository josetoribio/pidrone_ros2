#!/usr/bin/env python3

import rclpy
import argparse
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Header
from dt_vl53l0x import VL53L0X

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--variance", type=float, default=None,
                    help="how much random noise should be added to the distance readings? variance, meters")
args = parser.parse_args()
simulated_noise_sd = np.sqrt(args.variance) if args.variance else None


class ToFNode(Node):
    def __init__(self):
        super().__init__('tof_node')
        
        self._i2c_address = 0x29
        self._sensor_name = "tof"
        
        # Create VL53L0X sensor handler
        self._sensor = VL53L0X()
        self._sensor.open()
        self._sensor.start_ranging()
        
        # Create publisher
        self._pub = self.create_publisher(Range, '/pidrone/range', 10)
        
        # Create timer
        self.timer = self.create_timer(1.0 / 30, self._timer_cb)
        
        self.get_logger().info("ToF Node started")

    def _timer_cb(self):
        distance_mm = self._sensor.get_distance()
        
        # Add optional noise
        if simulated_noise_sd is not None:
            distance_mm += 1000 * np.random.normal(loc=0, scale=simulated_noise_sd)
        
        # Create range message
        msg = Range()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "tof"
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = 10
        msg.min_range = 0.05
        msg.max_range = 1.2
        msg.range = distance_mm / 1000.0
        
        # Publish message
        self._pub.publish(msg)

    def on_shutdown(self):
        try:
            self._sensor.stop_ranging()
        except Exception as e:
            self.get_logger().error(f"Error stopping sensor: {e}")
        self.get_logger().info("Shutting down ToF Node")


def main(args=None):
    rclpy.init(args=args)
    node = ToFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
