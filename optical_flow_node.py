#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from raspicam_node.msg import MotionVectors
import numpy as np
from sensor_msgs.msg import Range
import rclpy.logging


class OpticalFlowNode(Node):
    """
    Subscribe to the optical flow vectors and publish linear velocity as a Twist message.
    """
    def __init__(self, node_name):
        super().__init__(node_name)
        
        # flow variables
        camera_wh = (320, 240)        
        self.max_flow = camera_wh[0] / 16.0 * camera_wh[1] / 16.0 * 2**7
        self.flow_scale = .165
        self.flow_coeff = 100 * self.flow_scale / self.max_flow  # (multiply by 100 for cm to m conversion)
        self.altitude = 0.03  # initialize to a bit off the ground
        self.altitude_ts = rclpy.time.Time()
        self.setup()

    def setup(self):
        # publisher
        self.twistpub = self.create_publisher(TwistStamped, '/pidrone/picamera/twist', 1)

        # subscribers
        self._sub_mv = self.create_subscription(MotionVectors, '/raspicam_node/motion_vectors', self.motion_cb, 1)
        self._sub_alt = self.create_subscription(Range, '/pidrone/range', self.altitude_cb, 1)

    def motion_cb(self, msg):
        ''' Average the motion vectors and publish the
        twist message. 
        '''
        # signed 1-byte values
        x = msg.x
        y = msg.y

        # calculate the planar and yaw motions
        x_motion = np.sum(x) * self.flow_coeff * self.altitude
        y_motion = np.sum(y) * self.flow_coeff * self.altitude
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.twist.linear.x = x_motion
        twist_msg.twist.linear.y = -y_motion
        # Update and publish the twist message
        self.twistpub.publish(twist_msg)
        
        duration_from_last_altitude = self.get_clock().now() - self.altitude_ts
        if duration_from_last_altitude.seconds > 10:
            self.get_logger().warn("No altitude received for {:10.4f} seconds.".format(duration_from_last_altitude.seconds))

    def altitude_cb(self, msg):
        """
        The altitude of the robot
        Args:
            msg:  the message publishing the altitude
        """
        self.altitude = msg.range
        self.altitude_ts = msg.header.stamp


def main():
    rclpy.init()
    optical_flow_node = OpticalFlowNode("optical_flow_node")
    rclpy.spin(optical_flow_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
