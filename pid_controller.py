#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from std_msgs.msg import Empty, Bool
from geometry_msgs.msg import Pose, Twist

#from pidrone_pkg.msg import Mode, RC, State
from pidrone.msg import Mode,RC, State
import signal
import sys
import os
import argparse
import numpy as np
from pid_class import PID, PIDaxis
from three_dim_vec import Position, Velocity, Error, RPY
import command_values as cmds


class PIDController(Node):
    ''' Controls the flight of the drone by running a PID controller on the
    error calculated by the desired and current velocity and position of the drone
    '''

    def __init__(self):
        super().__init__('pid_controller')

        # Initialize the current and desired modes
        self.current_mode = Mode()
        self.desired_mode = Mode()

        # Initialize in velocity control
        self.position_control = False
        self.last_position_control = False

        # Initialize the current and desired positions
        self.current_position = Position()
        self.desired_position = Position(z=0.1)
        self.last_desired_position = Position(z=0.1)

        # Initialize the position error
        self.position_error = Error()

        # Initialize the current and desired velocities
        self.current_velocity = Velocity()
        self.desired_velocity = Velocity()

        # Initialize the velocity error
        self.velocity_error = Error()

        # Set the distance that a velocity command will move the drone (m)
        self.desired_velocity_travel_distance = 0.1
        # Set a static duration that a velocity command will be held
        self.desired_velocity_travel_time = 0.1

        # Set a static duration that a yaw velocity command will be held
        self.desired_yaw_velocity_travel_time = 0.25

        # Store the start time of the desired velocities
        self.desired_velocity_start_time = None
        self.desired_yaw_velocity_start_time = None

        # Initialize the primary PID
        self.pid = PID()

        # Initialize the error used for the PID which is vx, vy, z where vx and
        # vy are velocities, and z is the error in the altitude of the drone
        self.pid_error = Error()

        # Initialize the 'position error to velocity error' PIDs:
        # left/right (roll) pid
        self.lr_pid = PIDaxis(kp=20.0, ki=5.0, kd=10.0, midpoint=0, control_range=(-10.0, 10.0))
        # front/back (pitch) pid
        self.fb_pid = PIDaxis(kp=20.0, ki=5.0, kd=10.0, midpoint=0, control_range=(-10.0, 10.0))

        # Initialize the pose callback time
        self.last_pose_time = None

        # Initialize the desired yaw velocity
        self.desired_yaw_velocity = 0

        # Initialize the current and  previous roll, pitch, yaw values
        self.current_rpy = RPY()
        self.previous_rpy = RPY()

        # initialize the current and previous states
        self.current_state = State()
        self.previous_state = State()

        # a variable used to determine if the drone is moving between desired
        # positions
        self.moving = False

        # a variable that determines the maximum magnitude of the position error
        # Any greater position error will overide the drone into velocity
        # control
        self.safety_threshold = 1.5

        # determines if the position of the drone is known
        self.lost = False

        # determines if the desired poses are absolute or relative to the drone
        self.absolute_desired_position = False

        # determines whether to use open loop velocity path planning which is
        # accomplished by calculate_travel_time
        self.path_planning = True

        # Publishers
        self.cmdpub = self.create_publisher(RC, '/pidrone/fly_commands', 10)
        self.position_control_pub = self.create_publisher(Bool, '/pidrone/position_control', 10)
        self.heartbeat_pub = self.create_publisher(Empty, '/pidrone/heartbeat/pid_controller', 10)

        # Subscribers
        self.create_subscription(State, '/pidrone/state', self.current_state_callback, 10)
        self.create_subscription(Pose, '/pidrone/desired/pose', self.desired_pose_callback, 10)
        self.create_subscription(Twist, '/pidrone/desired/twist', self.desired_twist_callback, 10)
        self.create_subscription(Mode, '/pidrone/mode', self.current_mode_callback, 10)
        self.create_subscription(Mode, '/pidrone/desired/mode', self.desired_mode_callback, 10)
        self.create_subscription(Bool, '/pidrone/position_control', self.position_control_callback, 10)
        self.create_subscription(Empty, '/pidrone/reset_transform', self.reset_callback, 10)
        self.create_subscription(Bool, '/pidrone/picamera/lost', self.lost_callback, 10)

    # ROS SUBSCRIBER CALLBACK METHODS
    #################################
    def current_state_callback(self, state):
        """ Store the drone's current state for calculations """
        self.previous_state = self.current_state
        self.current_state = state
        self.state_to_three_dim_vec_structs()

    def desired_pose_callback(self, msg):
        """ Update the desired pose """
        # store the previous desired position
        self.last_desired_position = self.desired_position
        # set the desired positions equal to the desired pose message
        if self.absolute_desired_position:
            self.desired_position.x = msg.position.x
            self.desired_position.y = msg.position.y
            # the desired z must be above z and below the range of the ir sensor (.55meters)
            self.desired_position.z = msg.position.z if 0 <= self.desired_position.z <= 0.5 else self.last_desired_position.z
        else:
            self.desired_position.x = self.current_position.x + msg.position.x
            self.desired_position.y = self.current_position.y + msg.position.y
            # set the desired z position relative to the last desired position
            desired_z = self.last_desired_position.z + msg.position.z
            self.desired_position.z = desired_z if 0 <= desired_z <= 0.5 else self.last_desired_position.z

        if self.desired_position != self.last_desired_position:
            # the drone is moving between desired positions
            self.moving = True
            self.get_logger().info('Moving')

    def desired_twist_callback(self, msg):
        """ Update the desired twist """
        self.desired_velocity.x = msg.linear.x
        self.desired_velocity.y = msg.linear.y
        self.desired_velocity.z = msg.linear.z
        self.desired_yaw_velocity = msg.angular.z
        self.desired_velocity_start_time = None
        self.desired_yaw_velocity_start_time = None
        if self.path_planning:
            self.calculate_travel_time()

    def current_mode_callback(self, msg):
        """ Update the current mode """
        self.current_mode = msg.mode

    def desired_mode_callback(self, msg):
        """ Update the desired mode """
        self.desired_mode = msg.mode

    def position_control_callback(self, msg):
        """ Set whether or not position control is enabled """
        self.position_control = msg.data
        if self.position_control:
            self.desired_position = self.current_position
        if self.position_control != self.last_position_control:
            self.get_logger().info(f"Position Control: {self.position_control}")
            self.last_position_control = self.position_control

    def reset_callback(self, empty):
        """ Reset the desired and current poses of the drone and set desired velocities to zero """
        self.current_position = Position(z=self.current_position.z)
        self.desired_position = self.current_position
        self.desired_velocity.x = 0
        self.desired_velocity.y = 0

    def lost_callback(self, msg):
        self.lost = msg.data

    # Step Method
    #############
    def step(self):
        """ Returns the commands generated by the pid """
        self.calc_error()
        if self.position_control:
            if self.position_error.planar_magnitude() < self.safety_threshold and not self.lost:
                if self.moving:
                    if self.position_error.magnitude() > 0.05:
                        self.pid_error -= self.velocity_error * 100
                    else:
                        self.moving = False
                        self.get_logger().info('Not moving')
            else:
                self.position_control_pub.publish(False)

        if self.desired_velocity.magnitude() > 0 or abs(self.desired_yaw_velocity) > 0:
            self.adjust_desired_velocity()

        return self.pid.step(self.pid_error, self.desired_yaw_velocity)

    # Helper methods
    ################
    def state_to_three_dim_vec_structs(self):
        """
        Convert the values from the state estimator into ThreeDimVec structs to
        make calculations concise
        """
        # store the positions
        pose = self.current_state.pose_with_covariance.pose
        self.current_position.x = pose.position.x
        self.current_position.y = pose.position.y
        self.current_position.z = pose.position.z

        # store the linear velocities
        twist = self.current_state.twist_with_covariance.twist
        self.current_velocity.x = twist.linear.x
        self.current_velocity.y = twist.linear.y
        self.current_velocity.z = twist.linear.z

        # store the orientations
        self.previous_rpy = self.current_rpy
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        r, p, y = tf.transformations.euler_from_quaternion(quaternion)
        self.current_rpy = RPY(r, p, y)

    def adjust_desired_velocity(self):
        """ Set the desired velocity back to 0 once the drone has traveled the
        amount of time that causes it to move the specified desired velocity
        travel distance """
        
        clock = Clock()
        curr_time = clock.now()

        # set the desired planar velocities to zero if the duration is up
        if self.desired_velocity_start_time is not None:
            duration = curr_time - self.desired_velocity_start_time
            if duration > self.desired_velocity_travel_time:
                self.desired_velocity.x = 0
                self.desired_velocity.y = 0
                self.desired_velocity_start_time = None
        else:
            self.desired_velocity_start_time = curr_time

        # set the desired yaw velocity to zero if the duration is up
        if self.desired_yaw_velocity_start_time != None:
            duration = curr_time - self.desired_yaw_velocity_start_time
            if duration > self.desired_yaw_velocity_travel_time:
                self.desired_yaw_velocity = 0
                self.desired_yaw_velocity_start_time = None
        else:
            self.desired_yaw_velocity_start_time = curr_time

    def calc_error(self):
        """ Calculate the error in velocity, and if in position hold, add the
        error from lr_pid and fb_pid to the velocity error to control the
        position of the drone """
        
        clock = Clock()
        get_time = clock.now()
        
        pose_dt = 0
        if self.last_pose_time is not None:
            pose_dt = get_time - self.last_pose_time
        self.last_pose_time = get_time

        # calculate the velocity error
        self.velocity_error = self.desired_velocity - self.current_velocity
        dz = self.desired_position.z - self.current_position.z
        self.pid_error.x = self.velocity_error.x
        self.pid_error.y = self.velocity_error.y
        self.pid_error.z = dz
        self.pid_error = self.pid_error * 100
        
        if self.position_control:
            self.position_error = self.desired_position - self.current_position
            lr_step = self.lr_pid.step(self.position_error.x, pose_dt)
            fb_step = self.fb_pid.step(self.position_error.y, pose_dt)
            self.pid_error.x += lr_step
            self.pid_error.y += fb_step

    def calculate_travel_time(self):
        """ Calculate travel time for desired velocity """
        if self.desired_velocity.magnitude() > 0:
            travel_time = self.desired_velocity_travel_distance / self.desired_velocity.planar_magnitude()
        else:
            travel_time = 0.0
        self.desired_velocity_travel_time = travel_time

    def reset(self):
        """ Reset desired_position to current position """
        self.position_error = Error(0, 0, 0)
        self.desired_position = Position(self.current_position.x, self.current_position.y, 0.05)
        self.velocity_error = Error(0, 0, 0)
        self.desired_velocity = Velocity(0, 0, 0)
        self.pid.reset()
        self.lr_pid.reset()
        self.fb_pid.reset()

    def publish_cmd(self, cmd):
        """ Publish the controls to /pidrone/fly_commands """
        msg = RC()
        msg.roll = cmd[0]
        msg.pitch = cmd[1]
        msg.yaw = cmd[2]
        msg.throttle = cmd[3]
        self.cmdpub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    pid_controller = PIDController()

    signal.signal(signal.SIGINT, lambda signal, frame: sys.exit(0))

    try:
        rclpy.spin(pid_controller)
    except KeyboardInterrupt:
        pass
    finally:
        pid_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
