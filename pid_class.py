#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Point  # or any other type for error.x, error.y, error.z
from time import time as sys_time


class PIDaxis():
    def __init__(self, kp, ki, kd, i_range=None, d_range=None, control_range=(1000, 2000), midpoint=1500, smoothing=True):
        # Tuning
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # Config
        self.i_range = i_range
        self.d_range = d_range
        self.control_range = control_range
        self.midpoint = midpoint
        self.smoothing = smoothing
        # Internal
        self.reset()

    def reset(self):
        self._old_err = None
        self._p = 0
        self._i = 0
        self._d = 0
        self._dd = 0
        self._ddd = 0

    def step(self, err, time_elapsed):
        if self._old_err is None:
            # First time around prevent d term spike
            self._old_err = err

        # Find the p component
        self._p = err * self.kp

        # Find the i component
        self._i += err * self.ki * time_elapsed
        if self.i_range is not None:
            self._i = max(self.i_range[0], min(self._i, self.i_range[1]))

        # Find the d component
        self._d = (err - self._old_err) * self.kd / time_elapsed
        if self.d_range is not None:
            self._d = max(self.d_range[0], min(self._d, self.d_range[1]))
        self._old_err = err

        # Smooth over the last three d terms
        if self.smoothing:
            self._d = (self._d * 8.0 + self._dd * 5.0 + self._ddd * 2.0)/15.0
            self._ddd = self._dd
            self._dd = self._d

        # Calculate control output
        raw_output = self._p + self._i + self._d
        output = min(max(raw_output + self.midpoint, self.control_range[0]), self.control_range[1])

        return output


class PID:

    def __init__(self,

                 roll=PIDaxis(2.0, 1.0, 0.0, control_range=(1400, 1600), midpoint=1500, i_range=(-100, 100)),
                 roll_low=PIDaxis(0.0, 0.5, 0.0, control_range=(1400, 1600), midpoint=1500, i_range=(-150, 150)),

                 pitch=PIDaxis(2.0, 1.0, 0.0, control_range=(1400, 1600), midpoint=1500, i_range=(-100, 100)),
                 pitch_low=PIDaxis(0.0, 0.5, 0.0, control_range=(1400, 1600), midpoint=1500, i_range=(-150, 150)),

                 yaw=PIDaxis(0.0, 0.0, 0.0),

                 # Kv 2300 motors have midpoint 1300, Kv 2550 motors have midpoint 1250
                 throttle=PIDaxis(1,
                                  0, #0.5/height_factor * battery_factor,
                                  1,
                                  i_range=(-400, 400), control_range=(1200, 1375),
                                  d_range=(-40, 40), midpoint=1350)
                 ):

        self.trim_controller_cap_plane = 0.05
        self.trim_controller_thresh_plane = 0.0001

        self.roll = roll
        self.roll_low = roll_low

        self.pitch = pitch
        self.pitch_low = pitch_low

        self.yaw = yaw

        self.trim_controller_cap_throttle = 5.0
        self.trim_controller_thresh_throttle = 5.0

        self.throttle = throttle

        self._t = None

        # Tuning values specific to each drone
        self.roll_low.init_i = 0.0
        self.pitch_low.init_i = 0.0
        self.reset()

    def reset(self):
        """ Reset each pid and restore the initial i terms """
        # reset time variable
        self._t = None

        # reset individual PIDs
        self.roll.reset()
        self.roll_low.reset()
        self.pitch.reset()
        self.pitch_low.reset()
        self.throttle.reset()

        # restore tuning values
        self.roll_low._i = self.roll_low.init_i
        self.pitch_low._i = self.pitch_low.init_i

    def step(self, error, cmd_yaw_velocity=0):
        """ Compute the control variables from the error using the step methods
        of each axis pid.
        """
        # First time around prevent time spike
        if self._t is None:
            time_elapsed = 1
        else:
            time_elapsed = rclpy.clock.Clock().now().seconds_nanoseconds()[0] - self._t

        self._t = rclpy.clock.Clock().now().seconds_nanoseconds()[0]

        # Compute roll command
        ######################
        # if the x velocity error is within the threshold
        if abs(error.x) < self.trim_controller_thresh_plane:
            # pass the high rate i term off to the low rate pid
            self.roll_low._i += self.roll._i
            self.roll._i = 0
            # set the roll value to just the output of the low rate pid
            cmd_r = self.roll_low.step(error.x, time_elapsed)
        else:
            if error.x > self.trim_controller_cap_plane:
                self.roll_low.step(self.trim_controller_cap_plane, time_elapsed)
            elif error.x < -self.trim_controller_cap_plane:
                self.roll_low.step(-self.trim_controller_cap_plane, time_elapsed)
            else:
                self.roll_low.step(error.x, time_elapsed)

            cmd_r = self.roll_low._i + self.roll.step(error.x, time_elapsed)

        # Compute pitch command
        #######################
        if abs(error.y) < self.trim_controller_thresh_plane:
            self.pitch_low._i += self.pitch._i
            self.pitch._i = 0
            cmd_p = self.pitch_low.step(error.y, time_elapsed)
        else:
            if error.y > self.trim_controller_cap_plane:
                self.pitch_low.step(self.trim_controller_cap_plane, time_elapsed)
            elif error.y < -self.trim_controller_cap_plane:
                self.pitch_low.step(-self.trim_controller_cap_plane, time_elapsed)
            else:
                self.pitch_low.step(error.y, time_elapsed)

            cmd_p = self.pitch_low._i + self.pitch.step(error.y, time_elapsed)

        # Compute yaw command
        cmd_y = 1500 + cmd_yaw_velocity

        cmd_t = self.throttle.step(error.z, time_elapsed)
        
        #cmd_t = 1250

        print("%d, %.3f, %.3f, %.3f, %.3f" % (cmd_t, error.z, self.throttle._p, self.throttle._i, self.throttle._d))
        # Print statements for the low and high i components
        # print "Roll  low, hi:", self.roll_low._i, self.roll._i
        # print "Pitch low, hi:", self.pitch_low._i, self.pitch._i
        # print "Throttle low, hi:", self.throttle_low._i, self.throttle._i
        #print cmd_t
        return [cmd_r, cmd_p, cmd_y, cmd_t]

def main(args=None):
    rclpy.init(args=args)

    # Node creation
    node = Node("pid_controller")

    # Example usage
    pid = PID()

    # Simulated error and time (for example)
    error = Point(x=0.1, y=0.2, z=0.3)  # This should be replaced by actual data
    cmd_yaw_velocity = 0  # This should be the desired yaw velocity

    # Step the PID controller
    pid_output = pid.step(error, cmd_yaw_velocity)
    print(pid_output)

    # Shutdown the node
    rclpy.shutdown()

if __name__ == '__main__':
    main()
