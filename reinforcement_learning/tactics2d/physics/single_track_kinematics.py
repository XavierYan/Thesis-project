##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: single_track_kinematics.py
# @Description: This file implements a kinematic single-track model for a traffic participant.
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import Tuple, Union

import numpy as np

from tactics2d.participant.trajectory import State

from .physics_model_base import PhysicsModelBase


class SingleTrackKinematics(PhysicsModelBase):
    r"""This class implements a kinematic single-track bicycle model for a traffic participant.

    The is a simplified model to simulate the traffic participant's physics. The assumptions in this implementation include:

    1. The traffic participant is operating in a 2D plane (x-y).
    2. The left and right wheels always have the same steering angle and speed, so they can be regarded as a single wheel.
    3. The traffic participant is a rigid body, so its geometry does not change during the simulation.
    4. The traffic participant is Front-Wheel Drive (FWD).

    This implementation version is based on the following paper. It regard the geometry center as the reference point.

    ![Kinematic Single Track Model](https://cdn.jsdelivr.net/gh/MotacillaAlba/image-storage@main/img/kinematic_bicycle_model.png)

    ![Demo of the implementation (interval=100 ms, $\Delta t$=5 ms)](https://cdn.jsdelivr.net/gh/MotacillaAlba/image-storage@main/img/tactics2d-single_track_kinematics.gif)

    !!! quote "Reference"
        Kong, Jason, et al. "Kinematic and dynamic vehicle models for autonomous driving control design." *2015 IEEE intelligent vehicles symposium* (IV). IEEE, 2015.

    !!! warning
        This model will lose its accuracy when the time step is set too large or the traffic participant is made to travel at a high speed.

    Attributes:
        lf (float): The distance from the geometry center to the front axle center. The unit is meter.
        lr (float): The distance from the geometry center to the rear axle center. The unit is meter.
        steer_rate_range (Union[float, Tuple[float, float]], optional): The steering angle range. The valid input is a float or a tuple of two floats represents (min steering angle, max steering angle). The unit is radian.

            - When the steer_rate_range is a non-negative float, the steering angle is constrained to be within the range [-steer_rate_range, steer_rate_range].
            - When the steer_rate_range is a tuple, the steering angle is constrained to be within the range [min steering angle, max steering angle].
            - When the steer_rate_range is negative or the min steering angle is not less than the max steering angle, the steer_rate_range is set to None.

        speed_range (Union[float, Tuple[float, float]], optional): The speed range. The valid input is a float or a tuple of two floats represents (min speed, max speed). The unit is meter per second (m/s).
            - When the speed_range is a non-negative float, the speed is constrained to be within the range [-speed_range, speed_range].
            - When the speed_range is a tuple, the speed is constrained to be within the range [min speed, max speed].
            - When the speed_range is negative or the min speed is not less than the max speed, the speed_range is set to None.

        accel_range (Union[float, Tuple[float, float]], optional): The acceleration range. The valid input is a float or a tuple of two floats represents (min acceleration, max acceleration). The unit is meter per second squared (m/s$^2$).

            - When the accel_range is a non-negative float, the acceleration is constrained to be within the range [-accel_range, accel_range].
            - When the accel_range is a tuple, the acceleration is constrained to be within the range [min acceleration, max acceleration].
            - When the accel_range is negative or the min acceleration is not less than the max acceleration, the accel_range is set to None.

        interval (int, optional): The time interval between the current state and the new state. The unit is millisecond. Defaults to None.
        delta_t (int, optional): The time step for the simulation. The unit is millisecond. Defaults to `_DELTA_T`(5 ms). The expected value is between `_MIN_DELTA_T`(1 ms) and `interval`. It is recommended to keep delta_t smaller than 5 ms.
    """

    def __init__(
        self,
        lf: float,
        lr: float,
        steer_rate_range: Union[float, Tuple[float, float]] = None,
        speed_range: Union[float, Tuple[float, float]] = None,
        accel_range: Union[float, Tuple[float, float]] = None,
        delta_t: int = 1/25,
    ):
        """Initialize the kinematic single-track model.

        Args:
            lf (float): The distance from the center of mass to the front axle center. The unit is meter.
            lr (float): The distance from the center of mass to the rear axle center. The unit is meter.
            steer_rate_range (Union[float, Tuple[float, float]], optional): The range of steering angle. The valid input is a positive float or a tuple of two floats represents (min steering angle, max steering angle). The unit is radian.
            speed_range (Union[float, Tuple[float, float]], optional): The range of speed. The valid input is a positive float or a tuple of two floats represents (min speed, max speed). The unit is meter per second (m/s).
            accel_range (Union[float, Tuple[float, float]], optional): The range of acceleration. The valid input is a positive float or a tuple of two floats represents (min acceleration, max acceleration). The unit is meter per second squared (m/s$^2$).
            interval (int, optional): The time interval between the current state and the new state. The unit is millisecond.
            delta_t (int, optional): The discrete time step for the simulation. The unit is millisecond.
        """
        self.lf = lf
        self.lr = lr
        self.wheel_base = lf + lr
        self.delta_t = delta_t

        if isinstance(steer_rate_range, float):
            self.steer_rate_range = None if steer_rate_range < 0 else [-steer_rate_range, steer_rate_range]
        elif hasattr(steer_rate_range, "__len__") and len(steer_rate_range) == 2:
            if steer_rate_range[0] >= steer_rate_range[1]:
                self.steer_rate_range = None
            else:
                self.steer_rate_range = steer_rate_range
        else:
            self.steer_rate_range = None

        if isinstance(speed_range, float):
            self.speed_range = None if speed_range < 0 else [-speed_range, speed_range]
        elif hasattr(speed_range, "__len__") and len(speed_range) == 2:
            if speed_range[0] >= speed_range[1]:
                self.speed_range = None
            else:
                self.speed_range = speed_range
        else:
            self.speed_range = None

        if isinstance(accel_range, float):
            self.accel_range = None if accel_range < 0 else [-accel_range, accel_range]
        elif hasattr(accel_range, "__len__") and len(accel_range) == 2:
            if accel_range[0] >= accel_range[1]:
                self.accel_range = None
            else:
                self.accel_range = accel_range
        else:
            self.accel_range = None


    # def _step(self, state: State, acceleration: float, steering_rate: float) -> State:
    #     x, y = state.location
    #     heading = state.heading
    #     velocity = state.speed
    #     steering_angle = state.steering_angle  # 假设State类中已经有steering_angle属性

    #     vehicle_length = self.lf + self.lr  # 使用车辆的总轴距作为车长

    #     # 更新车辆状态
    #     delta_heading = (velocity / (0.6*vehicle_length)) * np.tan(steering_angle) * self.delta_t
    #     x += velocity * np.cos(heading) * self.delta_t
    #     y += velocity * np.sin(heading) * self.delta_t
    #     steering_angle += steering_rate * self.delta_t # 更新前轮转角
    #     heading += delta_heading
    #     velocity += acceleration * self.delta_t  # 更新速度

    #     # # 保持变量在合理的范围内
    #     # velocity = np.clip(velocity, *self.speed_range) if not self.speed_range is None else velocity
    #     # steering_angle = np.clip(steering_angle, *self.steer_range) if not self.steer_range is None else steering_angle

    #     # 创建新的State实例来表示更新后的状态
    #     new_state = State(
    #         frame=state.frame + 1, 
    #         x=x,
    #         y=y,
    #         heading=np.mod(heading, 2 * np.pi),  # 保证航向角在0到2π之间
    #         speed=velocity,
    #         steering_angle=steering_angle  # 假设State类中已经有steering_angle属性
    #     )

    #     return new_state
    
    def _step(self, state: State, acceleration: float, steering_angle: float) -> State:
        x, y = state.location
        heading = state.heading
        velocity = state.speed
        steering_angle_cur = state.steering_angle  # 假设State类中已经有steering_angle属性

        vehicle_length = self.lf + self.lr  # 使用车辆的总轴距作为车长

        # 更新车辆状态
        delta_heading = (velocity / (0.6*vehicle_length)) * np.tan(steering_angle_cur) * self.delta_t
        x += velocity * np.cos(heading) * self.delta_t
        y += velocity * np.sin(heading) * self.delta_t
        heading += delta_heading
        velocity += acceleration * self.delta_t  # 更新速度

        # # 保持变量在合理的范围内
        # velocity = np.clip(velocity, *self.speed_range) if not self.speed_range is None else velocity
        # steering_angle = np.clip(steering_angle, *self.steer_range) if not self.steer_range is None else steering_angle

        # 创建新的State实例来表示更新后的状态
        new_state = State(
            frame=state.frame + 1, 
            x=x,
            y=y,
            heading=np.mod(heading, 2 * np.pi),  # 保证航向角在0到2π之间
            speed=velocity,
            steering_angle=steering_angle  # 假设State类中已经有steering_angle属性
        )

        return new_state


    def step(self, state: State, acceleration: float, steering_rate: float) -> State:
        # accel = np.clip(accel, *self.accel_range) if not self.accel_range is None else accel
        # delta = np.clip(delta, *self.steer_rate_range) if not self.steer_rate_range is None else delta
        # interval = interval if interval is not None else self.interval

        next_state = self._step(state, acceleration, steering_rate)

        return next_state

    # def verify_state(self, state: State, last_state: State, interval: int = None) -> bool:
    #     """This function provides a very rough check for the state transition.

    #     Args:
    #         state (State): The current state of the traffic participant.
    #         last_state (State): The last state of the traffic participant.
    #         interval (int, optional): The time interval between the last state and the new state. The unit is millisecond.

    #     Returns:
    #         True if the new state is valid, False otherwise.
    #     """
    #     interval = interval if interval is None else state.frame - last_state.frame
    #     self.delta_t = float(interval) / 1000
    #     last_speed = last_state.speed

    #     if None in [self.steer_rate_range, self.speed_range, self.accel_range]:
    #         return True

    #     steer_rate_range = np.array(self.steer_rate_range)
    #     beta_range = np.arctan(self.lr / self.wheel_base * steer_rate_range)

    #     # check that heading is in the range. heading_range may be larger than 2 * np.pi
    #     heading_range = np.mod(
    #         last_state.heading + last_speed / self.wheel_base * np.sin(beta_range) * self.delta_t, 2 * np.pi
    #     )
    #     if (
    #         heading_range[0] < heading_range[1]
    #         and not heading_range[0] <= state.heading <= heading_range[1]
    #     ):
    #         return False
    #     if heading_range[0] > heading_range[1] and not (
    #         heading_range[0] <= state.heading or state.heading <= heading_range[1]
    #     ):
    #         return False

    #     # check that speed is in the range
    #     speed_range = np.clip(last_speed + np.array(self.accel_range) * self.delta_t, *self.speed_range)
    #     if not speed_range[0] <= state.speed <= speed_range[1]:
    #         return False

    #     # check that x, y are in the range
    #     x_range = last_state.x + speed_range * np.cos(last_state.heading + beta_range) * self.delta_t
    #     y_range = last_state.y + speed_range * np.sin(last_state.heading + beta_range) * self.delta_t

    #     if not x_range[0] < state.x < x_range[1] or not y_range[0] < state.y < y_range[1]:
    #         return False

    #     return True
