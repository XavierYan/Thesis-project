#  Copyright (c) 2023-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import Union

import numpy as np
import torch

import vmas.simulator.core
import vmas.simulator.utils
from vmas.simulator.dynamics.common import Dynamics


class KinematicBicycle(Dynamics):
    # For the implementation of the kinematic bicycle model, see the equation (2) of the paper Polack, Philip, et al. "The kinematic bicycle model: A consistent model for planning feasible trajectories for autonomous vehicles?." 2017 IEEE intelligent vehicles symposium (IV). IEEE, 2017.
    def __init__(
        self,
        world: vmas.simulator.core.World,
        width: float,
        l_f: float,
        l_r: float,
        max_steering_angle: float,
        integration: str = "rk4",  # one of "euler", "rk4"
    ):
        super().__init__()
        
        assert integration in (
            "rk4",
            "euler",
        ), "Integration method must be 'euler' or 'rk4'."
        
        self.width = width
        self.l_f = l_f  # Distance between the front axle and the center of gravity
        self.l_r = l_r  # Distance between the rear axle and the center of gravity
        self.max_steering_angle = max_steering_angle
        self.dt = world.dt
        self.integration = integration
        self.world = world

    def f(self, state, steering_command, v_command):
        theta = state[:, 2]  # Yaw angle
        beta = torch.atan2(
            torch.tan(steering_command) * self.l_r / (self.l_f + self.l_r), torch.tensor(1)
        )  # [-pi, pi] slip angle 
        dx = v_command * torch.cos(theta + beta)
        dy = v_command * torch.sin(theta + beta)
        dtheta = v_command / (self.l_f + self.l_r) * torch.cos(beta) * torch.tan(steering_command)
        return torch.stack(
            (dx, dy, dtheta), dim=1
        )  # [batch_size,3]
            
    def euler(self, state, steering_command, v_command):
        # Calculate the change in state using Euler's method
        # For Euler's method, see https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Active_Calculus_(Boelkins_et_al.)/07%3A_Differential_Equations/7.03%3A_Euler's_Method (the full link may not be recognized properly, please copy and paste in your browser)
        return self.dt * self.f(state, steering_command, v_command)

    def runge_kutta(self, state, steering_command, v_command):
        # Calculate the change in state using fourth-order Runge-Kutta method
        # For Runge-Kutta method, see https://math.libretexts.org/Courses/Monroe_Community_College/MTH_225_Differential_Equations/3%3A_Numerical_Methods/3.3%3A_The_Runge-Kutta_Method
        k1 = self.f(state, steering_command, v_command)
        k2 = self.f(state + self.dt * k1 / 2, steering_command, v_command)
        k3 = self.f(state + self.dt * k2 / 2, steering_command, v_command)
        k4 = self.f(state + self.dt * k3, steering_command, v_command)
        return (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @property
    def needed_action_size(self) -> int:
        return 2

    # def process_action(self):
    #     # Extracts the velocity and steering angle from the agent's actions and convert them to physical force and torque
    #     v_command = self.agent.action.u[:, 0]
    #     # Ensure speed is within bounds
    #     v_command = torch.clamp(
    #         v_command, -self.agent.max_speed, self.agent.max_speed
    #     )
    #     steering_command =  self.agent.action.u[:, 1]
    #     # Ensure steering angle is within bounds
    #     steering_command = torch.clamp(
    #         steering_command, -self.max_steering_angle, self.max_steering_angle
    #     )

    #     # Current state of the agent
    #     state = torch.cat((self.agent.state.pos, self.agent.state.rot), dim=1)

    #     v_cur_x = self.agent.state.vel[:, 0] # Current velocity in x-direction
    #     v_cur_y = self.agent.state.vel[:, 1] # Current velocity in y-direction
    #     v_cur_angular = self.agent.state.ang_vel[:, 0] # Current angular velocity

    #     # Select the integration method to calculate the change in state
    #     if self.integration == "euler":
    #         state_change = self.euler(state, steering_command, v_command)
    #     else:
    #         state_change = self.runge_kutta(state, steering_command, v_command)

    #     # Calculate the accelerations required to achieve the change in state.
    #     # Note that in `core.py`, position is updated by `pos += vel_new * dt` with `vel_new` being calculated by `vel_new += force / mass`. The same principle applies to rotation.
    #     acceleration_x = (state_change[:, 0] - v_cur_x * self.dt) / self.dt**2
    #     acceleration_y = (state_change[:, 1] - v_cur_y * self.dt) / self.dt**2
    #     acceleration_angular = (state_change[:, 2] - v_cur_angular * self.dt) / self.dt**2

    #     # Calculate the forces required for the linear accelerations
    #     force_x = self.agent.mass * acceleration_x
    #     force_y = self.agent.mass * acceleration_y

    #     # Calculate the torque required for the angular acceleration
    #     torque = self.agent.moment_of_inertia * acceleration_angular

    #     # Update the physical force and torque required for the user inputs
    #     self.agent.state.force[:, vmas.simulator.utils.X] = force_x
    #     self.agent.state.force[:, vmas.simulator.utils.Y] = force_y
    #     self.agent.state.torque = torque.unsqueeze(-1)


    # def process_action(self):
    #     # Extracts the velocity and steering angle from the agent's actions and convert them to physical force and torque
    #     acceleration = self.agent.action.u[:, 0]
    #     # print("acc", acceleration)
    #     acceleration = torch.clamp(acceleration, -7, 3)
    #     # Ensure speed is within bounds
    #     v_command = torch.norm(self.agent.state.vel, p=2) + acceleration * self.dt

    #     # print("v_command", v_command)

    #     steering_command =  self.agent.action.u[:, 1]
    #     # Ensure steering angle is within bounds
    #     steering_command = torch.clamp(
    #         steering_command, -self.max_steering_angle, self.max_steering_angle
    #     )
    #     # steering_command = torch.deg2rad(steering_command)


    #     # # Current state of the agent
    #     # state = torch.cat((self.agent.state.pos, self.agent.state.rot), dim=1)


    #     # # Select the integration method to calculate the change in state
    #     # if self.integration == "euler":
    #     #     state_change = self.euler(state, steering_command, v_command)
    #     # else:
    #     #     state_change = self.runge_kutta(state, steering_command, v_command)

    #     # force_x = self.agent.mass * acceleration * torch.cos(self.agent.state.rot)
    #     # force_y = self.agent.mass * acceleration * torch.sin(self.agent.state.rot)
    #     # torque = self.agent.moment_of_inertia * (state_change[:, 2] - self.agent.state.ang_vel[:, 0] * self.dt) / self.dt**2

    #     # # Update the physical force and torque required for the user inputs
    #     # self.agent.state.force[:, vmas.simulator.utils.X] = force_x
    #     # self.agent.state.force[:, vmas.simulator.utils.Y] = force_y
    #     # self.agent.state.torque = torque.unsqueeze(-1)

    #     state = torch.cat((self.agent.state.pos, self.agent.state.rot), dim=1)

    #     v_cur_x = self.agent.state.vel[:, 0] # Current velocity in x-direction
    #     v_cur_y = self.agent.state.vel[:, 1] # Current velocity in y-direction
    #     v_cur_angular = self.agent.state.ang_vel[:, 0] # Current angular velocity

    #     # Select the integration method to calculate the change in state
    #     if self.integration == "euler":
    #         state_change = self.euler(state, steering_command, v_command)
    #     else:
    #         state_change = self.runge_kutta(state, steering_command, v_command)

    #     # Calculate the accelerations required to achieve the change in state.
    #     # Note that in `core.py`, position is updated by `pos += vel_new * dt` with `vel_new` being calculated by `vel_new += force / mass`. The same principle applies to rotation.
    #     acceleration_x = acceleration * torch.cos(self.agent.state.rot[:, 0])
    #     acceleration_y = acceleration * torch.sin(self.agent.state.rot[:, 0])
    #     acceleration_angular = (state_change[:, 2] - v_cur_angular * self.dt) / self.dt**2

    #     # Calculate the forces required for the linear accelerations
    #     force_x = self.agent.mass * acceleration_x
    #     force_y = self.agent.mass * acceleration_y

    #     # Calculate the torque required for the angular acceleration
    #     torque = self.agent.moment_of_inertia * acceleration_angular

    #     # Update the physical force and torque required for the user inputs
    #     self.agent.state.force[:, vmas.simulator.utils.X] = force_x
    #     self.agent.state.force[:, vmas.simulator.utils.Y] = force_y
    #     self.agent.state.torque = torque.unsqueeze(-1)



    def process_action(self):
        # acceleration = self.agent.action.u[:, 0]
        ## print("acc", acceleration)
        # acceleration = torch.clamp(acceleration, -7, 3)
        # print("acc", acceleration)
        # Ensure speed is within bounds
        # print("before", torch.norm(self.agent.state.vel, p=2,dim = -1))
        # v_command = torch.norm(self.agent.state.vel, p=2, dim = -1) + acceleration * self.dt
        # print("vel", v_command)

        v_command = self.agent.action.u[:, 0]
        # v_command = torch.clamp(
        #     v_command, -self.agent.max_speed, self.agent.max_speed
        # )
        v_command = torch.clamp(
            v_command, 0, self.agent.max_speed
        )

        # print("before", torch.norm(self.agent.state.vel, p=2,dim = -1))
        # print("vel", v_command)
        steering_command =  self.agent.action.u[:, 1]
        # print("steering", steering_command)
        # Ensure steering angle is within bounds
        steering_command = torch.clamp(
            steering_command, -self.max_steering_angle, self.max_steering_angle
        )
        # steering_command = torch.deg2rad(steering_command)

        # Current state of the agent
        # state = torch.cat((self.agent.state.pos, self.agent.state.rot), dim=1)

        # v_cur_x = self.agent.state.vel[:, 0] # Current velocity in x-direction
        # v_cur_y = self.agent.state.vel[:, 1] # Current velocity in y-direction
        # v_cur_angular = self.agent.state.ang_vel[:, 0] # Current angular velocity

        # # Select the integration method to calculate the change in state
        # if self.integration == "euler":
        #     state_change = self.euler(state, steering_command, v_command)
        # else:
        #     state_change = self.runge_kutta(state, steering_command, v_command)

        # # Calculate the accelerations required to achieve the change in state.
        # # Note that in `core.py`, position is updated by `pos += vel_new * dt` with `vel_new` being calculated by `vel_new += force / mass`. The same principle applies to rotation.
        # acceleration_x = (state_change[:, 0] - v_cur_x * self.dt) / self.dt**2
        # acceleration_y = (state_change[:, 1] - v_cur_y * self.dt) / self.dt**2
        # acceleration_angular = (state_change[:, 2] - v_cur_angular * self.dt) / self.dt**2

        # # Calculate the forces required for the linear accelerations
        # force_x = self.agent.mass * acceleration_x
        # force_y = self.agent.mass * acceleration_y

        # # Calculate the torque required for the angular acceleration
        # torque = self.agent.moment_of_inertia * acceleration_angular

        # # Update the physical force and torque required for the user inputs
        # self.agent.state.force[:, vmas.simulator.utils.X] = force_x
        # self.agent.state.force[:, vmas.simulator.utils.Y] = force_y
        # self.agent.state.torque = torque.unsqueeze(-1)

        v_cur_x = v_command * torch.cos(self.agent.state.rot[:, 0]) 
        v_cur_y = v_command * torch.sin(self.agent.state.rot[:, 0]) 
        self.agent.state.vel = torch.stack((v_cur_x, v_cur_y), dim=-1) 

        self.agent.state.pos += self.agent.state.vel * self.dt 
        dtheta = v_command / (self.l_f + self.l_r) * torch.tan(steering_command)  
        self.agent.state.ang_vel[:, 0] = dtheta 
        self.agent.state.rot += self.agent.state.ang_vel * self.dt  

        self.agent.state.rot = torch.fmod(self.agent.state.rot, 2 * torch.pi)
        self.agent.state.rot = torch.where(self.agent.state.rot < 0, self.agent.state.rot + 2 * torch.pi, self.agent.state.rot)