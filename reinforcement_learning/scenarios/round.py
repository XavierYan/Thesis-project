import os
import random
import sys
import time
import lanelet2
from termcolor import colored, cprint
print(sys.path)
# !Important: Add project root to system path if you want to run this file directly
script_dir = os.path.dirname(__file__) # Directory of the current script
project_root = os.path.dirname(script_dir) # Project root directory
if project_root not in sys.path:
    sys.path.append(project_root)

print(sys.path)
    
import torch
from torch import Tensor
from typing import Dict

from scipy.interpolate import interp1d

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

import matplotlib.pyplot as plt

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, World
from vmas.simulator.scenario import BaseScenario
# from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle

from utilities.kinematic_bicycle import KinematicBicycle
from utilities.colors import Color

from utilities.helper_training import Parameters

from utilities.helper_scenario import Distances, Normalizers, Observations, Penalties, ReferencePathsAgentRelated, ReferencePathsMapRelated, Rewards, Thresholds, Collisions, Timer, Constants, CircularBuffer, StateBuffer, InitialStateBuffer, Noise, Evaluation, exponential_decreasing_fcn, get_color_and_line_style, get_distances_between_agents, get_perpendicular_distances, get_rectangle_vertices, get_short_term_reference_path, interX, angle_eliminate_two_pi, transform_from_global_to_local_coordinate,trajectory_to_vectors_with_ids

from utilities.get_cpm_lab_map import get_map_data
from utilities.get_reference_paths import get_reference_paths

import math
from lanelet2.projection import UtmProjector


from matplotlib.patches import Polygon
import torch
from tactics2d.map.parser import OSMParser
import sys
import sys
sys.path.append('/home/xavier/project/thesis/src/')
sys.path.append(".")
sys.path.append("..")

import json
import logging
import os
import platform
import time
import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import Point

# from tactics2d.dataset_parser import InteractionParser
from tactics2d.map.parser import OSMParser
from tactics2d.sensor import RenderManager, SingleLineLidar, TopDownCamera
from tactics2d.traffic import ScenarioDisplay
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State
from tactics2d.traffic import ScenarioManager, ScenarioStatus, TrafficStatus
from tactics2d.traffic.event_detection import NoAction, OffLane, OutBound, TimeExceed
from tactics2d.participant.trajectory.trajectory import Trajectory
import matplotlib.pyplot as plt


# from __future__ import annotations
from utilities.colors import Color

import math
import typing
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union, Sequence

import torch
from torch import Tensor

from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.joints import Joint
from vmas.simulator.physics import (
    _get_closest_point_line,
    _get_closest_point_box,
    _get_closest_line_box,
    _get_closest_box_box,
    _get_closest_points_line_line,
    _get_inner_point_box,
)
from vmas.simulator.sensors import Sensor
from vmas.simulator.utils import (
    Color,
    X,
    Y,
    override,
    LINE_MIN_DIST,
    COLLISION_FORCE,
    JOINT_FORCE,
    Observable,
    DRAG,
    LINEAR_FRICTION,
    ANGULAR_FRICTION,
    TorchUtils,
)
from vmas.simulator.core import Agent, Box, World, Action, AgentState, Entity,Shape ,Sphere


import os
import sys
import time
from termcolor import colored, cprint

    
import torch
from torch import Tensor
from typing import Dict

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

import matplotlib.pyplot as plt

from vmas import render_interactively

from utilities.colors import Color



initial_velocity = 5
rel_vel = 5                     #target speed

## Simulation parameters 
n_agents = 4                    # The number of agents
dt = 1/25                       # Sample time in [s]
max_steps = 1000                # Maximum simulation steps
is_real_time_rendering = True   # Simulation will be paused at each time step for real-time rendering

agent_max_speed = 14
agent_max_acc = 7           # Maximum allowed speed in [m/s]
agent_max_steering_angle = torch.pi/6   # Maximum allowed steering angle in degree
agent_mass = 0.5                # The mass of each agent in [kg]
viewer_zoom = 1.44

agent_mass = 0.5                # The mass of each agent in [kg]

## Geometry
world_x_dim = 2500              # The x-dimension of the world in [m]
world_y_dim = 1500               # The y-dimension of the world in [m]
agent_width = 2              # The width of the agent in [m]
agent_length = 4             # The length of the agent in [m]
wheelbase_front = agent_length / 2                  # Front wheelbase in [m]
wheelbase_rear = agent_length - wheelbase_front     # Rear wheelbase in [m]
lane_width = 5               # The (rough) width of each lane in [m]

## Reward
r_p_normalizer = 100    # Rewards and renalties must be normalized to range [-1, 1]

reward_progress = 10 / r_p_normalizer   # Reward for moving along reference paths
reward_vel = 10 / r_p_normalizer         # Reward for moving in high velocities. 
reward_reach_goal = 0 / r_p_normalizer  # Goal-reaching reward

## Penalty
penalty_deviate_from_ref_path = -10 / r_p_normalizer      # Penalty for deviating from reference paths
threshold_deviate_from_ref_path = (lane_width - agent_width) / 2 # Use for penalizing of deviating from reference path
penalty_near_boundary = -20 / r_p_normalizer *4             # Penalty for being too close to lanelet boundaries
penalty_near_other_agents = -20 / r_p_normalizer *6         # Penalty for being too close to other agents
penalty_collide_with_agents = -100 / r_p_normalizer *6      # Penalty for colliding with other agents 
penalty_collide_with_boundaries = -100 / r_p_normalizer *6  # Penalty for colliding with lanelet boundaries
penalty_change_steering = -2 / r_p_normalizer          # Penalty for changing steering too quick
penalty_time = 5 / r_p_normalizer                     # Penalty for losing time

threshold_reach_goal = agent_width / 2  # Threshold less than which agents are considered at their goal positions

threshold_change_steering = 5 # Threshold above which agents will be penalized for changing steering too quick [degree]

threshold_near_boundary_high = 1    # Threshold beneath which agents will started be 
                                                                        # penalized for being too close to lanelet boundaries
threshold_near_boundary_low = 0 # Threshold above which agents will be penalized for being too close to lanelet boundaries 

threshold_near_other_agents_c2c_high = agent_length + 4      # Threshold beneath which agents will started be 
                                                        # penalized for being too close to other agents (for center-to-center distance)
threshold_near_other_agents_c2c_low = agent_width # Threshold above which agents will be penalized (for center-to-center distance, 
                                                        # if a c2c distance is less than the half of the agent width, they are colliding, which will be penalized by another penalty)

threshold_near_other_agents_MTV_high = agent_length +1 # Threshold beneath which agents will be penalized for 
                                                    # being too close to other agents (for MTV-based distance)
threshold_near_other_agents_MTV_low = 0             # Threshold above which agents will be penalized for
                                                    # being too close to other agents (for MTV-based distance)
                                                    
threshold_no_reward_if_too_close_to_boundaries = agent_width / 10
threshold_no_reward_if_too_close_to_other_agents = agent_width / 6


is_testing_mode = False             # In testing mode, collisions do not lead to the termination of the simulation 
is_visualize_short_term_path = True
is_visualize_extra_info = False
render_title = "Multi-Agent Reinforcement Learning for Connected and Automated Vehicles"

# Reference path
n_points_short_term = 15             # The number of points on short-term reference paths
n_points_nearing_boundary = 15       # The number of points on nearing boundaries to be observed
sample_interval_ref_path = 1                 # Integer, sample interval from the long-term reference path for the short-term reference paths 
max_ref_path_points = 230           # The estimated maximum points on the reference path

## Observation
is_partial_observation = True       # Set to True if each agent can only observe a subset of other agents, i.e., limitations on sensor range are considered
                                    # Note that this also reduces the observation size, which may facilitate training
n_nearing_agents_observed = 4       # The number of most nearing agents to be observed by each agent. This parameter will be used if `is_partial_observation = True`

noise_level = 0.2 * agent_width     # Noise will be generated by the standary normal distribution. This parameter controls the noise level

n_stored_steps = 5      # The number of steps to store (include the current step). At least one
n_observed_steps = 1    # The number of steps to observe (include the current step). At least one, and at most `n_stored_steps`

# Training parameters
training_strategy = "1" # One of {"1", "2", "3", "4"}
                            # "1": Train in a single, comprehensive scenario
                            # "2": Train in a single, comprehensive scenario with prioritized replay buffer
                            # "3": Train in a single, comprehensive scenario with challenging initial state buffer
                            # "4": Train in a specific scenario (our)
buffer_size = 100               # Used only when training_strategy == "3"
n_steps_before_recording = 10   # The states of agents at time step `current_time_step - n_steps_before_recording` before collisions will be recorded and used later when resetting the envs
n_steps_stored = n_steps_before_recording # Store previous `n_steps_stored` steps of states
probability_record = 1.0            # Probability of recording a collision-event into the buffer
probability_use_recording = 0.2     # Probability of using an recording when resetting an env

scenario_probabilities = [1.0, 0.0, 0.0] # 1 for intersection, 2 for merge-in, 3 for merge-out scenario

is_ego_view = True                  # Global coordinate system (bird view) or local coordinate system (ego view)
is_apply_mask = True
is_observe_vertices = True
is_observe_distance_to_agents = True
is_observe_distance_to_boundaries = True # Whether to observe points on lanelet boundaries or the distance to lanelet boundaries
is_observe_distance_to_center_line = True

is_observe_ref_path_other_agents = False
is_use_mtv_distance = True
is_add_noise = True

colors = [
    Color.blue100, Color.purple100, Color.violet100, Color.bordeaux100, Color.red100, Color.orange100, Color.maygreen100, Color.green100, Color.turquoise100, Color.petrol100, Color.yellow100, Color.magenta100, Color.black100,
    Color.blue50, Color.purple50, Color.violet50, Color.bordeaux50, Color.red50, Color.orange50, Color.maygreen50, Color.green50, Color.turquoise50, Color.petrol50, Color.yellow50, Color.magenta50, Color.black50,
] # Each agent will get a different color


class ScenarioRoadTraffic(BaseScenario):
    def initialize_vehicle(self, track_id, start_x = 0, start_y = 0, end_x = 0, end_y = 0, heading = 0, vx = 5, vy = 5, ax = 0, ay = 0, steering_angle = 0,length = 4, width = 2, vehicle_type="medium_car"):

        trajectory = Trajectory(id_=track_id)

        initial_state = State(
            frame=0,
            x=start_x,
            y=start_y,
            heading=heading,
            vx=vx,
            vy=vy,
            ax=ax,
            ay=ay,
            steering_angle = steering_angle
        )

        trajectory.add_state(initial_state)

        # initialize vehicle
        vehicle = Vehicle(
            id_=track_id,
            type_=vehicle_type,
            trajectory=trajectory,
            start_x= start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,            
            length=length,
            width=width,
            ll_map=self.ll_map

        )

        return vehicle
    
    def calculate_distances_and_boundaries(self, env_i, i_agent):

        vehicle_polygon = self.world.agents[i_agent].vehicle.box

        width2 = self.world.agents[i_agent].shape.width / 2

        vehicle_center = vehicle_polygon.centroid

        reference_trajectory = self.world.agents[i_agent].vehicle.reference_trajectory

        left_boundary = self.world.agents[i_agent].vehicle.left_boundary
        right_boundary = self.world.agents[i_agent].vehicle.right_boundary



        vertices_dist_left = [left_boundary.distance(Point(coord)) for coord in vehicle_polygon.exterior.coords[:-1]]
        vertices_dist_right = [right_boundary.distance(Point(coord)) for coord in vehicle_polygon.exterior.coords[:-1]]

        dist_to_ref_path = vehicle_center.distance(reference_trajectory)
        dist_to_left_boundary = vehicle_center.distance(left_boundary) - width2
        dist_to_right_boundary = vehicle_center.distance(right_boundary) - width2

  
        self.distances.ref_paths[env_i, i_agent] = torch.tensor(dist_to_ref_path)
        self.distances.left_boundaries[env_i, i_agent, 0] = torch.tensor(dist_to_left_boundary)
        self.distances.left_boundaries[env_i, i_agent, 1:5] = torch.tensor(vertices_dist_left)
        self.distances.right_boundaries[env_i, i_agent, 0] = torch.tensor(dist_to_right_boundary)
        self.distances.right_boundaries[env_i, i_agent, 1:5] = torch.tensor(vertices_dist_right)

    
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # print("[DEBUG] make_world() road_traffic")
        # device = torch.device("mps") # For mac with m chip to use GPU acceleration (however, seems not be fully supported by VMAS)
        self.shared_reward = kwargs.get("shared_reward", False)
        
        width = kwargs.get("width", agent_width)
        l_f = kwargs.get("l_f", wheelbase_front)    # Front wheelbase
        l_r = kwargs.get("l_r", wheelbase_rear)     # Rear wheelbase
        # max_steering_angle = kwargs.get("max_steering_angle", torch.deg2rad(torch.tensor(agent_max_steering_angle, device=device, dtype=torch.float32)))
        max_steering_angle = torch.pi/6
        max_acc = kwargs.get("max_speed", agent_max_acc)
        max_speed = kwargs.get("max_speed", agent_max_speed)

        
        self.render_origin = [85, -50]

        # self.viewer_size = (int(world_x_dim * resolution_factor), int(world_y_dim * resolution_factor))
        self.viewer_size = (world_x_dim, world_y_dim)

        self.viewer_zoom = viewer_zoom

        # Specify parameters if not given
        if not hasattr(self, "parameters"):
            self.parameters = Parameters(
                n_agents=n_agents,
                is_partial_observation=is_partial_observation,
                is_testing_mode=is_testing_mode,
                is_visualize_short_term_path=is_visualize_short_term_path,
                max_steps=max_steps,
                training_strategy=training_strategy,
                n_nearing_agents_observed=n_nearing_agents_observed,
                is_real_time_rendering=is_real_time_rendering,
                n_points_short_term=n_points_short_term,
                dt=dt,

                is_ego_view=is_ego_view,
                is_apply_mask=is_apply_mask,
                is_observe_vertices=is_observe_vertices,
                is_observe_distance_to_agents=is_observe_distance_to_agents,
                is_observe_distance_to_boundaries=is_observe_distance_to_boundaries,
                is_observe_distance_to_center_line=is_observe_distance_to_center_line,
                
                is_use_mtv_distance=is_use_mtv_distance,
                scenario_probabilities=scenario_probabilities,
                is_add_noise=is_add_noise,
                is_observe_ref_path_other_agents=is_observe_ref_path_other_agents,
                is_visualize_extra_info=is_visualize_extra_info,
                render_title=render_title,
            )
        
        # Logs
        if self.parameters.is_testing_mode:
            print(colored(f"[INFO] Testing mode", "red"))
        else:        
            if self.parameters.training_strategy == "1":
                print(colored("[INFO] Vanilla model", "red"))
            elif self.parameters.training_strategy == "2":
                print(colored("[INFO] Enable prioritized experience replay", "red"), colored("(state of the art)", "blue"))
            elif self.parameters.training_strategy == "3":
                print(colored("[INFO] Enable using the challenging initial state buffer", "red"), colored("(state of the art)"), "blue")
            elif self.parameters.training_strategy == "4":
                print(colored("[INFO] Enable training in partial map", "red"), colored("(our)", "blue"))

        # Parameter adjustment to meet simulation requirements
        if self.parameters.training_strategy == "4":
            self.parameters.n_agents = 4
            print(colored(f"[INFO] Changed the number of agents to {self.parameters.n_agents}", "black"))
        self.parameters.n_nearing_agents_observed = min(self.parameters.n_nearing_agents_observed, self.parameters.n_agents - 1)
        self.n_agents = self.parameters.n_agents
        
        # Timer for the first env
        self.timer = Timer(
            start=time.time(),
            end=0,
            step=torch.zeros(batch_dim, device=device, dtype=torch.int32), # Each environment has its own time step
            step_duration=torch.zeros(self.parameters.max_steps, device=device, dtype=torch.float32),
            step_begin=time.time(),
            render_begin=0,
        )
        
        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=torch.tensor(world_x_dim, device=device, dtype=torch.float32),
            y_semidim=torch.tensor(world_y_dim, device=device, dtype=torch.float32),
            # x_semidim=None,
            # y_semidim=None,
            dt=self.parameters.dt
        )
        
        # Get map data 
        ll_origin = lanelet2.io.Origin(50.890926, 6.173557)
        projector = UtmProjector(ll_origin)
        self.ll_map = lanelet2.io.load("/home/xavier/project/thesis/src/dataset/map/Aachen_Roundabout_change.osm", projector)

        tree = ET.parse('/home/xavier/project/thesis/src/dataset/map/rounD_0.osm')  # 请替换为你的地图文件路径
        root = tree.getroot()

        parser = OSMParser(lanelet2=True)  # 设置lanelet2参数根据你的需要

        # 解析地图
        self.map_data = parser.parse(root,{
                    "proj": "utm",
                    "ellps": "WGS84",
                    "zone": 32,
                    "datum": "WGS84"
                },[6.173554,50.890924])
        
        self.ref_paths_agent_related = ReferencePathsAgentRelated(
            long_term=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32), # Long-term reference paths of agents
            long_term_vec_normalized=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32),
            left_boundary=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32),
            right_boundary=torch.zeros((batch_dim, self.n_agents, max_ref_path_points, 2), device=device, dtype=torch.float32),
            exit=torch.zeros((batch_dim, self.n_agents, 2, 2), device=device, dtype=torch.float32),
            n_points_long_term=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32),
            n_points_left_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32),
            n_points_right_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32),
            short_term=torch.zeros((batch_dim, self.n_agents, self.parameters.n_points_short_term, 2), device=device, dtype=torch.float32), # Short-term reference path
            short_term_indices = torch.zeros((batch_dim, self.n_agents, self.parameters.n_points_short_term), device=device, dtype=torch.int32),
            n_points_nearing_boundary=torch.tensor(n_points_nearing_boundary, device=device, dtype=torch.int32),
            nearing_points_left_boundary=torch.zeros((batch_dim, self.n_agents, n_points_nearing_boundary, 2), device=device, dtype=torch.float32), # Nearing left boundary
            nearing_points_right_boundary=torch.zeros((batch_dim, self.n_agents, n_points_nearing_boundary, 2), device=device, dtype=torch.float32), # Nearing right boundary
        )   


        # The shape of each agent is considered a rectangle with 4 vertices. 
        # The first vertex is repeated at the end to close the shape.
        self.vertices = torch.zeros((batch_dim, self.n_agents, 5, 2), device=device, dtype=torch.float32) 
 
        weighting_ref_directions = torch.linspace(1, 0.2, steps=self.parameters.n_points_short_term, device=device, dtype=torch.float32)
        weighting_ref_directions /= weighting_ref_directions.sum()
        self.rewards = Rewards(
            progress=torch.tensor(reward_progress, device=device, dtype=torch.float32),
            weighting_ref_directions=weighting_ref_directions, # Progress in the weighted directions (directions indicating by closer short-term reference points have higher weights)
            higth_v=torch.tensor(reward_vel, device=device, dtype=torch.float32),
            reach_goal=torch.tensor(reward_reach_goal, device=device, dtype=torch.float32),
        )
        self.rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)

        self.penalties = Penalties(
            deviate_from_ref_path=torch.tensor(penalty_deviate_from_ref_path, device=device, dtype=torch.float32),
            weighting_deviate_from_ref_path=1.5,
            near_boundary=torch.tensor(penalty_near_boundary, device=device, dtype=torch.float32),
            near_other_agents=torch.tensor(penalty_near_other_agents, device=device, dtype=torch.float32),
            collide_with_agents=torch.tensor(penalty_collide_with_agents, device=device, dtype=torch.float32),
            collide_with_boundaries=torch.tensor(penalty_collide_with_boundaries, device=device, dtype=torch.float32),
            change_steering=torch.tensor(penalty_change_steering, device=device, dtype=torch.float32),
            time=torch.tensor(penalty_time, device=device, dtype=torch.float32),
        )

        self.parameters.deviate_from_ref_path = torch.tensor(penalty_deviate_from_ref_path, device=device, dtype=torch.float32)
        self.parameters.weighting_deviate_from_ref_path = 1.5
        self.parameters.near_boundary = torch.tensor(penalty_near_boundary, device=device, dtype=torch.float32)
        self.parameters.near_other_agents = torch.tensor(penalty_near_other_agents, device=device, dtype=torch.float32)
        self.parameters.collide_with_agents = torch.tensor(penalty_collide_with_agents, device=device, dtype=torch.float32)
        self.parameters.collide_with_boundaries = torch.tensor(penalty_collide_with_boundaries, device=device, dtype=torch.float32)
        self.parameters.change_steering = torch.tensor(penalty_change_steering, device=device, dtype=torch.float32)
        self.parameters.time=torch.tensor(penalty_time, device=device, dtype=torch.float32)

        
        self.observations = Observations(
            is_partial=torch.tensor(self.parameters.is_partial_observation, device=device, dtype=torch.bool),
            n_nearing_agents=torch.tensor(self.parameters.n_nearing_agents_observed, device=device, dtype=torch.int32),
            noise_level=torch.tensor(noise_level, device=device, dtype=torch.float32),
            n_stored_steps=torch.tensor(n_stored_steps, device=device, dtype=torch.int32),
            n_observed_steps=torch.tensor(n_observed_steps, device=device, dtype=torch.int32),
            nearing_agents_indices=torch.zeros((batch_dim, self.n_agents, self.parameters.n_nearing_agents_observed), device=device, dtype=torch.int32)
        )
        assert self.observations.n_stored_steps >= 1, "The number of stored steps should be at least 1."
        assert self.observations.n_observed_steps >= 1, "The number of observed steps should be at least 1."
        assert self.observations.n_stored_steps >= self.observations.n_observed_steps, "The number of stored steps should be greater or equal than the number of observed steps."
        
        if self.parameters.is_ego_view:
            self.observations.past_pos = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2), device=device, dtype=torch.float32))
            self.observations.past_rot = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents), device=device, dtype=torch.float32))
            self.observations.past_vertices = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, 4, 2), device=device, dtype=torch.float32))
            self.observations.past_vel = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2), device=device, dtype=torch.float32))
            self.observations.past_abs_vel = CircularBuffer(torch.zeros((n_stored_steps,batch_dim, self.n_agents, self.n_agents, 1), device=device, dtype=torch.float32))
            self.observations.past_short_term_ref_points = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, self.parameters.n_points_short_term, 2), device=device, dtype=torch.float32))
            self.observations.past_left_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, self.ref_paths_agent_related.n_points_nearing_boundary, 2), device=device, dtype=torch.float32))
            self.observations.past_right_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents, self.ref_paths_agent_related.n_points_nearing_boundary, 2), device=device, dtype=torch.float32))
        else:
            # Bird view
            self.observations.past_pos = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, 2), device=device, dtype=torch.float32))
            self.observations.past_rot = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
            self.observations.past_vertices = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, 4, 2), device=device, dtype=torch.float32))
            self.observations.past_vel = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, 2), device=device, dtype=torch.float32))
            self.observations.past_abs_vel = CircularBuffer(torch.zeros((n_stored_steps,batch_dim, self.n_agents, self.n_agents, 1), device=device, dtype=torch.float32))
            self.observations.past_short_term_ref_points = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.parameters.n_points_short_term, 2), device=device, dtype=torch.float32))
            self.observations.past_left_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.ref_paths_agent_related.n_points_nearing_boundary, 2), device=device, dtype=torch.float32))
            self.observations.past_right_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.ref_paths_agent_related.n_points_nearing_boundary, 2), device=device, dtype=torch.float32))

        self.observations.past_action_vel = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_action_steering = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_ref_path = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_boundaries = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_left_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_right_boundary = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents), device=device, dtype=torch.float32))
        self.observations.past_distance_to_agents = CircularBuffer(torch.zeros((n_stored_steps, batch_dim, self.n_agents, self.n_agents), device=device, dtype=torch.float32))

        self.normalizers = Normalizers(
            pos=torch.tensor([agent_length * 4-3, agent_length * 2], device=device, dtype=torch.float32),
            pos_world=torch.tensor([world_x_dim, world_y_dim], device=device, dtype=torch.float32),
            v=torch.tensor(max_speed, device=device, dtype=torch.float32),
            rot=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32),
            action_steering=max_steering_angle,
            action_vel=torch.tensor(max_speed, device=device, dtype=torch.float32),
            distance_lanelet=torch.tensor(lane_width * 3, device=device, dtype=torch.float32),
            distance_ref=torch.tensor(lane_width * 3, device=device, dtype=torch.float32),
            distance_agent=torch.tensor(agent_length * 10, device=device, dtype=torch.float32),
        )
        
        # Distances to boundaries and reference path, and also the closest point on the reference paths of agents
        if self.parameters.is_use_mtv_distance:
            distance_type = "MTV" # One of {"c2c", "MTV"}
        else:
            distance_type = "c2c" # One of {"c2c", "MTV"}
        # print(colored("[INFO] Distance type: ", "black"), colored(distance_type, "blue"))
            
        self.distances = Distances(
            type = distance_type, # Type of distances between agents
            agents=torch.zeros(world.batch_dim, self.n_agents, self.n_agents, dtype=torch.float32),
            left_boundaries=torch.zeros((batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32), # The first entry for the center, the last 4 entries for the four vertices
            right_boundaries=torch.zeros((batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32),
            boundaries=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            ref_paths=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            closest_point_on_ref_path=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32),
            closest_point_on_left_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32),
            closest_point_on_right_b=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.int32),
        )

        self.thresholds = Thresholds(
            reach_goal=torch.tensor(threshold_reach_goal, device=device, dtype=torch.float32),
            deviate_from_ref_path=torch.tensor(threshold_deviate_from_ref_path, device=device, dtype=torch.float32),
            near_boundary_low=torch.tensor(threshold_near_boundary_low, device=device, dtype=torch.float32),
            near_boundary_high=torch.tensor(threshold_near_boundary_high, device=device, dtype=torch.float32),
            near_other_agents_low=torch.tensor(
                threshold_near_other_agents_c2c_low if self.distances.type == "c2c" else threshold_near_other_agents_MTV_low, 
                device=device,
                dtype=torch.float32
            ),
            near_other_agents_high=torch.tensor(
                threshold_near_other_agents_c2c_high if self.distances.type == "c2c" else threshold_near_other_agents_MTV_high, 
                device=device,
                dtype=torch.float32
            ),
            change_steering=torch.tensor(threshold_change_steering, device=device, dtype=torch.float32).deg2rad(),
            no_reward_if_too_close_to_boundaries=torch.tensor(threshold_no_reward_if_too_close_to_boundaries, device=device, dtype=torch.float32),
            no_reward_if_too_close_to_other_agents=torch.tensor(threshold_no_reward_if_too_close_to_other_agents, device=device, dtype=torch.float32),
            distance_mask_agents=self.normalizers.pos[0],
        )
        
        # Create agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Box(length=l_f+l_r, width=width),
                color=colors[i],
                collide=False,
                render_action=False,
                u_range=[max_speed, max_steering_angle],    # Control command serves as velocity command 
                u_multiplier=[1, 1],
                max_speed=max_speed,
                dynamics=KinematicBicycle(                  # Use the kinematic bicycle model for each agent
                    world, 
                    width=width, 
                    l_f=l_f, 
                    l_r=l_r, 
                    max_steering_angle=max_steering_angle, 
                    integration="rk4"                       # one of {"euler", "rk4"}
                )
            )                            
            world.add_agent(agent)
        
        self.constants = Constants(
            env_idx_broadcasting=torch.arange(batch_dim, device=device, dtype=torch.int32).unsqueeze(-1),
            empty_action_vel=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            empty_action_steering=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            mask_pos=torch.tensor(1, device=device, dtype=torch.float32),
            mask_zero=torch.tensor(0, device=device, dtype=torch.float32),
            mask_one=torch.tensor(1, device=device, dtype=torch.float32),
            reset_agent_min_distance=torch.tensor((l_f+l_r) ** 2 + width ** 2, device=device, dtype=torch.float32).sqrt() * 1.2,
        )
        
        # Initialize collision matrix
        self.collisions = Collisions(
            with_agents=torch.zeros((world.batch_dim, self.n_agents, self.n_agents), device=device, dtype=torch.bool),
            with_lanelets=torch.zeros((world.batch_dim, self.n_agents), device=device, dtype=torch.bool),
            with_entry_segments=torch.zeros((world.batch_dim, self.n_agents), device=device, dtype=torch.bool),
            with_exit_segments=torch.zeros((world.batch_dim, self.n_agents), device=device, dtype=torch.bool),
        )
        
        self.initial_state_buffer = InitialStateBuffer( # Used only when "training_strategy == '4'"
            probability_record=torch.tensor(probability_record, device=device, dtype=torch.float32),
            probability_use_recording=torch.tensor(probability_use_recording, device=device, dtype=torch.float32),
            buffer=torch.zeros((buffer_size, self.n_agents, 8), device=device, dtype=torch.float32), # [pos_x, pos_y, rot, vel_x, vel_y, scenario_id, path_id, point_id]
        )

        # Store the states of agents at previous several time steps
        self.state_buffer = StateBuffer(
            buffer=torch.zeros((n_steps_stored, batch_dim, self.n_agents, 6), device=device, dtype=torch.float32), # [pos_x, pos_y, rot, vel_x, vel_y, scenario_id, path_id, point_id],
        )
        
        self.evaluation = Evaluation(
            pos_traj=torch.zeros((batch_dim, self.n_agents, self.parameters.max_steps, 2), device=device, dtype=torch.float32),
            v_traj=torch.zeros((batch_dim, self.n_agents, self.parameters.max_steps, 2), device=device, dtype=torch.float32),
            rot_traj=torch.zeros((batch_dim, self.n_agents, self.parameters.max_steps), device=device, dtype=torch.float32),
            deviation_from_ref_path=torch.zeros((batch_dim, self.n_agents, self.parameters.max_steps), device=device, dtype=torch.float32),
            path_tracking_error_mean=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
        )
        
        return world

    def reset_world_at(self, env_index: int = None, agent_index: int = None):
        # print(f"[DEBUG] reset_world_at(): env_index = {env_index}")
        """
        This function resets the world at the specified env_index.

        Args:
        :param env_index: index of the environment to reset. If None a vectorized reset should be performed

        """
        if self.parameters.is_testing_mode == False:
            entrance_points = {
                "entrance_1": [(24.5, -36.7, 344.4), (32.5, -39.08, 344), (40.7, -41.9, 344.8)],
                "entrance_2": [(25.4, -33.5, 344.4), (34, -35.9, 344.4), (43.7, -38.8,344.4)],
                "entrance_3": [(136.6, -62.4, 157), (123.7, -57.4, 157.3), (115.3, -54.25, 157.1)],
                "entrance_4": [(134.2, -57.9,157), (125.9, -54.6,157), (117.36, -51.49,157.3)],
                "entrance_5": [(72.57, -79.86, 54.2)],
                "entrance_6": [(89.6, -15.84, 245)],
                "entrance_7": [(62.3, -52.7, 290)],
                "entrance_8": [(72.8, -28.6, 210)],
                "entrance_9": [(99.2, -37.5, 124)],
                "entrance_10": [(90.9, -65.2, 24)],

            }
            entrance_lanes = list(entrance_points.keys())

            exit_points = {
                "exit_1": (31.6, -30.7),
                "exit_2": (130.35, -65.5),
                "exit_3": (63.9, -76.71),
                "exit_4": (99.6, -17.6),
            }
            exit_lanes = list(exit_points.keys())

            adjacent_pairs = {
                "entrance_1": ["exit_1"],
                "entrance_2": ["exit_1"],
                "entrance_3": ["exit_3"],
                "entrance_4": ["exit_3"],
                "entrance_5": ["exit_2"],
                "entrance_6": ["exit_4"],
            }

        else:
            entrance_points = {
            "entrance_1": [(24.5, -36.7, 344.4), (40.7, -41.9, 344.8)],

            "entrance_2": [(25.4, -33.5, 344.4), (34, -35.9, 344.4), (43.7, -38.8,344.4)],
            "entrance_3": [(136.6, -62.4, 157), (123.7, -57.4, 157.3), (115.3, -54.25, 157.1)],
            "entrance_4": [(134.2, -57.9,157), (125.9, -54.6,157), (117.36, -51.49,157.3)],
            "entrance_3": [(136.6, -62.4, 157),(123.7, -57.4, 157.3),(115.3, -54.25, 157.1)],
            "entrance_4": [  (117.36, -51.49,157.3)],
            "entrance_5": [(72.57, -79.86, 54.2)],
            "entrance_6": [(89.6, -15.84, 245)],
            }
            entrance_lanes = list(entrance_points.keys())

            exit_points = {
                "exit_1": (31.6, -30.7),
                "exit_2": (130.35, -65.5),
                "exit_3": (63.9, -76.71),
                "exit_4": (99.6, -17.6),
            }
            exit_lanes = list(exit_points.keys())

        agents = self.world.agents

        is_reset_single_agent = agent_index is not None
        
        if is_reset_single_agent:
            assert env_index is not None

        for env_i in [env_index] if env_index is not None else range(self.world.batch_dim):
            # Begining of a new simulation (only record for the first env)
            if env_i == 0:
                self.timer.step_duration[:] = 0
                self.timer.start = time.time()
                self.timer.step_begin = time.time()
                self.timer.end = 0
                
            if not is_reset_single_agent:
                # Each time step of a simulation
                self.timer.step[env_i] = 0

            used_start_positions = set()

            for i_agent in range(self.n_agents) if not is_reset_single_agent else agent_index.unsqueeze(0):

                while True:
                    entrance_lane = random.choice(entrance_lanes)
                    start_pos = random.choice(entrance_points[entrance_lane])
                    if start_pos not in used_start_positions:
                        used_start_positions.add(start_pos)
                        break
                end_pos = exit_points[random.choice(exit_lanes)]

                # end_pos = exit_points[random.choice(exit_lanes)]
                # while True:
                #     entrance_lane = random.choice(entrance_lanes)
                #     start_pos = random.choice(entrance_points[entrance_lane])
                #     if start_pos not in used_start_positions:
                #         used_start_positions.add(start_pos)
                #         possible_exits = [e for e in exit_points.keys() if e not in adjacent_pairs[entrance_lane]]
                #         if possible_exits:
                #             exit_lane = random.choice(possible_exits)
                #             end_pos = exit_points[exit_lane]
                #         break                



                agents[i_agent].set_pos(torch.hstack([torch.tensor([start_pos[0]]), torch.tensor([start_pos[1]])]), batch_index=env_i)

                vel_start_abs = initial_velocity # Random initial velocity
                vel_start = torch.hstack([vel_start_abs * torch.cos(np.deg2rad(torch.tensor(start_pos[2]))), vel_start_abs * torch.sin(np.deg2rad(torch.tensor(start_pos[2])))])

                agents[i_agent].set_rot(torch.tensor(np.deg2rad(start_pos[2])), batch_index=env_i)
                agents[i_agent].set_vel(vel_start, batch_index=env_i)

                vehicle = self.initialize_vehicle(i_agent, start_pos[0], start_pos[1], end_pos[0], end_pos[1], np.deg2rad(start_pos[2]))

                reference_trajectory = vehicle.reference_trajectory
                x, y = zip(*reference_trajectory)
                distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
                cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
                distance_interp = np.arange(0, cumulative_distances[-1], 1)
                x_interp_func = interp1d(cumulative_distances, x, kind='linear')
                y_interp_func = interp1d(cumulative_distances, y, kind='linear')
                x_interp = x_interp_func(distance_interp)
                y_interp = y_interp_func(distance_interp)
                reference_trajectory_points = torch.tensor(list(zip(x_interp, y_interp)))
                n_points_long_term = reference_trajectory_points.shape[0]
                self.ref_paths_agent_related.long_term[env_i, i_agent, 0:n_points_long_term, :] = reference_trajectory_points
                self.ref_paths_agent_related.long_term[env_i, i_agent, n_points_long_term:, :] = reference_trajectory_points[-1,:]
                self.ref_paths_agent_related.n_points_long_term[env_i, i_agent] = n_points_long_term

                left_boundary = vehicle.left_boundary
                x, y = zip(*left_boundary)
                distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
                cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
                distance_interp_left = np.arange(0, cumulative_distances[-1], 1)
                x_interp_left_func = interp1d(cumulative_distances, x, kind='linear')
                y_interp_left_func = interp1d(cumulative_distances, y, kind='linear')
                x_interp_left = x_interp_left_func(distance_interp_left)
                y_interp_left = y_interp_left_func(distance_interp_left)
                left_boundary_points = torch.tensor(list(zip(x_interp_left, y_interp_left)))
                n_points_left_b = left_boundary_points.shape[0]
                self.ref_paths_agent_related.left_boundary[env_i, i_agent, 0:n_points_left_b, :] = left_boundary_points
                self.ref_paths_agent_related.left_boundary[env_i, i_agent, n_points_left_b:, :] = left_boundary_points[-1,:]
                self.ref_paths_agent_related.n_points_left_b[env_i, i_agent] = n_points_left_b


                right_boundary = vehicle.right_boundary
                x, y = zip(*right_boundary)
                distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
                cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
                distance_interp_right = np.arange(0, cumulative_distances[-1], 1)
                x_interp_right_func = interp1d(cumulative_distances, x, kind='linear')
                y_interp_right_func = interp1d(cumulative_distances, y, kind='linear')
                x_interp_right = x_interp_right_func(distance_interp_right)
                y_interp_right = y_interp_right_func(distance_interp_right)
                right_boundary_points = torch.tensor(list(zip(x_interp_right, y_interp_right)))
                n_points_right_b = right_boundary_points.shape[0]
                self.ref_paths_agent_related.right_boundary[env_i, i_agent, 0:n_points_right_b, :] = right_boundary_points
                self.ref_paths_agent_related.right_boundary[env_i, i_agent, n_points_right_b:, :] = right_boundary_points[-1,:]
                self.ref_paths_agent_related.n_points_right_b[env_i, i_agent] = n_points_right_b

                if self.parameters.is_testing_mode == False:
                    self.ref_paths_agent_related.exit[env_i, i_agent, 0, :] = left_boundary_points[-1, :]
                    self.ref_paths_agent_related.exit[env_i, i_agent, 1, :] = right_boundary_points[-1, :]
                else:
                    self.ref_paths_agent_related.exit[env_i, i_agent, 0, :] = left_boundary_points[-10, :]
                    self.ref_paths_agent_related.exit[env_i, i_agent, 1, :] = right_boundary_points[-10, :]

            # The operations below can be done for all envs in parallel
            if env_index is None:
                if env_i == (self.world.batch_dim - 1):
                    env_j = slice(None) # `slice(None)` is equivalent to `:`
                else:
                    continue
            else:
                env_j = env_i
                # self.calculate_distances_and_boundaries(env_i, i_agent)


            for i_agent in range(self.n_agents) if not is_reset_single_agent else agent_index.unsqueeze(0):
                # Distance from the center of gravity (CG) of the agent to its reference path
                self.distances.ref_paths[env_j, i_agent], self.distances.closest_point_on_ref_path[env_j, i_agent] = get_perpendicular_distances(
                    point=agents[i_agent].state.pos[env_j, :],
                    polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
                    n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_j, i_agent],
                )
                # Distances from CG to left boundary
                center_2_left_b, self.distances.closest_point_on_left_b[env_j, i_agent] = get_perpendicular_distances(
                    point=agents[i_agent].state.pos[env_j, :],
                    polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                    n_points_long_term=self.ref_paths_agent_related.n_points_left_b[env_j, i_agent],
                )
                self.distances.left_boundaries[env_j, i_agent, 0] = center_2_left_b - (agents[i_agent].shape.width / 2)
                # Distances from CG to right boundary
                center_2_right_b, self.distances.closest_point_on_right_b[env_j, i_agent] = get_perpendicular_distances(
                    point=agents[i_agent].state.pos[env_j, :],
                    polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                    n_points_long_term=self.ref_paths_agent_related.n_points_right_b[env_j, i_agent],
                )
                self.distances.right_boundaries[env_j, i_agent, 0] = center_2_right_b - (agents[i_agent].shape.width / 2)
                # Calculate the positions of the four vertices of the agents
                self.vertices[env_j, i_agent] = get_rectangle_vertices(
                    center=agents[i_agent].state.pos[env_j, :],
                    yaw=agents[i_agent].state.rot[env_j, :], 
                    width=agents[i_agent].shape.width, 
                    length=agents[i_agent].shape.length,
                    is_close_shape=True
                )
                # Distances from the four vertices of the agent to its left and right lanelet boundary
                for c_i in range(4):
                    self.distances.left_boundaries[env_j, i_agent, c_i + 1], _ = get_perpendicular_distances(
                        point=self.vertices[env_j, i_agent, c_i, :],
                        polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                        n_points_long_term=self.ref_paths_agent_related.n_points_left_b[env_j, i_agent],
                    )
                    self.distances.right_boundaries[env_j, i_agent, c_i + 1], _ = get_perpendicular_distances(
                        point=self.vertices[env_j, i_agent, c_i, :],
                        polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                        n_points_long_term=self.ref_paths_agent_related.n_points_right_b[env_j, i_agent],
                    )
                # Distance from agent to its left/right lanelet boundary is defined as the minimum distance among five distances (four vertices, CG)
                self.distances.boundaries[env_j, i_agent], _ = torch.min(
                    torch.hstack(
                        (
                            self.distances.left_boundaries[env_j, i_agent],
                            self.distances.right_boundaries[env_j, i_agent]
                        )
                    ),
                    dim=-1
                )
        
                # Get the short-term reference paths
                self.ref_paths_agent_related.short_term[env_j, i_agent], _ = get_short_term_reference_path(
                    polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
                    index_closest_point=self.distances.closest_point_on_ref_path[env_j, i_agent],
                    n_points_to_return=self.parameters.n_points_short_term, 
                    device=self.world.device,
                    n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_j, i_agent],
                    sample_interval=1,
                    n_points_shift=1,
                )

                # if not self.parameters.is_observe_distance_to_boundaries:
                    # Get nearing points on boundaries
                self.ref_paths_agent_related.nearing_points_left_boundary[env_j, i_agent], _ = get_short_term_reference_path(
                    polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                    index_closest_point=self.distances.closest_point_on_left_b[env_j, i_agent],
                    n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
                    device=self.world.device,
                    n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_j, i_agent],
                    sample_interval=1,
                    n_points_shift=1,
                )
                self.ref_paths_agent_related.nearing_points_right_boundary[env_j, i_agent], _ = get_short_term_reference_path(
                    polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                    index_closest_point=self.distances.closest_point_on_right_b[env_j, i_agent],
                    n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
                    device=self.world.device,
                    n_points_long_term=self.ref_paths_agent_related.n_points_long_term[env_j, i_agent],
                    sample_interval=1,
                    n_points_shift=1,
                )

            # Compute mutual distances between agents 
            # TODO Enable the possibility of computing the mutual distances of agents in a single env
            mutual_distances = get_distances_between_agents(self=self, distance_type=self.distances.type, is_set_diagonal=True)
            # Reset mutual distances of all envs
            self.distances.agents[env_j, :, :] = mutual_distances[env_j, :, :]

            # Reset the collision matrix
            self.collisions.with_agents[env_j, :, :] = False
            self.collisions.with_lanelets[env_j, :] = False
            self.collisions.with_exit_segments[env_j, :] = False
            
        # Reset the state buffer
        self.state_buffer.reset() 
        state_add = torch.cat(
            (
                torch.stack([a.state.pos for a in agents], dim=1),
                torch.stack([a.state.rot for a in agents], dim=1),
                torch.stack([a.state.vel for a in agents], dim=1),
                torch.zeros((self.world.batch_dim, self.n_agents, 1), device=self.world.device)
            ),
            dim=-1
        )
        self.state_buffer.add(state_add) # Add new state


    def process_action(self, agent: Agent):
        # print("[DEBUG] process_action()")
        if hasattr(agent, "dynamics") and hasattr(agent.dynamics, "process_force"):
            agent.dynamics.process_force()
            assert not agent.action.u.isnan().any()
            assert not agent.action.u.isinf().any()
        else:
            # The agent does not have a dynamics property, or it does not have a process_force method
            pass

    def reward(self, agent: Agent):
        # print("[DEBUG] reward()")
        # Initialize
        self.rew[:] = 0
        
        # Get the index of the current agent
        agent_index = self.world.agents.index(agent)

        # If rewards are shared among agents
        if self.shared_reward:
            # TODO Support shared reward
            raise NotImplementedError
            
        # [update] mutual distances between agents, vertices of each agent, and collision matrices
        if agent_index == 0: # Avoid repeated computations
            # Timer
            self.timer.step_duration[self.timer.step] = time.time() - self.timer.step_begin                
            self.timer.step_begin = time.time() # Set to the current time as the begin of the current time step
            self.timer.step += 1 # Increment step by 1
            # print(self.timer.step)

            # Update distances between agents
            self.distances.agents = get_distances_between_agents(self=self, distance_type=self.distances.type, is_set_diagonal=True)
            self.collisions.with_agents[:] = False   # Reset
            self.collisions.with_lanelets[:] = False # Reset
            self.collisions.with_entry_segments[:] = False # Reset
            self.collisions.with_exit_segments[:] = False # Reset

            for a_i in range(self.n_agents):
                self.vertices[:, a_i] = get_rectangle_vertices(
                    center=self.world.agents[a_i].state.pos,
                    yaw=self.world.agents[a_i].state.rot,
                    width=self.world.agents[a_i].shape.width,
                    length=self.world.agents[a_i].shape.length,
                    is_close_shape=True,
                )
                # Update the collision matrices
                if self.distances.type == "c2c":
                    for a_j in range(a_i+1, self.n_agents):
                        # Check for collisions between agents using the interX function
                        collision_batch_index = interX(self.vertices[:, a_i], self.vertices[:, a_j], False)
                        self.collisions.with_agents[torch.nonzero(collision_batch_index), a_i, a_j] = True
                        self.collisions.with_agents[torch.nonzero(collision_batch_index), a_j, a_i] = True
                elif self.distances.type == "MTV":
                    # Two agents collide if their MTV-based distance is zero
                    self.collisions.with_agents[:] = self.distances.agents == 0

                # Check for collisions between agents and lanelet boundaries
                collision_with_left_boundary = interX(
                    L1=self.vertices[:, a_i], 
                    L2=self.ref_paths_agent_related.left_boundary[:, a_i], 
                    is_return_points=False,
                ) # [batch_dim]
                collision_with_right_boundary = interX(
                    L1=self.vertices[:, a_i], 
                    L2=self.ref_paths_agent_related.right_boundary[:, a_i],
                    is_return_points=False,
                ) # [batch_dim]
                self.collisions.with_lanelets[(collision_with_left_boundary | collision_with_right_boundary), a_i] = True

                # Check for collisions with entry or exit segments (only need if agents' reference paths are not a loop)

                self.collisions.with_exit_segments[:, a_i] = interX(
                    L1=self.vertices[:, a_i],
                    L2=self.ref_paths_agent_related.exit[:, a_i],
                    is_return_points=False,
                )
                    
        # Distance from the center of gravity (CG) of the agent to its reference path
        self.distances.ref_paths[:, agent_index], self.distances.closest_point_on_ref_path[:, agent_index] = get_perpendicular_distances(
            point=agent.state.pos, 
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index],
        )
        # Distances from CG to left boundary
        center_2_left_b, self.distances.closest_point_on_left_b[:, agent_index] = get_perpendicular_distances(
            point=agent.state.pos[:, :], 
            polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_left_b[:, agent_index],
        )
        self.distances.left_boundaries[:, agent_index, 0] = center_2_left_b - (agent.shape.width / 2)
        # Distances from CG to right boundary
        center_2_right_b, self.distances.closest_point_on_right_b[:, agent_index] = get_perpendicular_distances(
            point=agent.state.pos[:, :], 
            polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_right_b[:, agent_index],
        )
        self.distances.right_boundaries[:, agent_index, 0] = center_2_right_b - (agent.shape.width / 2)
        # Distances from the four vertices of the agent to its left and right lanelet boundary
        for c_i in range(4):
            self.distances.left_boundaries[:, agent_index, c_i + 1], _ = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_left_b[:, agent_index],
            )
            self.distances.right_boundaries[:, agent_index, c_i + 1], _ = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_right_b[:, agent_index],
            )
        # Distance from agent to its left/right lanelet boundary is defined as the minimum distance among five distances (four vertices, CG)
        self.distances.boundaries[:, agent_index], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[:, agent_index],
                    self.distances.right_boundaries[:, agent_index]
                )
            ),
            dim=-1
        )
            
        ##################################################
        ## [reward] forward movement
        ##################################################
        latest_state = self.state_buffer.get_latest(n=1)
        move_vec = (agent.state.pos - latest_state[:, agent_index, 0:2]).unsqueeze(1) # Vector of the current movement
        
        ref_points_vecs = self.ref_paths_agent_related.short_term[:, agent_index] - latest_state[:, agent_index, 0:2].unsqueeze(1) # Vectors from the previous position to the points on the short-term reference path
        move_projected = torch.sum(move_vec * ref_points_vecs, dim=-1)
        move_projected_weighted = torch.matmul(move_projected, self.rewards.weighting_ref_directions) # Put more weights on nearing reference points

        reward_movement = move_projected_weighted / (agent.max_speed * self.world.dt) * self.rewards.progress
        self.rew += reward_movement # Relative to the maximum possible movement
        
        #################################################
        # [reward] high velocity
        #################################################   
        v_proj = torch.sum(agent.state.vel.unsqueeze(1) * ref_points_vecs, dim=-1).mean(-1)
        # factor_moving_direction = torch.where(v_proj>0, 1, 0.1) # Get penalty if move in negative direction
        
        # reward_vel = factor_moving_direction * v_proj / agent.max_speed * self.rewards.higth_v
        # self.rew += reward_vel



        ##################################################
        ## [reward] keep target speed
        ##################################################
        speed_norms = agent.state.vel.norm(dim=1).to(self.world.device)
        reward_rel_vel = torch.tensor([1]).to(self.world.device) - torch.abs(speed_norms - torch.tensor([rel_vel]).unsqueeze(-1).to(self.world.device))/torch.tensor([rel_vel]).to(self.world.device)
        self.rew += reward_rel_vel.squeeze(0)




        ##################################################
        ## [reward] reach goal
        ##################################################
        reward_goal = self.collisions.with_exit_segments[:, agent_index] * 5
        self.rew += reward_goal

        ##################################################
        ## [penalty] close to lanelet boundaries
        ##################################################        
        penalty_close_to_lanelets = exponential_decreasing_fcn(
            x=self.distances.boundaries[:, agent_index],
            x0=self.thresholds.near_boundary_low, 
            x1=self.thresholds.near_boundary_high,
        ) * self.penalties.near_boundary
        self.rew += penalty_close_to_lanelets

        ##################################################
        ## [penalty] close to other agents
        ##################################################
        mutual_distance_exp_fcn = exponential_decreasing_fcn(
            x=self.distances.agents[:, agent_index, :], 
            x0=self.thresholds.near_other_agents_low, 
            x1=self.thresholds.near_other_agents_high
        )
        penalty_close_to_agents = torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents
        self.rew += penalty_close_to_agents

        ##################################################
        ## [penalty] deviating from reference path
        ##################################################
        self.rew += self.distances.ref_paths[:, agent_index] / self.penalties.weighting_deviate_from_ref_path * self.penalties.deviate_from_ref_path*5

        ##################################################
        ## [penalty] changing steering too quick
        ##################################################
        steering_current = self.world.agents[agent_index].action.u[:,1]
        steering_past = self.state_buffer.get_latest(n=1)[:, agent_index,-1]

        steering_change = torch.clamp(
            (steering_current - steering_past).abs() - self.thresholds.change_steering, # Not forget to denormalize
            min=0,
        )
        steering_change_reward_factor = steering_change / (2 * agent.u_range[1] - 2 * self.thresholds.change_steering)
        penalty_change_steering = steering_change_reward_factor * self.penalties.change_steering
        self.rew += penalty_change_steering

        # ##################################################
        # ## [penalty] colliding with other agents
        # ##################################################
        is_collide_with_agents = self.collisions.with_agents[:, agent_index]        
        penalty_collide_other_agents = is_collide_with_agents.any(dim=-1) * self.penalties.collide_with_agents
        penalty_collide_other_agents = is_collide_with_agents.any(dim=-1)

        self.rew += penalty_collide_other_agents

        ##################################################
        ## [penalty] colliding with lanelet boundaries
        ##################################################
        is_collide_with_lanelets = self.collisions.with_lanelets[:, agent_index]
        penalty_collide_lanelet = is_collide_with_lanelets * self.penalties.collide_with_boundaries
        penalty_collide_lanelet = is_collide_with_lanelets

        self.rew += penalty_collide_lanelet


        ##################################################
        ## [penalty] still
        ##################################################
        distance = self.distances.agents[:, agent_index, :]
        speed_norms = agent.state.vel.norm(dim=1)
        min_distances, _ = distance.min(dim=1)
        penalty_condition = (min_distances > 8) & (speed_norms < 1)
        self.rew += penalty_condition*(-4)



        ##################################################
        ## [penalty/reward] time
        ##################################################
        # Get time reward if moving in positive direction; otherwise get time penalty
        time_reward = torch.where(v_proj>0, 1, 2) * agent.state.vel.norm(dim=-1) / agent.max_speed * self.penalties.time
        self.rew += time_reward

        # [update] previous positions and short-term reference paths
        if agent_index == (self.n_agents - 1): # Avoid repeated updating
            state_add = torch.cat(
                (
                    torch.stack([a.state.pos for a in self.world.agents], dim=1),
                    torch.stack([a.state.rot for a in self.world.agents], dim=1),
                    torch.stack([a.state.vel for a in self.world.agents], dim=1),
                    torch.stack([a.action.u[:, 1:2].to(self.world.device) for a in self.world.agents], dim=1)
                ),
                dim=-1
            )
            self.state_buffer.add(state_add)
        
        self.ref_paths_agent_related.short_term[:, agent_index], _ = get_short_term_reference_path(
            polyline=self.ref_paths_agent_related.long_term[:, agent_index], 
            index_closest_point=self.distances.closest_point_on_ref_path[:, agent_index],
            n_points_to_return=self.parameters.n_points_short_term, 
            device=self.world.device,
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index],
            sample_interval=1,
        )

        # if not self.parameters.is_observe_distance_to_boundaries:
        # Get nearing points on boundaries
        self.ref_paths_agent_related.nearing_points_left_boundary[:, agent_index], _ = get_short_term_reference_path(
            polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
            index_closest_point=self.distances.closest_point_on_left_b[:, agent_index],
            n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
            device=self.world.device,
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index],
            sample_interval=1,
            n_points_shift=-2,
        )
        self.ref_paths_agent_related.nearing_points_right_boundary[:, agent_index], _ = get_short_term_reference_path(
            polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
            index_closest_point=self.distances.closest_point_on_right_b[:, agent_index],
            n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
            device=self.world.device,
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[:, agent_index],
            sample_interval=1,
            n_points_shift=-2,
        )

        assert not self.rew.isnan().any(), "Rewards contain nan."
        assert not self.rew.isinf().any(), "Rewards contain inf."
        
        
        # Clamed the reward to avoid abs(reward) being too large
        rew_clamed = torch.clamp(self.rew, min=-6, max=6)
        # print(rew_clamed)

        return rew_clamed
    
    def observation(self, agent: Agent):
        # print("[DEBUG] observation()")
        """
        Generate an observation for the given agent in all envs.

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            The observation for the given agent in all envs.
        """
        agent_index = self.world.agents.index(agent)
        
        device = self.world.device  # 选择合适的设备（CUDA 或 CPU）
        
        positions_global = torch.stack([a.state.pos for a in self.world.agents], dim=0).transpose(0, 1).to(device)
        rotations_global = torch.stack([a.state.rot for a in self.world.agents], dim=0).transpose(0, 1).squeeze(-1).to(device)
        
        if agent_index == 0: # Avoid repeated computations
            # Add new observation & normalize

            # 确保张量在相同设备上
            self.distances.agents = self.distances.agents.to(device)
            self.observations.past_distance_to_agents.add(self.distances.agents )
            self.observations.past_distance_to_ref_path.add(self.distances.ref_paths)
            self.observations.past_distance_to_left_boundary.add(torch.min(self.distances.left_boundaries, dim=-1)[0])
            self.observations.past_distance_to_right_boundary.add(torch.min(self.distances.right_boundaries, dim=-1)[0])
            self.observations.past_distance_to_boundaries.add(self.distances.boundaries )

            if self.parameters.is_ego_view:
                pos_i_others = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents, 2), device=device, dtype=torch.float32) # Positions of other agents relative to agent i
                rot_i_others = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents), device=device, dtype=torch.float32) # Rotations of other agents relative to agent i
                vel_i_others = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents, 2), device=device, dtype=torch.float32) # Velocities of other agents relative to agent i
                vel_i_abs_others = torch.zeros((self.world.batch_dim, self.n_agents, self.n_agents, 1), device=device, dtype=torch.float32)
                ref_i_others = torch.zeros_like((self.observations.past_short_term_ref_points.get_latest()), device=device) # Reference paths of other agents relative to agent i
                l_b_i_others = torch.zeros_like((self.observations.past_left_boundary.get_latest()), device=device) # Left boundaries of other agents relative to agent i
                r_b_i_others = torch.zeros_like((self.observations.past_right_boundary.get_latest()), device=device) # Right boundaries of other agents relative to agent i
                ver_i_others = torch.zeros_like((self.observations.past_vertices.get_latest()), device=device) # Vertices of other agents relative to agent i
                
                for a_i in range(self.n_agents):
                    pos_i = self.world.agents[a_i].state.pos.to(device)
                    rot_i = self.world.agents[a_i].state.rot.to(device)

                    # Store new observation - position
                    pos_i_others[:, a_i] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=positions_global,
                        rot_i=rot_i,
                    )

                    # Store new observation - rotation
                    relative_heading = rotations_global - rot_i
                    rot_i_others[:, a_i] = torch.atan2(torch.sin(relative_heading), torch.cos(relative_heading))
                    
                    for a_j in range(self.n_agents):
                        # Store new observation - velocities
                        rot_rel = rot_i_others[:, a_i, a_j].unsqueeze(1)
                        vel_abs = torch.norm(self.world.agents[a_j].state.vel.to(device), dim=1).unsqueeze(1) 
                        vel_i_others[:, a_i, a_j] = torch.hstack(
                            (
                                vel_abs * torch.cos(rot_rel), 
                                vel_abs * torch.sin(rot_rel)
                            )
                        )
                        vel_i_abs_others[:, a_i, a_j] = vel_abs
                        
                        # Store new observation - reference paths
                        ref_i_others[:, a_i, a_j] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.short_term[:, a_j].to(device),
                            rot_i=rot_i,
                        )
                        
                        # Store new observation - left boundary
                        # if not self.parameters.is_observe_distance_to_boundaries:
                        l_b_i_others[:, a_i, a_j] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.nearing_points_left_boundary[:, a_j].to(device),
                            rot_i=rot_i,
                        )
                        
                        # Store new observation - right boundary
                        r_b_i_others[:, a_i, a_j] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.nearing_points_right_boundary[:, a_j].to(device),
                            rot_i=rot_i,
                        )
                        
                        # Store new observation - vertices
                        ver_i_others[:, a_i, a_j] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.vertices[:, a_j, 0:4, :].to(device),
                            rot_i=rot_i,
                        )
                # Add new observations & normalize
                self.observations.past_pos.add(pos_i_others )
                self.observations.past_rot.add(rot_i_others )
                self.observations.past_vel.add(vel_i_others )
                self.observations.past_short_term_ref_points.add(ref_i_others)
                self.observations.past_abs_vel.add(vel_i_abs_others)
                self.observations.past_left_boundary.add(l_b_i_others)
                self.observations.past_right_boundary.add(r_b_i_others )
                self.observations.past_vertices.add(ver_i_others )
            # else: # Global coordinate system
            #     # Store new observations
            #     self.observations.past_pos.add(positions_global / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
            #     self.observations.past_vel.add(torch.stack([a.state.vel.to(device) for a in self.world.agents], dim=1) / self.normalizers.v)
            #     self.observations.past_rot.add(rotations_global[:] / self.normalizers.rot)
            #     self.observations.past_vertices.add(self.vertices[:, :, 0:4, :].to(device) / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
            #     self.observations.past_short_term_ref_points.add(self.ref_paths_agent_related.short_term[:].to(device) / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
            #     self.observations.past_left_boundary.add(self.ref_paths_agent_related.nearing_points_left_boundary.to(device)  / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))
            #     self.observations.past_right_boundary.add(self.ref_paths_agent_related.nearing_points_right_boundary.to(device)  / (self.normalizers.pos if self.parameters.is_ego_view else self.normalizers.pos_world))

            # Add new observation - actions & normalize
            if agent.action.u is None:
                self.observations.past_action_vel.add(self.constants.empty_action_vel)
                self.observations.past_action_steering.add(self.constants.empty_action_steering)
            else:
                self.observations.past_action_vel.add(torch.stack([a.action.u[:, 0].to(device) for a in self.world.agents], dim=1) )
                self.observations.past_action_steering.add(torch.stack([a.action.u[:, 1].to(device) for a in self.world.agents], dim=1))

        # Observation of other agents
        if self.observations.is_partial:
            # Each agent observes only a fixed number of nearest agents
            nearing_agents_distances, nearing_agents_indices = torch.topk(self.distances.agents[:, agent_index], k=self.observations.n_nearing_agents, largest=False)

            if self.parameters.is_apply_mask:
                # Nearing agents that are distant will be masked
                mask_nearing_agents_too_far = (nearing_agents_distances >= self.thresholds.distance_mask_agents)
            else:
                # Otherwise no agents will be masked
                mask_nearing_agents_too_far = torch.zeros((self.world.batch_dim, self.parameters.n_nearing_agents_observed), device=device, dtype=torch.bool)
            
            indexing_tuple_1 = (self.constants.env_idx_broadcasting,) + \
                            ((agent_index,) if self.parameters.is_ego_view else ()) + \
                            (nearing_agents_indices,)
            
            # Positions of nearing agents
            obs_pos_other_agents = self.observations.past_pos.get_latest().to(device)[indexing_tuple_1] # [batch_size, n_nearing_agents, 2]
            obs_pos_other_agents[mask_nearing_agents_too_far] = self.constants.mask_zero.to(device) # Position mask

            # Rotations of nearing agents
            obs_rot_other_agents = self.observations.past_rot.get_latest().to(device)[indexing_tuple_1] # [batch_size, n_nearing_agents]
            obs_rot_other_agents[mask_nearing_agents_too_far] = self.constants.mask_zero.to(device) # Rotation mask

            # Velocities of nearing agents
            obs_vel_other_agents = self.observations.past_vel.get_latest().to(device)[indexing_tuple_1] # [batch_size, n_nearing_agents]
            obs_vel_other_agents[mask_nearing_agents_too_far] = self.constants.mask_zero.to(device) # Velocity mask
            
            # Reference paths of nearing agents
            obs_ref_path_other_agents = self.observations.past_short_term_ref_points.get_latest().to(device)[indexing_tuple_1] # [batch_size, n_nearing_agents, n_points_short_term, 2]
            obs_ref_path_other_agents[mask_nearing_agents_too_far] = self.constants.mask_zero.to(device) # Reference-path mask

            # vertices of nearing agents
            obs_vertices_other_agents = self.observations.past_vertices.get_latest().to(device)[indexing_tuple_1] # [batch_size, n_nearing_agents, 4, 2]
            obs_vertices_other_agents[mask_nearing_agents_too_far] = self.constants.mask_one.to(device) # Reference-path mask
            
            # Distances to nearing agents
            obs_distance_other_agents = self.observations.past_distance_to_agents.get_latest().to(device)[self.constants.env_idx_broadcasting, agent_index, nearing_agents_indices] # [batch_size, n_nearing_agents]
            obs_distance_other_agents[mask_nearing_agents_too_far] = self.constants.mask_one.to(device) # Distance mask
        # else:
        #     obs_pos_other_agents = self.observations.past_pos.get_latest().to(device)[:, agent_index] # [batch_size, n_agents, 2]
        #     obs_rot_other_agents = self.observations.past_rot.get_latest().to(device)[:, agent_index] # [batch_size, n_agents, (n_agents)]
        #     obs_vel_other_agents = self.observations.past_vel.get_latest().to(device)[:, agent_index] # [batch_size, n_agents, 2]
        #     obs_ref_path_other_agents = self.observations.past_short_term_ref_points.get_latest().to(device)[:, agent_index] # [batch_size, n_agents, n_points_short_term, 2]
        #     obs_vertices_other_agents = self.observations.past_vertices.get_latest().to(device)[:, agent_index] # [batch_size, n_agents, 4, 2]
        #     obs_distance_other_agents = self.observations.past_distance_to_agents.get_latest().to(device)[:, agent_index] # [batch_size, n_agents]
        #     obs_distance_other_agents[:, agent_index] = 0 # Reset self-self distance to zero

        # Flatten the last dimensions to combine all features into a single dimension
        # obs_pos_other_agents_flat = obs_pos_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        # obs_rot_other_agents_flat = obs_rot_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        # obs_vel_other_agents_flat = obs_vel_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        # obs_ref_path_other_agents_flat = obs_ref_path_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        # obs_vertices_other_agents_flat = obs_vertices_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)
        # obs_distance_other_agents_flat = obs_distance_other_agents.reshape(self.world.batch_dim, self.observations.n_nearing_agents, -1)

        # # Observation of other agents
        # obs_others_list = [
        #     obs_vertices_other_agents_flat if self.parameters.is_observe_vertices else    # [other] vertices
        #     torch.cat(
        #         [
        #             obs_pos_other_agents_flat,                                      # [others] positions
        #             obs_rot_other_agents_flat,                                      # [others] rotations
        #         ], dim=-1
        #     ),  
        #     obs_vel_other_agents_flat,                                              # [others] velocities
        #     obs_distance_other_agents_flat if self.parameters.is_observe_distance_to_agents else None, # [others] mutual distances
        #     obs_ref_path_other_agents_flat if self.parameters.is_observe_ref_path_other_agents else None,               # [others] reference paths
        # ]
        # obs_others_list = [o for o in obs_others_list if o is not None] # Filter out None values
        # obs_other_agents = torch.cat(obs_others_list, dim=-1).reshape(self.world.batch_dim, -1) # [batch_size, -1]        

        indexing_tuple_3 = (self.constants.env_idx_broadcasting,) + \
                            (agent_index,) + \
                            ((agent_index,) if self.parameters.is_ego_view else ())        
        indexing_tuple_vel = (self.constants.env_idx_broadcasting,) + \
                            (agent_index,) + \
                            ((agent_index, 0) if self.parameters.is_ego_view else ()) # In local coordinate system, only the first component is interesting, as the second is always 0
        # All observations
        # obs_list = [
        #     None if self.parameters.is_ego_view else self.observations.past_pos.get_latest().to(device)[indexing_tuple_3].reshape(self.world.batch_dim, -1), # [own] position,
        #     None if self.parameters.is_ego_view else self.observations.past_rot.get_latest().to(device)[indexing_tuple_3].reshape(self.world.batch_dim, -1), # [own] rotation,
        #     self.observations.past_vel.get_latest().to(device)[indexing_tuple_vel].reshape(self.world.batch_dim, -1),                  # [own] velocity
        #     self.observations.past_short_term_ref_points.get_latest().to(device)[indexing_tuple_3].reshape(self.world.batch_dim, -1),       # [own] short-term reference path
        #     self.observations.past_distance_to_ref_path.get_latest().to(device)[:, agent_index].reshape(self.world.batch_dim, -1) if self.parameters.is_observe_distance_to_center_line else None, # [own] distances to reference paths
        #     self.observations.past_distance_to_left_boundary.get_latest().to(device)[:, agent_index].reshape(self.world.batch_dim, -1) if self.parameters.is_observe_distance_to_boundaries else self.observations.past_left_boundary.get_latest().to(device)[indexing_tuple_3].reshape(self.world.batch_dim, -1), # [own] left boundaries 
        #     self.observations.past_distance_to_right_boundary.get_latest().to(device)[:, agent_index].reshape(self.world.batch_dim, -1) if self.parameters.is_observe_distance_to_boundaries else self.observations.past_right_boundary.get_latest().to(device)[indexing_tuple_3].reshape(self.world.batch_dim, -1), # [own] right boundaries 
        #     obs_other_agents, # [others]
        # ]
        # obs_list = [o for o in obs_list if o is not None] # Filter out None values
        # critic_obs = torch.hstack(obs_list)
        

        ego_shape = [self.world.agents[agent_index].shape.length, self.world.agents[agent_index].shape.width,0,0,0]
        ego_shape = torch.tensor(ego_shape).unsqueeze(0).expand(self.world.batch_dim,-1).to(device)
        # ego_vel = self.world.agents[agent_index].state.vel
        ego_vel = self.observations.past_vel.get_latest().to(device)[:,agent_index,agent_index]

        ego_state = torch.cat((ego_shape,ego_vel), dim=-1).unsqueeze(1)

        # ego_shape = [self.world.agents[agent_index].shape.length, self.world.agents[agent_index].shape.width,0,0,0,0,0]
        # ego_shape = torch.tensor(ego_shape).unsqueeze(0).expand(self.world.batch_dim,-1).to(device)
        # # ego_vel = torch.norm(self.world.agents[agent_index].state.vel, dim = -1).unsqueeze(-1)

        # ego_state = ego_shape.unsqueeze(1)

        neighbour_shape = [self.world.agents[agent_index].shape.length, self.world.agents[agent_index].shape.width]
        neighbour_shape = torch.tensor(neighbour_shape).unsqueeze(0).unsqueeze(0).expand(self.world.batch_dim,self.observations.n_nearing_agents,-1).to(device)
        others_vel = self.observations.past_abs_vel.get_latest().to(device)[indexing_tuple_1]
        others_vel[mask_nearing_agents_too_far] = self.constants.mask_zero.to(device)
        neighbour_shape[mask_nearing_agents_too_far] = self.constants.mask_zero.to(device)
        neighbour_states = torch.cat((neighbour_shape,obs_pos_other_agents),dim = -1)
        neighbour_states = torch.cat((neighbour_states,obs_rot_other_agents.unsqueeze(-1)),dim = -1)
        neighbour_states = torch.cat((neighbour_states,obs_vel_other_agents),dim = -1).unsqueeze(-2) #(length, width, pos[1], pos[2], rel_heading, vel_rel[1], vel_rel[2])


        ego_reference = self.observations.past_short_term_ref_points.get_latest().to(device)[indexing_tuple_3]

        left_lane_boundary = self.observations.past_left_boundary.get_latest()[indexing_tuple_3]
        right_lane_boundary = self.observations.past_right_boundary.get_latest()[indexing_tuple_3]  
        lane_boundary = torch.cat([left_lane_boundary, right_lane_boundary], dim=1)

        neighbour_reference = trajectory_to_vectors_with_ids(obs_ref_path_other_agents)

        ego_reference_vectors = trajectory_to_vectors_with_ids(ego_reference).squeeze(1)

        lane_boundary_vectors = trajectory_to_vectors_with_ids(lane_boundary)

        # assert not (obs.abs() > 2).any(), "Observations contain values greater than 1."
        # obs = {"critic_obs" : critic_obs, "ego_state" : ego_state, "ego_reference":ego_reference_vectors, "neighbour_states":neighbour_states,
        #        "neighbour_reference":neighbour_reference, "lane_boundary" :lane_boundary_vectors }
        obs = {"ego_state" : ego_state, "ego_reference":ego_reference_vectors, "neighbour_states":neighbour_states,
               "neighbour_reference":neighbour_reference, "lane_boundary" :lane_boundary_vectors }
        # print(ego_state)
        return obs

    
    def done(self):
        # print("[DEBUG] done()")
        is_collision_with_agents = self.collisions.with_agents.view(self.world.batch_dim,-1).any(dim=-1) # [batch_dim]
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)
        # is_leaving_entry_segment = self.collisions.with_entry_segments.any(dim=-1) & (self.timer.step >= 20)
        is_any_agents_leaving_exit_segment = self.collisions.with_exit_segments.any(dim=-1)
        is_max_steps_reached = self.timer.step == (self.parameters.max_steps - 1)
        
        if self.parameters.training_strategy == "3": # Record into the initial state buffer
            if torch.rand(1) > (1 - self.initial_state_buffer.probability_record): # Only a certain probability to record
                for env_collide in torch.where(is_collision_with_agents)[0]:
                    self.initial_state_buffer.add(self.state_buffer.get_latest(n=n_steps_stored)[env_collide])
                    # print(colored(f"[LOG] Record states with path ids: {self.ref_paths_agent_related.path_id[env_collide]}.", "blue"))
        
        if self.parameters.is_testing_mode:
            is_done = is_max_steps_reached # In test mode, we only reset the whole env if the maximum time steps are reached
            
            # Reset single agent
            agents_reset = (
                self.collisions.with_agents.any(dim=-1) |
                self.collisions.with_lanelets |
                self.collisions.with_entry_segments |
                self.collisions.with_exit_segments
            )
            agents_reset_indices = torch.where(agents_reset)
            for env_idx, agent_idx in zip(agents_reset_indices[0], agents_reset_indices[1]):
                if not is_done[env_idx]:
                    self.reset_world_at(env_index=env_idx, agent_index=agent_idx)
        else:
            if self.parameters.training_strategy == "4":
                # is_done = is_max_steps_reached
                is_done = is_max_steps_reached | is_collision_with_agents | is_collision_with_lanelets
                
                # Reset single agnet
                agents_reset = (
                    # self.collisions.with_agents.any(dim=-1) |
                    # self.collisions.with_lanelets |
                    self.collisions.with_entry_segments |
                    self.collisions.with_exit_segments
                )
                agents_reset_indices = torch.where(agents_reset)
                for env_idx, agent_idx in zip(agents_reset_indices[0], agents_reset_indices[1]):
                    if not is_done[env_idx]:
                        self.reset_world_at(env_index=env_idx, agent_index=agent_idx)
                        # print(f"Reset agent {agent_idx} in env {env_idx}")
                
            else:
                is_done = is_max_steps_reached | is_collision_with_agents | is_collision_with_lanelets |  is_any_agents_leaving_exit_segment
                # assert(not is_leaving_entry_segment.any())
                # assert(not is_any_agents_leaving_exit_segment.any())

            assert not (is_collision_with_agents & (self.timer.step == 0)).any()
            assert not (is_collision_with_lanelets & (self.timer.step == 0)).any()
            # assert not (is_leaving_entry_segment & (self.timer.step == 0)).any()
            assert not (is_max_steps_reached & (self.timer.step == 0)).any()
            assert not (is_any_agents_leaving_exit_segment & (self.timer.step == 0)).any()
            
        # Logs
        # if is_collision_with_agents.any():
        #     print("Collide with other agents.")
        # if is_collision_with_lanelets.any():
        #     print("Collide with lanelet.")
        # if is_leaving_entry_segment.any():
        #     print("At least one agent is leaving its entry segment.")
        # if is_max_steps_reached.any():
        #     print("The number of the maximum steps is reached.")
        # if is_any_agents_leaving_exit_segment.any():
        #     print("At least one agent is leaving its exit segment.")            

        # return torch.tensor(False).unsqueeze(0).expand(self.world.batch_dim,-1)


        return is_done


    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """
        This function computes the info dict for "agent" in a vectorized way
        The returned dict should have a key for each info of interest and the corresponding value should
        be a tensor of shape (n_envs, info_size)

        Implementors can access the world at "self.world"

        To increase performance, tensors created should have the device set, like:
        torch.tensor(..., device=self.world.device)

        :param agent: Agent batch to compute info of
        :return: info: A dict with a key for each info of interest, and a tensor value  of shape (n_envs, info_size)
        """
        agent_index = self.world.agents.index(agent) # Index of the current agent

        is_action_empty = agent.action.u is None

        is_collision_with_agents = self.collisions.with_agents[:, agent_index].any(dim=-1) # [batch_dim]
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)

        info = {
            "pos": agent.state.pos / self.normalizers.pos_world,
            "rot": angle_eliminate_two_pi(agent.state.rot) / self.normalizers.rot,
            "vel": agent.state.vel / self.normalizers.v,
            "act_vel": (agent.action.u[:, 0] / self.normalizers.action_vel) if not is_action_empty else self.constants.empty_action_vel[:, agent_index],
            "act_steer": (agent.action.u[:, 1] / self.normalizers.action_steering) if not is_action_empty else self.constants.empty_action_steering[:, agent_index],
            "ref": (self.ref_paths_agent_related.short_term[:, agent_index] / self.normalizers.pos_world).reshape(self.world.batch_dim, -1),
            "distance_ref": self.distances.ref_paths[:, agent_index] / self.normalizers.distance_ref,
            "distance_left_b": self.distances.left_boundaries[:, agent_index].min(dim=-1)[0] / self.normalizers.distance_lanelet,
            "distance_right_b": self.distances.right_boundaries[:, agent_index].min(dim=-1)[0] / self.normalizers.distance_lanelet,
            "is_collision_with_agents": is_collision_with_agents,
            "is_collision_with_lanelets": is_collision_with_lanelets,
        }
        
        return info
    
    
    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        if self.parameters.is_real_time_rendering:
            if self.timer.step[0] == 0:
                pause_duration = 0 # Not sure how long should the simulation be paused at time step 0, so rather 0
            else:
                pause_duration = self.world.dt - (time.time() - self.timer.render_begin)
            if pause_duration > 0:
                time.sleep(pause_duration)
            # print(f"Paused for {pause_duration} sec.")
            
            self.timer.render_begin = time.time() # Update

        geoms = []
        
        # Visualize all lanelets

        # for i in range(len(self.map_data["lanelets"])):
        #     lanelet = self.map_data["lanelets"][i]
        # for lane in self.map_data1.lanes.values():   
        #     geom = rendering.PolyLine(
        #         v = list(lane.geometry.coords),
        #         close=False,
        #     )
        #     xform = rendering.Transform()
        #     geom.add_attr(xform)            
        #     geom.set_color(*Color.black100)
        #     geoms.append(geom)
            
        #     xform = rendering.Transform()
        #     geom.add_attr(xform)            
        #     geom.set_color(*Color.black100)
        #     geoms.append(geom)
        for lane in self.map_data.roadlines.values():
            way_type = lane.type_
            way_subtype = lane.subtype
            color, line_style,line_width = get_color_and_line_style(way_type, way_subtype)
            
            geom = rendering.PolyLine(
                v=list(lane.geometry.coords),
                close=False,
            )
            
            xform = rendering.Transform()
            geom.add_attr(xform)
            if color:
                geom.set_color(*color)
            if line_style:
                geom.add_attr(rendering.LineStyle(line_style))
            if line_width:
                geom.add_attr(rendering.LineWidth(line_width))
                # print(f"Setting line width to {line_width} for {way_type} with subtype {way_subtype}")  # Debugging print
            
            geoms.append(geom)
        
        if self.parameters.is_visualize_extra_info:
            hight_a = -0.10
            hight_b = -0.20
            hight_c = -0.30

            # Title
            geom = rendering.TextLine(
                text=self.parameters.render_title,
                x=25,  
                y=-15 + hight_a * 10,  
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)  
            geoms.append(geom)

            # Time and time step
            geom = rendering.TextLine(
                text=f"t: {self.timer.step[0]*self.parameters.dt:.2f} sec",
                x=25,
                y=-15 + hight_b * 10,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)  
            geoms.append(geom)

            geom = rendering.TextLine(
                text=f"n: {self.timer.step[0]}",
                x=25,
                y=-15 + hight_c * 10,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)  
            geoms.append(geom)

            
        for agent_i in range(self.n_agents):

            if self.parameters.is_visualize_short_term_path:
                geom = rendering.PolyLine(
                    v = self.ref_paths_agent_related.short_term[env_index, agent_i],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)            
                geom.set_color(*colors[agent_i])
                geoms.append(geom)
                
                for i_p in self.ref_paths_agent_related.short_term[env_index, agent_i]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(
                        i_p[0], 
                        i_p[1]
                    )
                    circle.set_color(*colors[agent_i])
                    geoms.append(circle)
            
            # Visualize nearing points on boundaries
            if not self.parameters.is_observe_distance_to_boundaries:
                # Left boundary
                geom = rendering.PolyLine(
                    v = self.ref_paths_agent_related.nearing_points_left_boundary[env_index, agent_i],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)            
                geom.set_color(*colors[agent_i])
                geoms.append(geom)
                
                for i_p in self.ref_paths_agent_related.nearing_points_left_boundary[env_index, agent_i]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(
                        i_p[0], 
                        i_p[1]
                    )
                    circle.set_color(*colors[agent_i])
                    geoms.append(circle)
                    
                # Right boundary
                geom = rendering.PolyLine(
                    v = self.ref_paths_agent_related.nearing_points_right_boundary[env_index, agent_i],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)            
                geom.set_color(*colors[agent_i])
                geoms.append(geom)
                
                for i_p in self.ref_paths_agent_related.nearing_points_right_boundary[env_index, agent_i]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(
                        i_p[0], 
                        i_p[1]
                    )
                    circle.set_color(*colors[agent_i])
                    geoms.append(circle)
            
            # Agent IDs
            geom = rendering.TextLine(
                text=f"{agent_i}",
                x=(self.world.agents[agent_i].state.pos[env_index, 0] / self.world.x_semidim) * self.viewer_size[0],
                y=(self.world.agents[agent_i].state.pos[env_index, 1] / self.world.y_semidim) * self.viewer_size[1],
                # x=(self.world.agents[agent_i].state.pos[env_index, 0] - self.render_origin[0] + self.world.x_semidim / 2) * resolution_factor / self.viewer_zoom,
                # y=(self.world.agents[agent_i].state.pos[env_index, 1] - self.render_origin[1] + self.world.y_semidim / 2) * resolution_factor / self.viewer_zoom,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)  
            geoms.append(geom)
                
            # Lanelet boundaries of agents' reference path
            if self.parameters.is_visualize_lane_boundary:
                if agent_i == 0:
                    # Left boundary
                    geom = rendering.PolyLine(
                        v = self.ref_paths_agent_related.nearing_points_left_boundary[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)            
                    geom.set_color(*colors[agent_i])
                    geoms.append(geom)
                    # Right boundary
                    geom = rendering.PolyLine(
                        v = self.ref_paths_agent_related.nearing_points_right_boundary[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)            
                    geom.set_color(*colors[agent_i])
                    geoms.append(geom)
                    # Entry
                    # geom = rendering.PolyLine(
                    #     v = self.ref_paths_agent_related.entry[env_index, agent_i],
                    #     close=False,
                    # )
                    # xform = rendering.Transform()
                    # geom.add_attr(xform)
                    # geom.set_color(*colors[agent_i])
                    # geoms.append(geom)
                    # Exit
                    geom = rendering.PolyLine(
                        v = self.ref_paths_agent_related.exit[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*colors[agent_i])
                    geoms.append(geom)
            
        return geoms


if __name__ == "__main__":
    scenario = ScenarioRoadTraffic()
    render_interactively(
        scenario=scenario, control_two_agents=False, shared_reward=False,
    )
    