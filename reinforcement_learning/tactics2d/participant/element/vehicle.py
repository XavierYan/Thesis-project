##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: vehicle.py
# @Description: This file defines a class for a four-wheeled vehicle.
# @Author: Yueyuan Li
# @Version: 1.0.0

import logging
from typing import Any, List, Tuple

import lanelet2
import numpy as np
from shapely.affinity import affine_transform, translate
from shapely.geometry import LinearRing, Point
from shapely.geometry import LineString, Polygon

import torch

from tactics2d.participant.trajectory import State, Trajectory
from tactics2d.physics import SingleTrackKinematics
from utils_map import *

from .participant_base import ParticipantBase
from .participant_template import EPA_MAPPING, EURO_SEGMENT_MAPPING, NCAP_MAPPING, VEHICLE_TEMPLATE
from lanelet2.core import BasicPoint3d
import numpy as np


class Vehicle(ParticipantBase):
    r"""This class defines a four-wheeled vehicle with its common properties.

    The definition of different driven modes of the vehicles can be found here:

    - [Front-Wheel Drive (FWD)](https://en.wikipedia.org/wiki/Front-wheel_drive)
    - [Rear-Wheel Drive (RWD)](https://en.wikipedia.org/wiki/Rear-wheel_drive)
    - [Four-Wheel Drive (4WD)](https://en.wikipedia.org/wiki/Four-wheel_drive)
    - [All-Wheel Drive (AWD)](https://en.wikipedia.org/wiki/All-wheel_drive)

    The default physics model provided for the vehicles is a kinematics bicycle model with the vehicle's geometry center as the origin. This model is recommended only for the vehicles with front-wheel drive (FWD). For other driven modes, the users should better custom their own physics model.

    !!! info "TODO"
        More physics models will be added in the future based on issues and requirements. You are welcome to suggest implementation of other physics models [here](https://github.com/WoodOxen/tactics2d/issues).

    Attributes:
        id_ (int): The unique identifier of the vehicle.
        type_ (str): The type of the vehicle. Defaults to "medium_car".
        trajectory (Trajectory): The trajectory of the vehicle. Defaults to an empty trajectory.
        color (tuple): The color of the vehicle. The color of the traffic participant. This attribute will be left to the sensor module to verify and convert to the appropriate type. You can refer to [Matplotlib's way](https://matplotlib.org/stable/users/explain/colors/colors.html) to specify validate colors. Defaults to light-turquoise (43, 203, 186).
        length (float): The length of the vehicle. The unit is meter. Defaults to None.
        width (float): The width of the vehicle. The unit is meter. Defaults to None.
        height (float): The height of the vehicle. The unit is meter. Defaults to None.
        kerb_weight: (float): The weight of the vehicle. The unit is kilogram (kg). Defaults to None.
        wheel_base (float): The wheel base of the vehicle. The unit is meter. Defaults to None.
        front_overhang (float): The front overhang of the vehicle. The unit is meter. Defaults to None.
        rear_overhang (float): The rear overhang of the vehicle. The unit is meter. Defaults to None.
        driven_mode: (str): The driven way of the vehicle. The available options are ["FWD", "RWD", "4WD", "AWD"]. Defaults to "FWD".
        max_steer (float): The maximum approach angle of the vehicle. The unit is radian. Defaults to $\pi$/6.
        max_speed (float): The maximum speed of the vehicle. The unit is meter per second. Defaults to 55.56 (= 200 km/h).
        max_accel (float): The maximum acceleration of the vehicle. The unit is meter per second squared. Defaults to 3.0.
        max_decel (float): The maximum deceleration of the vehicle. The unit is meter per second squared. Defaults to 10.
        steer_range (Tuple[float, float]): The range of the vehicle steering angle. The unit is radian. Defaults to (-$\pi$/6, $\pi$/6).
        speed_range (Tuple[float, float]): The range of the vehicle speed. The unit is meter per second (m/s). Defaults to (-16.67, 55.56) (= -60~200 km/h).
        accel_range (Tuple[float, float]): The range of the vehicle acceleration. The unit is meter per second squared. Defaults to (-10, 3).
        verify (bool): Whether to verify the trajectory to bind or the state to add. Defaults to False.
        physics_model (PhysicsModelBase): The physics model of the cyclist. Defaults to SingleTrackKinematics.
        shape (float): The shape of the cyclist. It is represented as a bounding box with its original point located at the mass center. This attribute is **read-only**.
        current_state (State): The current state of the traffic participant. This attribute is **read-only**.
    """

    __annotations__ = {
        "type_": str,
        "length": float,
        "width": float,
        "height": float,
        "kerb_weight": float,
        "wheel_base": float,
        "front_overhang": float,
        "rear_overhang": float,
        "driven_mode": str,
        "max_steer": float,
        "max_speed": float,
        "max_accel": float,
        "max_decel": float,
        "verify": bool,
    }
    _default_color = (43, 203, 186, 255)  # light-turquoise
    _driven_modes = {"FWD", "RWD", "4WD", "AWD"}

    def __init__(
        self, id_: Any, type_: str = "medium_car", trajectory: Trajectory = None, start_x = None, start_y = None,end_x = None, end_y = None,ll_map = None,length = 4,width = 3,**kwargs
    ):

        super().__init__(id_, type_, trajectory, **kwargs)

        self.max_steer = np.round(np.pi / 6, 3) if self.max_steer is None else self.max_steer
        self.max_speed = 55.56 if self.max_speed is None else self.max_speed
        self.max_accel = 3.0 if self.max_accel is None else self.max_accel
        self.max_decel = 10.0 if self.max_decel is None else self.max_decel

        self.speed_range = (-16.67, self.max_speed)
        self.steer_range = (-self.max_steer, self.max_steer)
        self.accel_range = (-self.max_accel, self.max_accel)
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.length = length
        self.width = width
        self.trajectory = trajectory


        # 基于起始和终点位置寻找最近的车道线ID列表
        self.route_lanelet_ids = self._find_route_lanelet_ids(ll_map)
        self.reference_trajectory = self._get_points_from_lanelets(ll_map, self.route_lanelet_ids)
        self.left_boundary, self.right_boundary = self._get_boundaries_from_lanelets(ll_map, self.route_lanelet_ids)


        if self.driven_mode is None:
            self.driven_mode = "FWD"


        self._auto_construct_physics_model()

        self._bbox = LinearRing(
            [
                [0.5 * self.length, -0.5 * self.width],
                [0.5 * self.length, 0.5 * self.width],
                [-0.5 * self.length, 0.5 * self.width],
                [-0.5 * self.length, -0.5 * self.width],
            ]
        )
        self.box = self.get_pose()

    def _get_points_from_lanelets(self, ll_map, lanelet_ids):
        points = []
        for lanelet_id in lanelet_ids:
            lanelet = ll_map.laneletLayer[lanelet_id]  # Fetch the lanelet by ID from the map
            for point in lanelet.centerline:  # Assuming each lanelet has a 'centerline' attribute
                points.append((point.x, point.y))  # Append the (x, y) coordinates to the points list
        return points
    
    def _get_boundaries_from_lanelets(self, ll_map, lanelet_ids):
        left_boundary_points = []
        right_boundary_points = []
        for lanelet_id in lanelet_ids:
            lanelet = ll_map.laneletLayer[lanelet_id]
            left_boundary = lanelet.leftBound
            right_boundary = lanelet.rightBound

            left_boundary_points.extend([(point.x, point.y) for point in left_boundary])
            right_boundary_points.extend([(point.x, point.y) for point in right_boundary])

        return left_boundary_points, right_boundary_points

    def _find_route_lanelet_ids(self,ll_map) -> List[int]:
        trafficRules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)
        routingGraph = lanelet2.routing.RoutingGraph(ll_map, trafficRules)

        start_point = BasicPoint3d(self.start_x, self.start_y, 0)
        end_point = BasicPoint3d(self.end_x, self.end_y, 0)

        start_lanelets = find_nearest_lanelets(start_point,ll_map, 10)
        end_lanelets = find_nearest_lanelets(end_point, ll_map,10)

        route_lanelet_ids = []
        for start_lanelet in start_lanelets:
            for end_lanelet in end_lanelets:
                route = routingGraph.getRoute(start_lanelet, end_lanelet)
                if route:
                    # 获取路由的所有lanelet IDs
                    route_lanelet_ids.extend([lanelet.id for lanelet in route.shortestPath()])
                    return route_lanelet_ids  # 找到一条路由后即返回
        return route_lanelet_ids
    
    @property
    def geometry(self) -> LinearRing:
        return self._bbox

    def _auto_construct_physics_model(self):
        self.physics_model = SingleTrackKinematics(
            lf=self.length / 2,
            lr=self.length / 2,
            steer_rate_range=self.steer_range,
            speed_range=self.speed_range,
            accel_range=self.accel_range,
        )
        # Auto-construct the physics model for front-wheel-drive (FWD) vehicles.
        # if self.driven_mode == "FWD":
        #     if not None in [self.front_overhang, self.rear_overhang]:
        #         self.physics_model = SingleTrackKinematics(
        #             lf=self.length / 2 - self.front_overhang,
        #             lr=self.length / 2 - self.rear_overhang,
        #             steer_range=self.steer_range,
        #             speed_range=self.speed_range,
        #             accel_range=self.accel_range,
        #         )
        #     elif not self.length is None:
        #         self.physics_model = SingleTrackKinematics(
        #             lf=self.length / 2,
        #             lr=self.length / 2,
        #             steer_range=self.steer_range,
        #             speed_range=self.speed_range,
        #             accel_range=self.accel_range,
        #         )
        #     else:
        #         self.verify = False
        #         logging.info(
        #             "Cannot construct a physics model for the vehicle. The state verification is turned off."
        #         )
        # # Auto-construct the physics model for rear-wheel-drive (RWD) vehicles.
        # elif self.driven_mode == "RWD":
        #     pass
        # # Auto-construct the physics model for all-wheel-drive (AWD) vehicles.
        # else:
        #     pass

    def load_from_template(self, type_name: str, overwrite: bool = True, template: dict = None):
        """Load the vehicle properties from the template.

        Args:
            type_name (str): The type of the vehicle.
            overwrite (bool, optional): Whether to overwrite the existing properties. Defaults to False.
            template (dict, optional): The template of the vehicle. Defaults to VEHICLE_TEMPLATE.
        """
        if template is None:
            template = VEHICLE_TEMPLATE

        if type_name in EURO_SEGMENT_MAPPING:
            type_name = EURO_SEGMENT_MAPPING[type_name]
        elif type_name in EPA_MAPPING:
            type_name = EPA_MAPPING[type_name]
        elif type_name in NCAP_MAPPING:
            type_name = NCAP_MAPPING[type_name]

        if type_name in template:
            for key, value in template[type_name].items():
                if key == "0_100_km/h":
                    if overwrite or self.max_accel is None:
                        self.max_accel = np.round(100 * 1000 / 3600 / template[type_name][key], 3)
                else:
                    if overwrite or getattr(self, key) is None:
                        setattr(self, key, value)
        else:
            logging.warning(
                f"{type_name} is not in the vehicle template. The default values will be used."
            )

        self.speed_range = (-16.67, self.max_speed)
        self.accel_range = (-self.max_decel, self.max_accel)

        if not None in [self.length, self.width]:
            self._bbox = LinearRing(
                [
                    [0.5 * self.length, -0.5 * self.width],
                    [0.5 * self.length, 0.5 * self.width],
                    [-0.5 * self.length, 0.5 * self.width],
                    [-0.5 * self.length, -0.5 * self.width],
                ]
            )

    def add_state(self, state: State):
        """This function adds a state to the vehicle.

        Args:
            state (State): The state to add.
        """
        if not self.verify or self.physics_model is None:
            self.trajectory.add_state(state)
        elif self.physics_model.verify_state(state, self.trajectory.current_state):
            self.trajectory.append_state(state)
        else:
            raise RuntimeError(
                "Invalid state checked by the physics model %s."
                % (self.physics_model.__class__.__name__)
            )

    # def bind_trajectory(self, trajectory: Trajectory):
    #     """This function binds a trajectory to the vehicle.

    #     Args:
    #         trajectory (Trajectory): The trajectory to bind.

    #     Raises:
    #         TypeError: If the input trajectory is not of type [`Trajectory`](#tactics2d.participant.trajectory.Trajectory).
    #     """
    #     if not isinstance(trajectory, Trajectory):
    #         raise TypeError("The trajectory must be an instance of Trajectory.")

    #     if self.verify:
    #         if not self._verify_trajectory(trajectory):
    #             self.trajectory = Trajectory(self.id_)
    #             logging.warning(
    #                 f"The trajectory is invalid. Vehicle {self.id_} is not bound to the trajectory."
    #             )
    #         else:
    #             self.trajectory = trajectory
    #     else:
    #         self.trajectory = trajectory
    #         logging.debug(f"Vehicle {self.id_} is bound to a trajectory without verification.")

    # def get_pose(self, frame: int = None) -> LinearRing:
    #     """This function gets the pose of the vehicle at the requested frame.

    #     Args:
    #         frame (int, optional): The frame to get the vehicle's pose.

    #     Returns:
    #         pose (LinearRing): The vehicle's bounding box which is rotated and moved based on the current state.
    #     """
    #     state = self.trajectory.get_state(frame)
    #     transform_matrix = [
    #         np.cos(state.heading),
    #         -np.sin(state.heading),
    #         np.sin(state.heading),
    #         np.cos(state.heading),
    #         state.location[0],
    #         state.location[1],
    #     ]
    #     return affine_transform(self._bbox, transform_matrix)

    def get_pose(self, frame: int = None) -> LinearRing:
            """This function gets the pose of the vehicle at the requested frame.

            Args:
                frame (int, optional): The frame to get the vehicle's pose.

            Returns:
                pose (LinearRing): The vehicle's bounding box which is rotated and moved based on the current state.
            """
            state = self.trajectory.get_state(frame)
            transform_matrix = [
                np.cos(state.heading),
                -np.sin(state.heading),
                np.sin(state.heading),
                np.cos(state.heading),
                state.location[0],
                state.location[1],
            ]
            return affine_transform(self._bbox, transform_matrix)
    
    def get_trace(self, frame_range: Tuple[int, int] = None) -> LinearRing:
        """This function gets the trace of the vehicle within the requested frame range.

        Args:
            frame_range (Tuple[int, int], optional): The requested frame range. The first element is the start frame, and the second element is the end frame. The unit is millisecond (ms).

        Returns:
            The trace of the cyclist within the requested frame range.
        """
        # states = self.get_states(frame_range)
        # trace = None
        # if len(states) == 0:
        #     pass
        # elif len(states) == 1:
        #     trace = self.get_pose(frame=states[0].frame)
        # else:
        #     center_line = []
        #     start_pose = np.array(list(self.get_pose(frame=states[0].frame).coords))
        #     end_pose = np.array(list(self.get_pose(frame=states[-1].frame).coords))
        #     start_point = tuple(np.mean(start_pose[2:4], axis=0))  # the midpoint of the rear
        #     end_point = tuple(np.mean(end_pose[0:2], axis=0))  # the midpoint of the front
        #     center_line.append(start_point)
        #     for state in states:
        #         trajectory.append(state.location)
        #     center_line.append(end_point)
        #     trajectory = LineString(trajectory)

        #     left_bound = trajectory.offset_curve(self.width / 2)
        #     right_bound = trajectory.offset_curve(-self.width / 2)

        #     trace = LinearRing(list(left_bound.coords) + list(reversed(list(right_bound.coords))))

        # return trace
        return None

    def _get_history_vector(self, num_frames: int = 1):


        last_state = self.trajectory.last_state

        # 从最新状态中获取当前位置
        current_x, current_y = last_state.location

        current_heading = last_state.heading
        # 确保请求的帧数不为负
        num_frames = max(num_frames, 0)

        # 获取当前帧索引
        current_frame_index = last_state.frame

        # 检索历史帧
        history_frames = self.trajectory.frames[max(0, current_frame_index - num_frames):current_frame_index + 1]

        # 检索历史帧对应的状态
        history_states = [self.trajectory.history_states[frame] for frame in history_frames]

        # history_states.reverse()

        # 转换历史数据为坐标向量
        history_vectors = []
        if len(history_states) > 1:
            for i in range(1, len(history_states)):
                # 计算时间戳差
                time_stamp =  history_states[i].frame - last_state.frame

                # 构造向量
                vector = [
                    history_states[i-1].x, history_states[i-1].y,
                    history_states[i].x, history_states[i].y, 
                    current_heading,  # 预留字段
                    history_states[i].speed, self.length, self.width,  # 速度信息
                    time_stamp  # 时间戳差
                ]
                history_vectors.append(vector)
        else:
            # 只有一帧时，使用当前状态的坐标作为起点和终点
            if len(history_states) == 1:
                vector = [
                    history_states[0].x, history_states[0].y,
                    history_states[0].x, history_states[0].y,
                    current_heading,
                    history_states[0].speed, self.length, self.width,
                    0  # 时间戳差为0
                ]
                history_vectors.append(vector)

        # 如果历史向量不足，用零向量填充
        if len(history_vectors) < num_frames:
            zero_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            history_vectors += [zero_vector] * (num_frames - len(history_vectors))
        
        # history_vectors = torch.tensor(history_vectors)


        normalized_history_trajectory = normalize_and_adjust_trajectory(history_vectors, current_x, current_y, current_heading)

        return normalized_history_trajectory


        
    def _get_lane_vectors(self,ll_map):
        # 获取最新的状态
        last_state = self.trajectory.last_state

        # 从最新状态中获取当前位置
        current_x, current_y = last_state.location

        current_heading = last_state.heading

        # 将当前位置转换为 BasicPoint3d
        current_point = BasicPoint3d(current_x, current_y, 0)

        # 收集左右边界的点
        left_lane_vector = collect_boundary_points(ll_map, self.route_lanelet_ids, 'left', current_point,num_vectors=20)
        right_lane_vector = collect_boundary_points(ll_map, self.route_lanelet_ids, 'right', current_point,num_vectors=20)

        # 对左右边界点进行插值处理
        left_lane_vector = interpolate_vectors(left_lane_vector,interval=1, max_vectors=15)
        right_lane_vector = interpolate_vectors(right_lane_vector,interval=1, max_vectors=15)

        # 将插值后的向量转换为张量
        left_lane_vector = torch.tensor(left_lane_vector, dtype=torch.float)
        right_lane_vector = torch.tensor(right_lane_vector, dtype=torch.float)

        # 将左右边界向量堆叠成一个张量
        lane_boundaries = torch.stack((left_lane_vector, right_lane_vector), dim=0)

        normalized_lane_boundaries = torch.stack([normalize_trajectory(boundary, current_x, current_y, current_heading) for boundary in lane_boundaries])

        return normalized_lane_boundaries
    

    def _get_reference_vectors(self,ll_map ):
        # 获取最新的状态
        last_state = self.trajectory.last_state

        # 从最新状态中获取当前位置
        current_x, current_y = last_state.location

        current_heading = last_state.heading

        # 将当前位置转换为 BasicPoint3d
        current_point = BasicPoint3d(current_x, current_y, 0)

        # 收集中心线的点
        reference_trajectory = collect_centerlines_points(ll_map, self.route_lanelet_ids, current_point, num_vectors=16)

        # 对中心线点进行插值处理
        reference_trajectory = interpolate_reference_vectors(reference_trajectory, interval=1, max_vectors=15)

        # 将插值后的参考轨迹转换为张量
        reference_trajectory = torch.tensor(reference_trajectory, dtype=torch.float)

        normalized_reference_trajectory = normalize_trajectory(reference_trajectory, current_x, current_y, current_heading)

        return normalized_reference_trajectory
    

    def _get_global_history_vector(self, num_frames: int = 1):


        last_state = self.trajectory.last_state

        # 从最新状态中获取当前位置
        current_x, current_y = last_state.location

        current_heading = last_state.heading
        # 确保请求的帧数不为负
        num_frames = max(num_frames, 0)

        # 获取当前帧索引
        current_frame_index = last_state.frame

        # 检索历史帧
        history_frames = self.trajectory.frames[max(0, current_frame_index - num_frames):current_frame_index + 1]

        # 检索历史帧对应的状态
        history_states = [self.trajectory.history_states[frame] for frame in history_frames]

        # history_states.reverse()

        # 转换历史数据为坐标向量
        history_vectors = []
        if len(history_states) > 1:
            for i in range(1, len(history_states)):
                # 计算时间戳差
                time_stamp =  history_states[i].frame - last_state.frame

                # 构造向量
                vector = [
                    history_states[i-1].x, history_states[i-1].y,
                    history_states[i].x, history_states[i].y, 
                    current_heading,  # 预留字段
                    history_states[i].speed, self.length, self.width,  # 速度信息
                    time_stamp  # 时间戳差
                ]
                history_vectors.append(vector)
        else:
            # 只有一帧时，使用当前状态的坐标作为起点和终点
            if len(history_states) == 1:
                vector = [
                    history_states[0].x, history_states[0].y,
                    history_states[0].x, history_states[0].y,
                    current_heading,
                    history_states[0].speed, self.length, self.width,
                    0  # 时间戳差为0
                ]
                history_vectors.append(vector)

        # 如果历史向量不足，用零向量填充
        if len(history_vectors) < num_frames:
            zero_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            history_vectors += [zero_vector] * (num_frames - len(history_vectors))
        
        global_history_trajectory = torch.tensor(history_vectors)

        return global_history_trajectory


        
    def _get_global_lane_vectors(self,ll_map):
        # 获取最新的状态
        last_state = self.trajectory.last_state

        # 从最新状态中获取当前位置
        current_x, current_y = last_state.location

        current_heading = last_state.heading

        # 将当前位置转换为 BasicPoint3d
        current_point = BasicPoint3d(current_x, current_y, 0)

        # 收集左右边界的点
        left_lane_vector = collect_boundary_points(ll_map, self.route_lanelet_ids, 'left', current_point,num_vectors=20)
        right_lane_vector = collect_boundary_points(ll_map, self.route_lanelet_ids, 'right', current_point,num_vectors=20)

        # 对左右边界点进行插值处理
        left_lane_vector = interpolate_vectors(left_lane_vector,interval=1, max_vectors=15)
        right_lane_vector = interpolate_vectors(right_lane_vector,interval=1, max_vectors=15)

        # 将插值后的向量转换为张量
        left_lane_vector = torch.tensor(left_lane_vector, dtype=torch.float)
        right_lane_vector = torch.tensor(right_lane_vector, dtype=torch.float)

        # 将左右边界向量堆叠成一个张量
        lane_boundaries = torch.stack((left_lane_vector, right_lane_vector), dim=0)

        return lane_boundaries
    

    def _get_global_reference_vectors(self,ll_map ):
        # 获取最新的状态
        last_state = self.trajectory.last_state

        # 从最新状态中获取当前位置
        current_x, current_y = last_state.location

        current_heading = last_state.heading

        # 将当前位置转换为 BasicPoint3d
        current_point = BasicPoint3d(current_x, current_y, 0)

        # 收集中心线的点
        reference_trajectory = collect_centerlines_points(ll_map, self.route_lanelet_ids, current_point, num_vectors=16)

        # 对中心线点进行插值处理
        reference_trajectory = interpolate_reference_vectors(reference_trajectory, interval=1, max_vectors=15)

        # 将插值后的参考轨迹转换为张量
        reference_trajectory = torch.tensor(reference_trajectory, dtype=torch.float)

        # normalized_reference_trajectory = normalize_trajectory(reference_trajectory, current_x, current_y, current_heading)

        return reference_trajectory
    
    def get_single_env(self,ll_map):

       return self._get_history_vector().unsqueeze(0).float(), self._get_reference_vectors(ll_map).unsqueeze(0),torch.zeros((1,2,5,9)),torch.zeros((1,2,15,5)),self._get_lane_vectors(ll_map).unsqueeze(0)
    
    def get_global_env(self,ll_map):

       return self._get_global_history_vector().float(), self._get_global_reference_vectors(ll_map),self._get_global_lane_vectors(ll_map)
    
    def ensure_min_num_trajectories(self, trajectories, min_trajectories=4, vectors_per_trajectory = 49, vector_dim=9):

        padded_trajectories = trajectories.copy()
        
        # 检查轨迹数量是否少于min_trajectories
        num_missing_trajectories = min_trajectories - len(padded_trajectories)
        while num_missing_trajectories > 0:
            # 生成全零轨迹补足至min_trajectories条
            zero_trajectory = [[0] * vector_dim for _ in range(vectors_per_trajectory)]
            padded_trajectories.append(zero_trajectory)
            num_missing_trajectories = num_missing_trajectories -1
        
        return padded_trajectories
    
    def get_env(self, ll_map, neighbors):

        last_state = self.trajectory.last_state

        # 从最新状态中获取当前位置
        current_x, current_y = last_state.location

        current_heading = last_state.heading
        # 获取车辆自身的状态
        self_state = self._get_history_vector().unsqueeze(0).float()
        reference_traj = self._get_reference_vectors(ll_map).unsqueeze(0)
        lane_boundaries = self._get_lane_vectors(ll_map).unsqueeze(0)

        # 收集并处理周围车辆的状态
        nearby_trajectories = [n._get_global_history_vector().float() for n in neighbors.values()]
        nearby_reference_trajectories = [n._get_global_reference_vectors(ll_map) for n in neighbors.values()]

        # 确保至少有两个邻近车辆的信息，如果不足则用零向量填充
        nearby_trajectories = self.ensure_min_num_trajectories(nearby_trajectories, min_trajectories=2, vectors_per_trajectory=1, vector_dim=9)
        nearby_reference_trajectories = self.ensure_min_num_trajectories(nearby_reference_trajectories, min_trajectories=2, vectors_per_trajectory=15, vector_dim=5)
        
        normalized_nearby_trajectories = torch.stack([normalize_and_adjust_trajectory(trajectory, current_x, current_y, current_heading) for trajectory in nearby_trajectories])
        normalized_nearby_reference_trajectories = torch.stack([normalize_trajectory(reference_trajectory,current_x, current_y, current_heading) for reference_trajectory in nearby_reference_trajectories])
        
        return (self_state, reference_traj, normalized_nearby_trajectories.float().unsqueeze(0), normalized_nearby_reference_trajectories.float().unsqueeze(0), lane_boundaries)
        # return (self_state, reference_traj, torch.zeros((1,2,1,9)),torch.zeros((1,2,15,5)), lane_boundaries)

    
    
    def get_ctr_sq(self, model,ll_map):
        observation = self.get_single_env(ll_map)
        action = model(*observation)
        return action
    
    def get_ctr(self, model,ll_map):
        observation = self.get_single_env(ll_map)
        action = model(*observation)
        return action[0][0], action[0][1]

    

    def get_traj(self, model, ll_map):
        observation = self.get_single_env(ll_map)
        with torch.no_grad():
            action = model(*observation)
        trajectory = predict_trajectory(self.trajectory.last_state.x, self.trajectory.last_state.y, self.trajectory.last_state.heading, self.trajectory.last_state.speed, self.trajectory.last_state.steering_angle, action.squeeze(),self.length)
        return trajectory

        
