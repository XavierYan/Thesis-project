
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import lanelet2
from lanelet2.core import LaneletMap, GPSPoint ,BasicPoint3d, BasicPoint2d
from lanelet2.projection import UtmProjector
# from route_planning_utils import *
from lanelet2.routing import RoutingGraph
# from utils import dataset_utils

import numpy as np
import torch


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import lanelet2
from lanelet2.core import LaneletMap, GPSPoint ,BasicPoint3d, BasicPoint2d
from lanelet2.projection import UtmProjector
from lanelet2.routing import RoutingGraph


import numpy as np
import torch
from scipy.interpolate import interp1d


# calculate rotation matrix
def rotate_and_translate(points, origin, heading):

    rotation_matrix = torch.tensor([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])
    

    translated_points = points - origin
    

    transformed_points = torch.matmul(translated_points, rotation_matrix)
    
    return transformed_points

#transfer points to vectors
def trajectory_to_vectors(trajectory_points):
    device = trajectory_points.device
    if trajectory_points.ndim == 3:
        # [batch_size, n_agents, n_points_short_term, 2]
        n_agents, n_points_short_term, _ = trajectory_points.shape
        points_current = trajectory_points[ :, :-1, :] 
        points_next = trajectory_points[ :, 1:, :]  

        vectors = torch.cat([points_current, points_next], dim=-1)

        ids = torch.arange(1, n_points_short_term, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand(n_agents, -1, -1)

    elif trajectory_points.ndim == 2:
        # [batch_size, n_points_short_term, 2]
        n_points_short_term, _= trajectory_points.shape
        points_current = trajectory_points[:-1, :]  
        points_next = trajectory_points[1:, :]  

        vectors = torch.cat([points_current, points_next], dim=-1)


        ids = torch.arange(1, n_points_short_term, device=device, dtype=torch.float32).unsqueeze(-1)

    else:
        raise ValueError("Unsupported input shape: {}".format(trajectory_points.shape))

    vectors_with_ids = torch.cat([vectors, ids.float()], dim=-1)

    return vectors_with_ids


class VehicleTrajectoryDataset(Dataset):
    def __init__(self, tracks_folder, map_path, origin_lat = 50.890926, origin_lon = 6.173557, history_frames=25, future_frames=10, nearby_distance=13, max_neighbors=4,num_points_to_extract = 15):
            self.history_frames = history_frames
            self.future_frames = future_frames
            self.nearby_distance = nearby_distance
            self.max_neighbors = max_neighbors
            self.tracks_folder = tracks_folder
            self.map_path = map_path
            self.ll_map, self.projector = self._load_lanelet_map(origin_lat, origin_lon)
            self.df= self._process_data()
            self.trafficRules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)
            self.routingGraph = lanelet2.routing.RoutingGraph(self.ll_map, self.trafficRules) 
            self.num_points_to_extract  = num_points_to_extract

    def get_points_from_lanelets(self, ll_map, lanelet_ids):
        points = []
        for lanelet_id in lanelet_ids:
            lanelet = ll_map.laneletLayer[lanelet_id]  # Fetch the lanelet by ID from the map
            for point in lanelet.centerline:  # Assuming each lanelet has a 'centerline' attribute
                points.append((point.x, point.y))  # Append the (x, y) coordinates to the points list
        return points
    
    def get_boundaries_from_lanelets(self, ll_map, lanelet_ids):
        left_boundary_points = []
        right_boundary_points = []
        for lanelet_id in lanelet_ids:
            lanelet = ll_map.laneletLayer[lanelet_id]
            left_boundary = lanelet.leftBound
            right_boundary = lanelet.rightBound

            left_boundary_points.extend([(point.x, point.y) for point in left_boundary])
            right_boundary_points.extend([(point.x, point.y) for point in right_boundary])

        return left_boundary_points, right_boundary_points
    
    def _load_lanelet_map(self, origin_lat, origin_lon):
        ll_origin = lanelet2.io.Origin(origin_lat, origin_lon)
        projector = UtmProjector(ll_origin)
        self.ll_map = lanelet2.io.load(self.map_path, projector)
        return self.ll_map, projector
    
    def __len__(self):
        return len(self.df)

    def _process_data(self):
        frame_rate = 25
        delta_t = 1 / frame_rate

        all_tracks = []
        for filename in os.listdir(self.tracks_folder):
            if filename.endswith('csv'):
                file_path = os.path.join(self.tracks_folder, filename)
                tracks_df = pd.read_csv(file_path)
                all_tracks.append(tracks_df)


        processed_tracks_df = pd.concat(all_tracks, ignore_index=True)

        # transfer angle to radius
        processed_tracks_df['heading'] = processed_tracks_df['heading'] * np.pi / 180

        # calculate yaw rate
        processed_tracks_df['delta_heading'] = processed_tracks_df.groupby(['recordingId', 'trackId'])['heading'].shift(-1) - processed_tracks_df['heading']
        processed_tracks_df['delta_heading'] = (processed_tracks_df['delta_heading'].fillna(method='ffill') + np.pi)%(2*np.pi) - np.pi
        processed_tracks_df['yaw_rate'] = processed_tracks_df['delta_heading'] / delta_t

        # calculate vel and acc
        processed_tracks_df['velocity'] = np.sqrt(processed_tracks_df['xVelocity']**2 + processed_tracks_df['yVelocity']**2)
        processed_tracks_df['acceleration'] = np.sqrt(processed_tracks_df['xAcceleration']**2 + processed_tracks_df['yAcceleration']**2)

        # calculate slip angle
        processed_tracks_df['slip_angle'] = np.arctan2(processed_tracks_df['latVelocity'], processed_tracks_df['lonVelocity'])

        processed_tracks_df['l_r'] = 0.5 * processed_tracks_df['length']

        # calculate steering angle
        # processed_tracks_df['tantheta'] = (processed_tracks_df['yaw_rate'] * processed_tracks_df['wheel_base']) / (processed_tracks_df['velocity']) / np.cos(processed_tracks_df['slip_angle'])
        # processed_tracks_df['steering_angle'] = np.arctan(processed_tracks_df['tantheta'])
        # processed_tracks_df['steering_angle'] = np.arctan(processed_tracks_df['yaw_rate'] * processed_tracks_df['wheel_base']) / (processed_tracks_df['velocity'])
        processed_tracks_df['steering_angle'] = np.where(
            processed_tracks_df['velocity'] != 0,
            np.arctan(processed_tracks_df['yaw_rate'] * processed_tracks_df['length'] / processed_tracks_df['velocity']),
            0.0
        )


        # calculate steering rate
        processed_tracks_df['steering_angle_rate'] = (processed_tracks_df.groupby(['recordingId', 'trackId'])['steering_angle'].shift(-1) - processed_tracks_df['steering_angle'])*25
        processed_tracks_df['steering_angle_rate'] = processed_tracks_df['steering_angle_rate'].fillna(method='ffill')
        processed_tracks_df['steering_angle'] = processed_tracks_df['steering_angle']*180/np.pi

        filtered_df = processed_tracks_df.copy()
        filtered_df = filtered_df[~filtered_df['class'].isin(['pedestrian','motorcycle','bicycle','truck','trailer'])]
        filtered_df = filtered_df.reset_index(drop=True)

        return filtered_df
    
    #ensure the fixed shape
    def ensure_min_num_trajectories(self, trajectories, min_trajectories=4, vectors_per_trajectory = 49, vector_dim=9):
        padded_trajectories = trajectories.copy()
        
        num_missing_trajectories = min_trajectories - len(padded_trajectories)
        while num_missing_trajectories > 0:
            zero_trajectory = [[0] * vector_dim for _ in range(vectors_per_trajectory)]
            padded_trajectories.append(zero_trajectory)
            num_missing_trajectories = num_missing_trajectories -1
        
        return padded_trajectories

    def __getitem__(self, idx):
        current_frame = self.df.iloc[idx]
        track_id = current_frame['trackId']
        recording_id = current_frame['recordingId']
        current_time = current_frame['frame']
        current_heading = current_frame['heading']
        current_x, current_y = current_frame['xCenter'], current_frame['yCenter']
        current_x, current_y = current_frame['xCenter'], current_frame['yCenter']
        target_x, target_y = current_frame['xTarget'], current_frame['yTarget']
        start_x, start_y = current_frame['xStart'], current_frame['yStart']
        route_lanelet_ids = current_frame['route_lanelet_ids']
        route_lanelet_ids = [int(num) for num in route_lanelet_ids.split(',')]
        current_point = BasicPoint3d(current_x,current_y,0)
        origin = torch.tensor([current_x, current_y], dtype=torch.float)
        # print("frame: " , current_frame)
        # print("trackid: ", track_id)
        # print("recordid: ", recording_id)


        # history_mask = (self.df['trackId'] == track_id) & (self.df['recordingId'] == recording_id) & \
        #             (self.df['frame'] <= current_time) & (self.df['frame'] >= current_time - self.history_frames)
        # history = self.df[history_mask].sort_values(by='frame', ascending=True)[['xCenter', 'yCenter','heading','velocity','length','width','frame']].values


        # history_vectors = []
        # for i in range(1, len(history)):
        #     time_stamp = history[i][-1] - current_time
        #     vector = [history[i-1][0], history[i-1][1], history[i][0], history[i][1] , 0,history[i][3],history[i][4],history[i][5],time_stamp]
        #     history_vectors.append(vector)
        # if len(history_vectors) < self.history_frames:
        #     history_vectors += [[0, 0, 0, 0, 0, 0, 0, 0, 0]] * (self.history_frames  - len(history_vectors))
        # history_trajectory = torch.tensor(history_vectors, dtype=torch.float)
        velo = np.sqrt(current_frame["xVelocity"]**2+current_frame["yVelocity"]**2)
        history_trajectory = [current_frame["length"], current_frame["width"], 0, 0,0,velo,0]
        
        # calculate short term reference
        reference_trajectory = self.get_points_from_lanelets(self.ll_map, route_lanelet_ids)
        x, y = zip(*reference_trajectory)
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        distance_interp = np.arange(0, cumulative_distances[-1], 1)
        x_interp_func = interp1d(cumulative_distances, x, kind='linear')
        y_interp_func = interp1d(cumulative_distances, y, kind='linear')
        x_interp = x_interp_func(distance_interp)
        y_interp = y_interp_func(distance_interp)
        reference_trajectory_points = torch.tensor(list(zip(x_interp, y_interp)))
        query_point = torch.tensor([current_x, current_y], dtype=torch.float)
        distances = torch.norm(reference_trajectory_points - query_point, dim=1)
        nearest_index = torch.argmin(distances)
        end_index = min(nearest_index + self.num_points_to_extract, len(reference_trajectory_points))
        reference_trajectory = reference_trajectory_points[nearest_index:end_index]
        if len(reference_trajectory) < self.num_points_to_extract:
            last_point = reference_trajectory[-1]
            additional_points = last_point.repeat(self.num_points_to_extract - len(reference_trajectory), 1)
            reference_trajectory = torch.cat((reference_trajectory, additional_points), dim=0)
        norm_reference_trajectory = rotate_and_translate(reference_trajectory, origin, current_heading)


        # calculate short term boundary
        left_boundary, right_boundary = self.get_boundaries_from_lanelets(self.ll_map, route_lanelet_ids)
        x, y = zip(*left_boundary)
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        distance_interp = np.arange(0, cumulative_distances[-1], 1)
        x_interp_func = interp1d(cumulative_distances, x, kind='linear')
        y_interp_func = interp1d(cumulative_distances, y, kind='linear')
        x_interp = x_interp_func(distance_interp)
        y_interp = y_interp_func(distance_interp)
        left_boundary_points = torch.tensor(list(zip(x_interp, y_interp)))
        query_point = torch.tensor([current_x, current_y], dtype=torch.float)
        distances = torch.norm(left_boundary_points - query_point, dim=1)
        nearest_index = torch.argmin(distances)
        end_index = min(nearest_index + self.num_points_to_extract, len(left_boundary_points))
        left_boundary = left_boundary_points[nearest_index:end_index]
        if len(left_boundary) < self.num_points_to_extract:
            last_point = left_boundary[-1]
            additional_points = last_point.repeat(self.num_points_to_extract - len(left_boundary), 1)
            left_boundary = torch.cat((left_boundary, additional_points), dim=0)
        norm_left_boundary = rotate_and_translate(left_boundary, origin, current_heading)
        

        x, y = zip(*right_boundary)
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        distance_interp = np.arange(0, cumulative_distances[-1], 1)
        x_interp_func = interp1d(cumulative_distances, x, kind='linear')
        y_interp_func = interp1d(cumulative_distances, y, kind='linear')
        x_interp = x_interp_func(distance_interp)
        y_interp = y_interp_func(distance_interp)
        right_boundary_points = torch.tensor(list(zip(x_interp, y_interp)))
        query_point = torch.tensor([current_x, current_y], dtype=torch.float)
        distances = torch.norm(right_boundary_points - query_point, dim=1)
        nearest_index = torch.argmin(distances)
        end_index = min(nearest_index + self.num_points_to_extract, len(right_boundary_points))
        right_boundary = right_boundary_points[nearest_index:end_index]
        if len(right_boundary) < self.num_points_to_extract:
            last_point = right_boundary[-1]
            additional_points = last_point.repeat(self.num_points_to_extract - len(right_boundary), 1)
            right_boundary = torch.cat((right_boundary, additional_points), dim=0)
        norm_right_boundary = rotate_and_translate(right_boundary, origin, current_heading)



    

        # label
        future_mask = (self.df['trackId'] == track_id) & (self.df['recordingId'] == recording_id) & \
                    (self.df['frame'] > current_time) & (self.df['frame'] <= current_time + self.future_frames)
        future_data = self.df[future_mask].sort_values(by='frame')[['acceleration', 'steering_angle']].values
        label = torch.zeros(self.future_frames * 2)
        label[:len(future_data)*2] = torch.tensor(future_data, dtype=torch.float).flatten()

        # neighbors
        current_frame_neighbors_mask = (self.df['frame'] == current_time) & (self.df['recordingId'] == recording_id) & \
                                    (self.df['trackId'] != track_id)
        neighbors = self.df[current_frame_neighbors_mask][['trackId', 'length','width','xCenter', 'yCenter','xTarget','xVelocity','yVelocity', 'yTarget','xStart','yStart','heading']]
        neighbors['distance'] = np.sqrt((neighbors['xCenter'] - current_x)**2 + (neighbors['yCenter'] - current_y)**2)
        nearest_neighbors = neighbors[neighbors['distance'] <= self.nearby_distance].nsmallest(self.max_neighbors, 'distance')

        # process neighbors' info
        nearby_trajectories = []
        norm_nearby_reference_trajectories = []
        for _, neighbor in nearest_neighbors.iterrows():
            neighbor_track_id = neighbor['trackId']
            currentn_x, currentn_y = neighbor['xCenter'], neighbor['yCenter']
            targetn_x, targetn_y = neighbor['xTarget'], neighbor['yTarget']
            startn_x, startn_y = neighbor['xStart'], neighbor['yStart']
            n_route_lanelet_ids = current_frame['route_lanelet_ids']
            n_route_lanelet_ids = [int(num) for num in n_route_lanelet_ids.split(',')]
            currentn_point = BasicPoint3d(currentn_x,currentn_y,0)
            neighbor_history_mask = (self.df['trackId'] == neighbor_track_id) & (self.df['recordingId'] == recording_id) & \
                                    (self.df['frame'] <=current_time) & (self.df['frame'] >= current_time - self.history_frames)
            neighbor_history = self.df[neighbor_history_mask].sort_values(by='frame', ascending=True)[['xCenter', 'yCenter','heading','velocity','length','width','frame']].values
            relx = neighbor["xCenter"] - current_x
            rely = neighbor["yCenter"] - current_y
            relheading = neighbor["heading"] - current_heading
            relheading = torch.atan2(torch.sin(torch.tensor(relheading)), torch.cos(torch.tensor(relheading)))
            vel = torch.sqrt(torch.tensor(neighbor["xVelocity"])**2 + torch.tensor(neighbor["yVelocity"])**2)
            relvx = vel*torch.cos(relheading)
            relvy = vel*torch.sin(relheading)
            nearby_trajectories.append([neighbor["length"],neighbor["width"],relx,rely,torch.rad2deg(relheading),relvx,relvy])


            reference_trajectoryn = self.get_points_from_lanelets(self.ll_map, n_route_lanelet_ids)
            x, y = zip(*reference_trajectoryn)
            distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
            distance_interp = np.arange(0, cumulative_distances[-1], 1)
            x_interp_func = interp1d(cumulative_distances, x, kind='linear')
            y_interp_func = interp1d(cumulative_distances, y, kind='linear')
            x_interp = x_interp_func(distance_interp)
            y_interp = y_interp_func(distance_interp)
            reference_trajectory_pointsn = torch.tensor(list(zip(x_interp, y_interp)))
            query_point = torch.tensor([currentn_x, currentn_y], dtype=torch.float)
            distances = torch.norm(reference_trajectory_pointsn - query_point, dim=1)
            nearest_index = torch.argmin(distances)
            end_index = min(nearest_index + self.num_points_to_extract, len(reference_trajectory_pointsn))
            reference_trajectoryn = reference_trajectory_pointsn[nearest_index:end_index]
            if len(reference_trajectoryn) < self.num_points_to_extract:
                last_point = reference_trajectoryn[-1]
                additional_points = last_point.repeat(self.num_points_to_extract - len(reference_trajectoryn), 1)
                reference_trajectoryn = torch.cat((reference_trajectoryn, additional_points), dim=0)
            norm_reference_trajectoryn = rotate_and_translate(reference_trajectoryn, origin, current_heading)
            # norm_reference_trajectoryn = torch.tensor(norm_reference_trajectoryn, dtype=torch.float32)
            norm_nearby_reference_trajectories.append(norm_reference_trajectoryn)

        while len(nearby_trajectories) < self.max_neighbors:
            nearby_trajectories.append([0, 0, 0, 0, 0, 0, 0])
            norm_nearby_reference_trajectories.append(torch.zeros((15, 2))) 


        history_trajectory = torch.tensor(history_trajectory, dtype=torch.float).unsqueeze(0)

        # norm_reference_trajectory = torch.tensor(norm_reference_trajectory, dtype=torch.float)
        norm_reference_trajectory = trajectory_to_vectors(norm_reference_trajectory)

        norm_left_boundary = norm_left_boundary.unsqueeze(0)
        norm_right_boundary = norm_right_boundary.unsqueeze(0)
        norm_lane_boundaries = torch.cat((norm_left_boundary, norm_right_boundary),dim=0)
        norm_lane_boundaries = trajectory_to_vectors(norm_lane_boundaries)

        nearby_trajectories = torch.tensor(nearby_trajectories, dtype=torch.float).unsqueeze(-2)
        # nearby_trajectories = trajectory_to_vectors(nearby_trajectories)
        norm_nearby_reference_trajectories = torch.stack(norm_nearby_reference_trajectories)
        norm_nearby_reference_trajectories = trajectory_to_vectors(norm_nearby_reference_trajectories)






        return history_trajectory, norm_reference_trajectory,nearby_trajectories, norm_nearby_reference_trajectories,norm_lane_boundaries, label
# dataset = VehicleTrajectoryDataset('/home/xavier/project/thesis/src/dataset/train','/home/xavier/project/thesis/src/dataset/map/Aachen_Roundabout_change.osm')
# history_trajectory, reference_trajectory,nearby_trajectories, nearby_reference_trajectories,processed_lane_boundaries,label = dataset[120]
# print("1")


    def get_feature(self, idx):
        try:

            features = self.__getitem__(idx)

            features_np = {
                'history_trajectory': features[0].numpy(),
                'reference_trajectory': features[1].numpy(),
                'nearby_trajectories': features[2].numpy(),
                'nearby_reference_trajectories': features[3].numpy(),
                'lane_boundaries': features[4].numpy(),
                'label': features[5].numpy()
            }

            return features_np

        except ValueError as e:
            print(f"Error processing index {idx}: {e}")
            print(f"History trajectory shape: {features[0].shape}")
            print(f"Reference trajectory shape: {features[1].shape}")
            print(f"Nearby trajectories shape: {features[2].shape}")
            print(f"Nearby reference trajectories shape: {features[3].shape}")
            print(f"Lane boundaries shape: {features[4].shape}")
            print(f"Label shape: {features[5].shape}")
            sys.exit(1)


            
from multiprocessing import Pool
import pickle
from tqdm import tqdm

def process_single_index(idx):
    try:

        features = dataset.get_feature(idx)
        return idx, features  
    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        return idx, None

def save_features_to_pickle(dataset, save_path, num_processes=10):
    indices = range(len(dataset))
    features_dict = {} 

    with Pool(processes=num_processes) as pool:
        for idx, features in tqdm(pool.imap_unordered(process_single_index, indices), total=len(indices)):
            if features is not None:
                features_dict[idx] = features
    features_list = [features_dict[idx] for idx in sorted(features_dict.keys())]

    with open(save_path, 'wb') as f:
        pickle.dump(features_list, f)

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

dataset = VehicleTrajectoryDataset('dataset/train/','map/round.osm')
save_features_to_pickle(dataset, '/generated_features/train/0405_0601_acc_angle_deg.features.pkl')