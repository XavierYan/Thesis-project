


import lanelet2
import numpy as np
import torch


def find_nearest_lanelet(pos, ll_map,distance = 5):
    trafficRules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)
    lanelets=lanelet2.geometry.findWithin3d(ll_map.laneletLayer, pos, distance)
    if(len(lanelets)>0):
        # print("Lanelets found at given map-position!")
        for ll in lanelets:
            if(trafficRules.canPass(ll[1])):
                return ll[1]
        print("No Lanelet found which can be used by a vehicle!")
        return
    else:
        print("No Lanelets found at given map-position!")
        return


def find_nearest_lanelets(pos, ll_map,distance = 5):

    trafficRules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)
    lanelets=lanelet2.geometry.findWithin3d(ll_map.laneletLayer, pos, distance)
    pass_lanelets = []
    if(len(lanelets)>0):
        # print("Lanelets found at given map-position!")
        for ll in lanelets:
            if(trafficRules.canPass(ll[1])):
                pass_lanelets.append(ll[1])
        if (len(pass_lanelets) == 0):

            print("No Lanelet found which can be used by a vehicle!")
            return
        else:
            return pass_lanelets
    else:
        print("No Lanelets found at given map-position!")
        return
    


def extract_points_from_trajectory(trajectory):
    points = []
    for vector in trajectory:
        start_point = (vector[0].item(), vector[1].item())
        end_point = (vector[2].item(), vector[3].item())
        if not points or points[-1] != start_point:  # 避免重复添加相同的起点
            points.append(start_point)
        points.append(end_point)
    return points

def rotate_point(x, y, theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return cos_theta * x - sin_theta * y, sin_theta * x + cos_theta * y

def normalize_vector(vector, current_x, current_y, current_heading):
    vector = np.array(vector)
    if np.all(vector == 0):  # 检查向量是否全为0
        return torch.zeros_like(torch.tensor(vector))  # 返回一个全0的Tensor
    x_start, y_start, x_end, y_end = vector[:4]
    # 平移并旋转起始点和结束点
    translated_x_start, translated_y_start = x_start - current_x, y_start - current_y
    rotated_x_start, rotated_y_start = rotate_point(translated_x_start, translated_y_start, -current_heading)
    translated_x_end, translated_y_end = x_end - current_x, y_end - current_y
    rotated_x_end, rotated_y_end = rotate_point(translated_x_end, translated_y_end, -current_heading)
    # 更新向量，并保留其他特征不变
    new_vector = [rotated_x_start, rotated_y_start, rotated_x_end, rotated_y_end] + list(vector[4:])
    return torch.tensor(new_vector, dtype=torch.float)

def normalize_trajectory(trajectory, current_x, current_y, current_heading):
    normalized_trajectory = [normalize_vector(vector, current_x, current_y, current_heading) for vector in trajectory]
    return torch.stack(normalized_trajectory)  # 使用torch.stack合并Tensor列表

def adjust_heading(vehicle_heading, current_vehicle_heading):
    adjusted_heading = vehicle_heading - current_vehicle_heading
    adjusted_heading = np.arctan2(np.sin(adjusted_heading), np.cos(adjusted_heading))
    return adjusted_heading

def normalize_and_adjust_vector(vector, current_x, current_y, current_heading):
    vector = np.array(vector)
    if np.all(vector == 0):  # 检查向量是否全为0
        return torch.zeros_like(torch.tensor(vector))  # 返回一个全0的Tensor
    x_start, y_start, x_end, y_end, heading = vector[:5]
    # 平移并旋转起始点和结束点
    translated_x_start, translated_y_start = x_start - current_x, y_start - current_y
    rotated_x_start, rotated_y_start = rotate_point(translated_x_start, translated_y_start, -current_heading)
    translated_x_end, translated_y_end = x_end - current_x, y_end - current_y
    rotated_x_end, rotated_y_end = rotate_point(translated_x_end, translated_y_end, -current_heading)
    # 调整heading角
    adjusted_heading = adjust_heading(heading, current_heading)
    # 更新向量，并保留其他特征不变
    new_vector = [rotated_x_start, rotated_y_start, rotated_x_end, rotated_y_end, adjusted_heading] + list(vector[5:])
    return torch.tensor(new_vector, dtype=torch.float)

def normalize_and_adjust_trajectory(trajectory, current_x, current_y, current_heading):
    normalized_trajectory = [normalize_and_adjust_vector(vector, current_x, current_y, current_heading) for vector in trajectory]
    return torch.stack(normalized_trajectory)  # 使用torch.stack合并Tensor列表

def get_attributes(linestring):

    attributes = linestring.attributes
    way_type = attributes["type"] if "type" in attributes else None

    subtype = attributes["subtype"] if "subtype" in attributes else None
    return way_type, subtype

def get_feature_encode(way_type, subtype):
    feature_encode = [0, 0, 0, 0] 
    if way_type == "curbstone" or way_type == 'guard_rail' or way_type == 'road_border':
        feature_encode[0] = 1
    elif way_type == "line_thin":
        if subtype == "dashed":
            feature_encode[1] = 1
            feature_encode[2] = 1
        else:
            feature_encode[1] = 1
    elif way_type == "virtual":
        feature_encode[3] = 1
    return feature_encode

def calculate_distance(point, line_points):
    point = np.array(point)
    distances = np.linalg.norm(line_points - point, axis=1)
    return np.min(distances), np.argmin(distances)


def find_nearest_lanelet(ll_map,lanelet_ids, point):
    nearest_lanelet_id = None
    min_distance = float('inf')
    point = [point.x, point.y]  # 将BasicPoint3d转换为包含x和y的列表

    for lanelet_id in lanelet_ids:
        lanelet = ll_map.laneletLayer[lanelet_id]
        # 将每个点转换为包含x和y的列表
        centerline = [[p.x, p.y] for p in lanelet.centerline]

        distance, _= calculate_distance(point, centerline)  # 计算点到中心线的距离

        if distance < min_distance:  # 如果找到更近的Lanelet，则更新最近的Lanelet和最小距离
            min_distance = distance
            nearest_lanelet_id = lanelet_id

    return nearest_lanelet_id

def collect_boundary_points(ll_map,lanelet_ids,boundary_side, point,num_vectors=20):
    nearest_lanelet_id = find_nearest_lanelet(ll_map,lanelet_ids, point)
    point_np = np.array([point.x, point.y])
    collected_vectors = []
    
    start_index = lanelet_ids.index(nearest_lanelet_id)

    for lanelet_id in lanelet_ids[start_index:]:
        # print(lanelet_id)
        lanelet = ll_map.laneletLayer[lanelet_id]
        boundary = lanelet.leftBound if boundary_side == 'left' else lanelet.rightBound
        way_type, subtype = get_attributes(boundary)
        feature_encode = get_feature_encode(way_type, subtype)

        boundary_np = np.array([[p.x, p.y] for p in boundary])
        

        if lanelet_id == nearest_lanelet_id:
            _, nearest_point_index = calculate_distance(point_np, boundary_np)
            for i in range(nearest_point_index, len(boundary_np)-1):
                if len(collected_vectors) >= num_vectors: break
                start_point = boundary_np[i]
                end_point = boundary_np[i + 1]
                vector = np.concatenate([start_point, end_point, feature_encode])
                collected_vectors.append(vector)
        else:  # 对于后续的lanelet，从第一个边界点开始收集
            for i in range(len(boundary_np) - 1):
                if len(collected_vectors) >= num_vectors: break
                start_point = boundary_np[i]
                end_point = boundary_np[i + 1]
                vector = np.concatenate([start_point, end_point, feature_encode])
                collected_vectors.append(vector)

        if len(collected_vectors) >= num_vectors: break

    # # 用0向量填充剩余的空间
    # while len(collected_vectors) < num_vectors:
    #     zero_vector = np.concatenate([[0, 0], [0, 0], feature_encode])
    #     collected_vectors.append(zero_vector)

    return np.array(collected_vectors)


def collect_centerlines_points(ll_map,lanelet_ids, point, num_vectors=20):
    nearest_lanelet_id = find_nearest_lanelet(ll_map,lanelet_ids, point)
    point_np = np.array([point.x, point.y])
    collected_vectors = []
    start_index = lanelet_ids.index(nearest_lanelet_id)

    for lanelet_id in lanelet_ids[start_index:]:
        lanelet = ll_map.laneletLayer[lanelet_id]
        centerline = lanelet.centerline
        centerline_np = np.array([[p.x, p.y] for p in centerline])

        if lanelet_id == nearest_lanelet_id:
            _, nearest_point_index = calculate_distance(point_np, centerline_np)
            for i in range(nearest_point_index, len(centerline_np)-1):
                if len(collected_vectors) >= num_vectors: break
                start_point = centerline_np[i]
                end_point = centerline_np[i + 1]
                vector = np.concatenate([start_point, end_point])
                collected_vectors.append(vector)
        else:  # 对于后续的lanelet，从第一个边界点开始收集
            for i in range(len(centerline_np) - 1):
                if len(collected_vectors) >= num_vectors: break
                start_point = centerline_np[i]
                end_point = centerline_np[i + 1]
                vector = np.concatenate([start_point, end_point])
                collected_vectors.append(vector)
        if len(collected_vectors) >= num_vectors: break

    # while len(collected_vectors) < num_vectors:
    #     zero_vector = np.concatenate([[0, 0], [0, 0,0]])
    #     collected_vectors.append(zero_vector)

    return collected_vectors


def interpolate_vectors(lane_boundaries, interval=1, max_vectors=15):

    
    points = []
    num = len(lane_boundaries[0])
    for vector in lane_boundaries:
        if all(value == 0 for value in vector[:4]):
            continue  # 忽略全零向量

        start_point = torch.tensor(vector[:2])
        end_point = torch.tensor(vector[2:4])
        features = vector[4:]

        direction = end_point - start_point
        length = torch.norm(direction)

        if length > 0:
            num_points_to_insert = min(int(length // interval), max_vectors +1 - len(points))
            for i in range(num_points_to_insert):
                new_point = start_point + direction * ((i + 1) * interval / length)
                points.append(new_point.tolist())

                if len(points) >= max_vectors+1:
                    break

        if len(points) >= max_vectors+1:
            break

    # 将点转换为向量
    # vectors = [points[i] + points[i + 1] + features for i in range(len(points) - 1)]
    vectors = [np.concatenate((points[i], points[i + 1], features)) for i in range(len(points) - 1)]
    
    # 填充剩余的向量，如果有必要的话
    while len(vectors) < max_vectors:
        vectors.append([0] * num)  # 假设向量和特征的总长度为8

    # interpolated_vectors.append(vectors)

    return vectors

def interpolate_reference_vectors(lane_boundaries, interval=1, max_vectors=15):

    
    points = []
    num = len(lane_boundaries[0])
    for vector in lane_boundaries:
        if all(value == 0 for value in vector[:4]):
            continue  # 忽略全零向量

        start_point = torch.tensor(vector[:2])
        end_point = torch.tensor(vector[2:4])
        features = vector[4:]

        direction = end_point - start_point
        length = torch.norm(direction)

        if length > 0:
            num_points_to_insert = min(int(length // interval), max_vectors+1 - len(points))
            for i in range(num_points_to_insert):
                new_point = start_point + direction * ((i + 1) * interval / length)
                points.append(new_point.tolist())

                if len(points) >= max_vectors+1 :
                    break

        if len(points) >= max_vectors+1 :
            break

    # 将点转换为向量
    # vectors = [points[i] + points[i + 1] + features for i in range(len(points) - 1)]
    vectors = [np.concatenate((points[i], points[i + 1], [i+1])) for i in range(len(points) - 1)]
    
    # 填充剩余的向量，如果有必要的话
    while len(vectors) < max_vectors+1 - 1:
        vectors.append([0] * 5)  # 假设向量和特征的总长度为8

    # interpolated_vectors.append(vectors)

    return vectors


# def update_state(x, y, heading, velocity, steering_angle, acceleration, steering_rate, vehicle_length=2,delta_t=1/25):
#     # heading = np.rad2deg(heading)
#     delta_heading = (velocity / (0.6*vehicle_length)) * np.tan(steering_angle) * delta_t
    
#     x += velocity * np.cos(heading) * delta_t
#     y += velocity * np.sin(heading) * delta_t
#     # 更新前轮转角
#     steering_angle += steering_rate
#     heading = heading+delta_heading
#     # 更新速度
#     velocity += acceleration * delta_t
#     # print(steering_angle)
#     return x, y, heading, velocity, steering_angle

# def predict_trajectory(x, y, heading, velocity, steering_angle, model_outputs,  vehicle_length=4,delta_t=1/25):
#     # print(steering_angle)
#     trajectory = [(x, y)]
#     # steering_angle = np.deg2rad(steering_angle)
#     # print(steering_angle)
#     model_outputs = model_outputs.view(-1, 2).numpy()
#     for acceleration, steering_rate in model_outputs:
#         # steering_rate = np.deg2rad(steering_rate)
        
#         x, y, heading, velocity, steering_angle = update_state(
#             x, y, heading, velocity, steering_angle,
#             acceleration, steering_rate,  vehicle_length,delta_t
#         )
#         trajectory.append((x, y))

#     return trajectory

def update_state(x, y, heading, velocity, steering_angle, acceleration,steering_angle_next, vehicle_length=2,delta_t=1/25):
    # heading = np.rad2deg(heading)
    delta_heading = (velocity / (0.6*vehicle_length)) * np.tan(steering_angle) * delta_t
    
    x += velocity * np.cos(heading) * delta_t
    y += velocity * np.sin(heading) * delta_t
    # 更新前轮转角
    steering_angle = steering_angle_next
    heading = heading+delta_heading
    # 更新速度
    velocity += acceleration * delta_t
    # print(steering_angle)
    return x, y, heading, velocity, steering_angle

def predict_trajectory(x, y, heading, velocity, steering_angle, model_outputs,  vehicle_length=4,delta_t=1/25):
    # print(steering_angle)
    trajectory = [(x, y)]
    # steering_angle = np.deg2rad(steering_angle)
    # print(steering_angle)
    model_outputs = model_outputs.view(-1, 2).numpy()
    for acceleration, steering_angle_next in model_outputs:
        # steering_rate = np.deg2rad(steering_rate)
        
        x, y, heading, velocity, steering_angle = update_state(
            x, y, heading, velocity, steering_angle,
            acceleration, steering_angle_next,  vehicle_length,delta_t
        )
        trajectory.append((x, y))

    return trajectory