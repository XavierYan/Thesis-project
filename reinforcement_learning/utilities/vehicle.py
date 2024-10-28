import lanelet2
import numpy as np


from utils_map import *

from lanelet2.core import BasicPoint3d
import numpy as np


class Vehicle(object):

    def __init__(
        self,  trajectory = None, start_x = None, start_y = None,end_x = None, end_y = None,ll_map = None,length = 4,width = 3
    ):


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



        self.route_lanelet_ids = self._find_route_lanelet_ids(ll_map)
        self.reference_trajectory = self._get_points_from_lanelets(ll_map, self.route_lanelet_ids)
        self.left_boundary, self.right_boundary = self._get_boundaries_from_lanelets(ll_map, self.route_lanelet_ids)


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

                    route_lanelet_ids.extend([lanelet.id for lanelet in route.shortestPath()])
                    return route_lanelet_ids  
        return route_lanelet_ids
    

        
