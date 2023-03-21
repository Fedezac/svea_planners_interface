#! /usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)


class GridMapInterface(object):
    _gridmap_msg = None
    _gridmap_sub = None
    _map_topic = None
    _gridmap = None

    def __init__(self):
        self._map_topic = load_param('~map_topic', '/map')

    def init_ros_subscribers(self):
        self._gridmap_sub = rospy.Subscriber(self._map_topic, OccupancyGrid, self._gridmap_cb, queue_size=1)
        
    def _gridmap_cb(self, msg):
        self._gridmap_msg = OccupancyGrid(msg.header, msg.info, msg.data)

    def _get_delta(self):
        if self._gridmap_msg  is not None:
            return [self._gridmap_msg.info.resolution, self._gridmap_msg.info.resolution]
    
    def _get_limits(self):
        if self._gridmap_msg is not None:
            return [[0, self._gridmap_msg.info.width * self._gridmap_msg.info.resolution], [0, self._gridmap_msg.info.height * self._gridmap_msg.info.resolution]]
    
    def _get_obstacles(self):
        if self._gridmap_msg  is not None:
            gridmap = np.array(self._gridmap_msg.data).reshape((self._gridmap_msg.info.height, self._gridmap_msg.info.width)).T * 0.01
            # Cell values are changed to -0.01 (unknown), 0 (free), 1 (obstacle)
            obstacles = []
            for (x, y), cell in np.ndenumerate(gridmap):
                #!! Strong assumption: unknown cells are considered as obstacles
                if cell < 0 or cell == 1:
                    obstacles.append([x, y, self._gridmap_msg.info.resolution])
            return obstacles

    def get_planner_world(self):
        while self._gridmap_msg is None:
            continue
        return self._get_delta(), self._get_limits(), self._get_obstacles()
