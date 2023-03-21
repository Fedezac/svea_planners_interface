#! /usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

class PathInterface(object):
    _path_pub = None
    _path = None
    _path_topic = None
    _ros_path = None
    _limits = None
    _delta = None

    def __init__(self, limits, delta):
        self._path_topic = load_param('~path_topic', '/path')
        self._path_pub = rospy.Publisher(self._path_topic, Path, latch=True, queue_size=1)
        self._path = list()
        self._ros_path = Path()
        self._limits = limits
        self._delta = delta

    def create_ros_path(self, path):
        for point in path:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0
            self._path.append(pose)

    def publish_path(self):
        if self._path:
            self._ros_path.header.frame_id = 'map'
            self._ros_path.header.stamp = rospy.Time.now()
            self._ros_path.poses = self._path
            print('Publishing path (length = {}) ...'.format(len(self._path)))
            self._path_pub.publish(self._ros_path)
