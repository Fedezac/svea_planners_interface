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
    _points_path = None

    def __init__(self, points_path=[]):
        self._path_topic = load_param('~path_topic', '/path')
        self._path_pub = rospy.Publisher(self._path_topic, Path, latch=True, queue_size=1)
        self._pose_path = list()
        self._rviz_path = Path()
        self._points_path = points_path

    def create_pose_path(self):
        for point in self._points_path:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0
            self._pose_path.append(pose)

    def publish_rviz_path(self):
        if self._pose_path:
            self._rviz_path.header.frame_id = 'map'
            self._rviz_path.header.stamp = rospy.Time.now()
            self._rviz_path.poses = self._pose_path
            print('Publishing path (length = {}) ...'.format(len(self._pose_path)))
            self._path_pub.publish(self._rviz_path)

    def get_points_path(self):
        return self._points_path
    
    def get_points_path_reduced(self, granularity=4):
        indexes = np.linspace(0, len(self._points_path) - 1, num=int(np.round(len(self._points_path) / granularity)), dtype=int).tolist()
        return np.array(self._points_path)[indexes]
