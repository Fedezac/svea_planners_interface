#! /usr/bin/env python3

import rospy
from gridmap_interface import GridMapInterface
from path_interface import PathInterface
from astar import AStarPlanner, AStarWorld
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

def publish_map_internal_representation(delta, limits, obs):
    obstacles = []
    rows_int = int(np.round(limits[0][1] / delta[0]))
    cols_int = int(np.round(limits[1][1] / delta[1]))
    data = np.zeros(rows_int * cols_int).reshape(rows_int, cols_int)
    [obstacles.append(tup[0:2]) for tup in obs]
    obs_np = np.asarray(obstacles)
    data[obs_np[:, 0], obs_np[:, 1]] = 100  
    map_pub = rospy.Publisher('/map_from_grid', OccupancyGrid, latch=True, queue_size=1)
    map_msg = OccupancyGrid()
    map_msg.data = [item for sublist in np.array(data, dtype=int).tolist() for item in sublist]
    map_msg.header.frame_id = 'map'
    map_msg.header.stamp = rospy.Time.now()
    map_msg.info.resolution = delta[1]
    map_msg.info.width = cols_int
    map_msg.info.height = rows_int
    quat = quaternion_from_euler(0, math.pi , -math.pi / 2)
    map_msg.info.origin.orientation.x = quat[0]
    map_msg.info.origin.orientation.y = quat[1]
    map_msg.info.origin.orientation.z = quat[2]
    map_msg.info.origin.orientation.w = quat[3]
    print('Publishing map...')
    map_pub.publish(map_msg)


def main():
    rospy.init_node('planner_test')
    debug = load_param('~debug', True)
    a = GridMapInterface()
    a.init_ros_subscribers()
    delta, limits, obstacles = a.get_planner_world()

    start_x = 6.3
    start_y = 5.5
    goal_x = 4.1
    goal_y = 4.6 
    
    if debug:
        publish_map_internal_representation(delta, limits, obstacles)

    world = AStarWorld(delta=delta, limit=limits, obstacles=np.multiply([delta[0], delta[1], 1], np.array(obstacles)).tolist(), obs_margin=0.05)
    planner = AStarPlanner(world, [start_x, start_y], [goal_x, goal_y])
    path = np.asarray(planner.create_path())

    path_interface = PathInterface(limits, delta, path)
    path_interface.create_pose_path()
    path_interface.publish_rviz_path()
    print(path_interface.get_points_path_reduced())
    rospy.spin()
    
    
if __name__ == '__main__':
    main()