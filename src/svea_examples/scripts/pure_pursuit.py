#! /usr/bin/env python3

import numpy as np

import rospy
from rospy import Publisher, Rate
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler

from svea.models.bicycle import SimpleBicycleModel
from svea.states import VehicleState
from svea.simulators.sim_SVEA import SimSVEA
from svea.interfaces import LocalizationInterface
from svea_mocap.mocap import MotionCaptureInterface
from svea.controllers.pure_pursuit import PurePursuitController
from svea.svea_managers.path_following_sveas import SVEAPurePursuit
from svea.data import TrajDataHandler, RVIZPathHandler
# Added for obstace pose visualization
from svea.simulators.viz_utils import (
    publish_pose_array
)
from geometry_msgs.msg import PoseArray, PointStamped
from sensor_msgs.msg import LaserScan
from svea_planners.planner_interface import PlannerInterface 


def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)


def assert_points(pts):
    assert isinstance(pts, (list, tuple)), 'points is of wrong type, expected list'
    for xy in pts:
        assert isinstance(xy, (list, tuple)), 'points contain an element of wrong type, expected list of two values (x, y)'
        assert len(xy), 'points contain an element of wrong type, expected list of two values (x, y)'
        x, y = xy
        assert isinstance(x, (int, float)), 'points contain a coordinate pair wherein one value is not a number'
        assert isinstance(y, (int, float)), 'points contain a coordinate pair wherein one value is not a number'


def publish_initialpose(state, n=10):

    p = PoseWithCovarianceStamped()
    p.header.frame_id = 'map'
    p.pose.pose.position.x = state.x
    p.pose.pose.position.y = state.y

    q = quaternion_from_euler(0, 0, state.yaw)
    p.pose.pose.orientation.z = q[2]
    p.pose.pose.orientation.w = q[3]

    pub = Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    rate = Rate(10)

    for _ in range(n):
        pub.publish(p)
        rate.sleep()


class pure_pursuit:

    DELTA_TIME = 0.01
    TRAJ_LEN = 10
    GOAL_THRESH = 0.2
    # Obstacle threshold for obstacle management
    OBS_THRESH = 0.2
    TARGET_VELOCITY = 0.35
    RATE = 1e9
    POINTS = [] 

    def __init__(self):

        ## Initialize node

        rospy.init_node('pure_pursuit')

        ## Parameters
        self.OBSTACLES = load_param('~obstacles_points')
        self.IS_SIM = load_param('~is_sim', False)
        self.USE_RVIZ = load_param('~use_rviz', False)
        # Get remote rviz parameter
        self.REMOTE_RVIZ = load_param('~remote_rviz', True)
        self.STATE = load_param('~state', [0, 0, 0, 0])
        print(self.STATE)

        ## Set initial values for node

        # initial state
        self.state = VehicleState(*self.STATE)
        publish_initialpose(self.state)
        self.goal = [self.state.x, self.state.y]
        xs, ys = self.compute_traj()

        ## Create simulators, models, managers, etc.

        if self.IS_SIM:

            # simulator need a model to simulate
            self.sim_model = SimpleBicycleModel(self.state)

            # start the simulator immediately, but paused
            self.simulator = SimSVEA(self.sim_model,
                                     dt=self.DELTA_TIME,
                                     run_lidar=True,
                                     start_paused=True).start()
        
        # start the SVEA manager
        self.svea = SVEAPurePursuit(MotionCaptureInterface,
                                    PurePursuitController,
                                    xs, ys,
                                    data_handler=RVIZPathHandler if self.USE_RVIZ or self.REMOTE_RVIZ else TrajDataHandler)

        self.svea.controller.target_velocity = self.TARGET_VELOCITY
        # Create publisher in order to publish obstacle points onto RVIz
        self.init_pts_publisher = rospy.Publisher("/obstacles_points",
                                         PoseArray, queue_size=1) 
        # Subscriber for the obstacle points topic
        self.obs_sub = rospy.Subscriber('/clicked_point', PointStamped, self.obs_callback)
        
        self.svea.start(wait=True)

        # everything ready to go -> unpause simulator
        if self.IS_SIM:
            self.simulator.toggle_pause_simulation()

    def obs_callback(self, msg):
        obs = [msg.point.x, msg.point.y, 0]
        self.OBSTACLES.append(obs)

    def plan(self):
        start_x = 7.7
        start_y = 4.5
        goal_x = 6.9 
        goal_y = 6.5
        GRANULARITY = 4
        debug = False
        
        pi = PlannerInterface()
        pi.set_start([start_x, start_y])
        pi.set_goal([goal_x, goal_y])
        pi.initialize_planner_world()
        pi.compute_path()
        pi.initialize_path_interface()
        #[[7.700000114738941, 4.500000067055225], [7.700000114738941, 4.700000070035458], [7.650000113993883, 4.90000007301569], [7.500000111758709, 5.15000007674098], [7.400000110268593, 5.350000079721212], [7.350000109523535, 5.550000082701445], [7.200000107288361, 5.800000086426735], [7.1500001065433025, 6.000000089406967], [7.050000105053186, 6.200000092387199], [6.900000102818012, 6.45000009611249]]
        self.POINTS = pi.get_points_path(GRANULARITY)
        self.POINTS = self.POINTS.tolist()
        assert_points(self.POINTS)
        print(self.POINTS)

        if debug:
            pi.publish_internal_representation()

        pi.publish_rviz_path()
        # create goal state
        self.curr = 0
        self.goal = self.POINTS[self.curr]
        print("goal = {}".format(self.goal))
        

    def run(self):
        self.plan()
        xs, ys = self.compute_traj()
        self.svea.update_traj(xs, ys)
        while self.keep_alive():
            self.spin()

    def keep_alive(self):
        #!! self.svea.is_finished becomes true if in pure_pursuit controller, function _calc_target_index,
        return not (self.svea.is_finished or rospy.is_shutdown())

    def spin(self):
        #!! Safe to send controls is localization node is up and running
        safe = self.svea.localizer.is_ready
        #!! Previous was state and not self.state (so when computing new trajectory for new goal, the trajectory would 
        #!! always start fro (0,0) and not the new position)
        # limit the rate of main loop by waiting for state
        self.state = self.svea.wait_for_state()

        # Obstacle Management
        x = []
        y = []
        yaws = []
        # if state.x,y is too close to obtacle point, then stop
        for obs in self.OBSTACLES:
            # Added for obstacle pose visualization
            x.append(obs[0])
            y.append(obs[1])
            yaws.append(0)
            if np.hypot(self.state.x - obs[0], self.state.y - obs[1]) < self.OBS_THRESH:
                # If vehicle is too close to an obstacle, then stop its motion
                self.svea.send_control(0, 0)
                safe = False
        publish_pose_array(self.init_pts_publisher, x, y, yaws)

        if safe:
            steering, velocity = self.svea.compute_control(self.state)
            self.svea.send_control(steering, velocity)

        if np.hypot(self.state.x - self.goal[0], self.state.y - self.goal[1]) < self.GOAL_THRESH:
            self.update_goal()
            xs, ys = self.compute_traj()
            self.svea.update_traj(xs, ys)

        self.svea.visualize_data()

    def update_goal(self):
        self.curr += 1
        if self.curr == len(self.POINTS):
            self.svea.send_control(0, 0)
            self.svea.controller.is_finished = True
            print("Goal reached!")
        else:
            if self.curr < len(self.POINTS):
                print("new_goal = {}, curr = {}, len={}".format(self.goal, self.curr, len(self.POINTS)))
                self.goal = self.POINTS[self.curr]

    def compute_traj(self):
        xs = np.linspace(self.state.x, self.goal[0], self.TRAJ_LEN)
        ys = np.linspace(self.state.y, self.goal[1], self.TRAJ_LEN)
        return xs, ys


if __name__ == '__main__':

    ## Start node ##

    pure_pursuit().run()