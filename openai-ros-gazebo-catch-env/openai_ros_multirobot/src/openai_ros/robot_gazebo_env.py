import rospy, tf
import numpy as np
import gym
from gym.utils import seeding
from .gazebo_connection import GazeboConnection
from .controllers_connection import ControllersConnection
from gazebo_msgs.srv import DeleteModel, SpawnModel, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Quaternion,Pose, Point
#https://bitbucket.org/theconstructcore/theconstruct_msgs/src/master/msg/RLExperimentInfo.msg
from openai_ros_multirobot.msg import RLExperimentInfo
from openai_ros.openai_ros_common import ROSLauncher
import time
import random



# https://github.com/openai/gym/blob/master/gym/core.py
class RobotGazeboEnv(gym.Env):

    def __init__(self, robot_name_space, controllers_list, reset_controls, start_init_physics_parameters=True, reset_world_or_sim="SIMULATION"):

        # To reset Simulations
        rospy.logdebug("START init RobotGazeboEnv")
        self.gazebo = GazeboConnection(start_init_physics_parameters,reset_world_or_sim)
        self.controllers_object = ControllersConnection(namespace=robot_name_space, controllers_list=controllers_list)
        self.reset_controls = reset_controls
        self.seed()
        self.step_number = 0
        # Set up ROS related variables
        self.episode_num = 0
        self.cumulated_episode_reward = 0.0
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
        rospy.logdebug("END init RobotGazeboEnv")
        self.home_pose = [0, 0]
    # Env methods
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done, info
        """

        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """
        #if self.step_number > 200:
            #self.reset()
        rospy.logdebug("START STEP OpenAIROS")

        self.gazebo.unpauseSim()
        self._set_action(action)
        #self._prey_step()
        self.gazebo.pauseSim()
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {}
        reward = self._compute_reward(obs, done)
        
        self.cumulated_episode_reward = self.cumulated_episode_reward+ reward
        self.step_number += 1
        rospy.logdebug("END STEP OpenAIROS")

        return obs, reward, done, info

    def reset(self):
        self.object_disposer_robot_win = 0
        self.step_number = 0
        rospy.logdebug("Reseting RobotGazeboEnvironment")
        self._reset_sim()
        
        self._init_env_variables()
        self._update_episode()
        obs = self._get_obs()
        rospy.logdebug("END Reseting RobotGazeboEnvironment")
        return obs
        

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

    def _update_episode(self):
        """
        Publishes the cumulated reward of the episode and
        increases the episode number by one.
        :return:
        """
        rospy.logwarn("PUBLISHING REWARD...")
        self._publish_reward_topic(
                                    self.cumulated_episode_reward,
                                    self.episode_num
                                    )
        rospy.logwarn("PUBLISHING REWARD...DONE="+str(self.cumulated_episode_reward)+",EP="+str(self.episode_num))

        self.episode_num += 1
        self.cumulated_episode_reward = 0.0


    def _publish_reward_topic(self, reward, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)

    # Extension methods
    # ----------------------------

    def _spwan(self):
            rospy.wait_for_service("gazebo/set_model_state")
            self.spawn_model = rospy.ServiceProxy("gazebo/set_model_state", SetModelState)

            object_disposer_robot_x = np.random.uniform(low=-35.4, high=25.4) 
            object_disposer_robot_y = np.random.uniform(low=15.0, high=20.0)
            object_disposer_robot_theta = np.random.uniform(low=0.0, high=2*np.pi)
            move_direction= [0,np.pi]
            object_disposer_robot_move_direction=random.choice(move_direction)
            object_disposer_robot_orient = Quaternion(*tf.transformations.quaternion_from_euler(0,0,object_disposer_robot_move_direction))
            object_disposer_robot_pose   =   Pose(Point(x=object_disposer_robot_x, y=object_disposer_robot_y,    z=0.15),   object_disposer_robot_orient)
            object_disposer_robot_model = ModelState()
            object_disposer_robot_model.model_name = "object_disposer_robot"
            object_disposer_robot_model.pose = object_disposer_robot_pose
            object_disposer_robot_model.reference_frame = "world"


            #add line follower robot in random position
            line_follower_car_x = np.random.uniform(low=-35.4, high=25.4) 
            line_follower_car_y = np.random.uniform(low=-4.4, high=4.4)
            line_follower_car_theta = np.random.uniform(low=0.0, high=2*np.pi)
            line_follower_car_orient = Quaternion(*tf.transformations.quaternion_from_euler(0,0,0))
            line_follower_car_pose   =   Pose(Point(x=line_follower_car_x, y=-28,    z=0.15),   line_follower_car_orient)
            line_follower_car_model = ModelState()
            line_follower_car_model.model_name = "line_follower_car"
            line_follower_car_model.pose = line_follower_car_pose
            line_follower_car_model.reference_frame = "world"

            #add box (object on rode) in random position.
            object_box_x = np.random.uniform(low=-35.4, high=object_disposer_robot_x-5)
            if object_disposer_robot_move_direction==0:
                pass
            else:
                object_box_x = np.random.uniform(low=object_disposer_robot_x+5, high=25.4)

            object_box_y = np.random.uniform(low=15.0, high=20.0)
            object_box_theta = np.random.uniform(low=0.0, high=2*np.pi)
            object_box_orient = Quaternion(*tf.transformations.quaternion_from_euler(0,0,0))
            object_box_pose   =   Pose(Point(x=object_box_x, y=object_box_y,    z=0.15),   object_box_orient)
            object_box_model = ModelState()
            object_box_model.model_name = "object_box"
            object_box_model.pose = object_box_pose
            object_box_model.reference_frame = "world"


            self.spawn_model(object_disposer_robot_model)   # add randomly line follower robot on rode
            self.spawn_model(line_follower_car_model)       # add randommly object_disposer robot on rode       
            self.spawn_model(object_box_model)              # add randomly box on rode


    def _reset_sim(self):
        """Resets a simulation
        """
        rospy.logdebug("RESET SIM START")
        if self.reset_controls :
            rospy.logdebug("RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            
            self.gazebo.pauseSim()
            
            self.gazebo.resetSim()
            if self.episode_num > 0:
                self._spwan()
            self.gazebo.unpauseSim()
            

            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        else:
            rospy.logwarn("DONT RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            
            self.gazebo.pauseSim()
            
            self.gazebo.resetSim()
            if self.episode_num > 0:
                self._spwan()
            self.gazebo.unpauseSim()
            
            #ros_ws_abspath = rospy.get_param("/turtlebot2/ros_ws_abspath", None)
            #create_random_launch_files()
            #ROSLauncher(rospackage_name="dql_robot",
                #launch_file_name="put_prey_in_world.launch",
                #ros_ws_abspath=ros_ws_abspath)
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        rospy.logdebug("RESET SIM END")
        return True

   

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _prey_step(self):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """Indicates whether or not the episode is done ( the robot has fallen for example).
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        raise NotImplementedError()

