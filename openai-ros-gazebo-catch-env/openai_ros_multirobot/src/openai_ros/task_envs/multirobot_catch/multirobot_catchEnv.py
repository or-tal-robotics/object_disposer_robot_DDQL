import rospy
import numpy as np
from gym import spaces
from openai_ros.robot_envs import multirobot_catch_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os
import cv2
import time
import math

def get_image_moment(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,0]) 
    upper = np.array([255,255,10]) 
    mask = cv2.inRange(hsv, lower, upper) 
    M = cv2.moments(mask)
    return M['m00']


class CatchEnv(multirobot_catch_env.TurtleBot2catchEnv):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot2 in some kind of maze.
        It will learn how to move around the maze without crashing.
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot2/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="dql_robot",
                    launch_file_name="rode_simulation_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        rospy.logdebug("finish loading sumo_world.launch")

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros_multirobot",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/multirobot_catch/config",
                               yaml_file_name="multirobot_catch.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(CatchEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot2/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)
        
        
        
        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot2/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot2/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot2/angular_speed')
        
        
        # We only use two integers
        self.observation_space = spaces.Box(low=0, high=255, shape= (640, 480, 3))
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # Rewards
        self.catch_reward = rospy.get_param("/turtlebot2/catch_reward")
        self.cooperative_catch_reward = rospy.get_param("/turtlebot2/cooperative_catch_reward")
        self.time_penelty = rospy.get_param("/turtlebot2/time_penelty")
        self.robot_out_of_bounds_penalty = rospy.get_param("/turtlebot2/robot_out_of_bounds_penalty")
        self.robot_hit_robot_penalty = rospy.get_param("/turtlebot2/robot_hit_robot_penalty")
        self.max_x = rospy.get_param("/turtlebot2/max_x") 
        self.max_y = rospy.get_param("/turtlebot2/max_y") 
        self.min_x = rospy.get_param("/turtlebot2/min_x") 
        self.min_y = rospy.get_param("/turtlebot2/min_y")
        self.goal_max_x = rospy.get_param("/turtlebot2/goal_max_x") 
        self.goal_max_y = rospy.get_param("/turtlebot2/goal_max_y") 
        self.goal_min_x = rospy.get_param("/turtlebot2/goal_min_x") 
        self.goal_min_y = rospy.get_param("/turtlebot2/goal_min_y")

        #my simulation rewards
        self.catch_box=rospy.get_param("/turtlebot2/catch_box")
        self.put_box_out=rospy.get_param("/turtlebot2/put_box_out")
        self.line_follower_and_object_crash=rospy.get_param("/turtlebot2/line_follower_and_object_crash")
        self.line_follower_and_object_disposer_crash=rospy.get_param("/turtlebot2/line_follower_and_object_disposer_crash")
        self.object_disposer_out_without_box=rospy.get_param("/turtlebot2/object_disposer_out_without_box")


        self.cumulated_steps = 0 #number of steps now.
        self.max_steps = rospy.get_param('/turtlebot2/max_steps_episode') #max possible number of steps
        
        
        self.predator_win = 0
        self.prey_win = 0
        #flags
        self.box_collected_flag=0
        self.crash=0

        self.counter_box=0 #count how much boxes out of rode.
        self.object_disposer_out_of_line=0 #object disposer out of rode (yellow line)
        self.object_disposer_out_of_game_area=0 #cheak if object disposer robot out of game are (grey area)

        self.print_out_of_lines=0   #print massage that robot disposer out of yellow line
        self.print_in_of_lines=0    #print massage that robot disposer in yellow line

        #flags for boxes out
        self.box_1_out=0
        self.box_2_out=0
        self.box_3_out=0
        self.box_4_out=0

        self.box_1_out_print=0
        self.box_2_out_print=0
        self.box_3_out_print=0
        self.box_4_out_print=0

        self.steps_flag=0

        #flag if reward is given
        self.box_1_reward=0
        self.box_2_reward=0
        self.box_3_reward=0
        self.box_4_reward=0
        self.object_disposer_returned_to_lines_reward=0
        self.box_1=0
        self.white_area_steps_counter=0
        self.white_area_steps_max=200
        self.white_area_steps_flag=0
        self.flag_box_out=0
        self.reward_for_back_to_lines=0

        self.reward=0.0

        self.flag_try=0

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self.object_robot_disposer_win = 0
        #self.prey_win = 0

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        
        #print("Start Set Action ==>"+str(action))
        linear_speed = 0.0
        angular_speed = 0.0

        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = 0.0
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = 0.0
            angular_speed = -self.angular_speed
            self.last_action = "TURN_RIGHT"
        elif action == 3: #BACK
            linear_speed =-0.6
            self.last_action = "BACK"
        # elif action == 3: #RIGHT FORWARD
        #     linear_speed = self.linear_turn_speed
        #     angular_speed = -self.angular_speed
        #     self.last_action = "FORWARDS_TURN_RIGHT"
        # elif action == 4: #LEFT FORWARD
        #     linear_speed = self.linear_turn_speed
        #     angular_speed = self.angular_speed
        #     self.last_action = "FORWARDS_TURN_LEFT"
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)  
        time.sleep(0.1) #0.005
        self.cumulated_steps=self.cumulated_steps+1

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        #observations from camera
        img_observations = self.get_camera_rgb_image_raw()

        #new addition 10.3.20
        #observations from laser scanner
        laser_observations=self.get_laser_image()

        #laser_observations = [self.LaserScan_prey, self.LaserScan_predator]
        rospy.logdebug("END Get Observation ==>")
        #new addition 10.3.20
        return img_observations



    #reset function to simulation
    def _is_done(self, observations):
        self._episode_done = False

        object_disposer_position = np.array(self.get_object_disposer_robot_position())
        
        line_follower_car_position= np.array(self.get_line_follower_car_position())
        
        object_box_position=np.array(self.get_object_box_position())
        
        object_box_2_position=np.array(self.get_object_box_2_position())

        object_box_3_position=np.array(self.get_object_box_3_position())

        object_box_4_position=np.array(self.get_object_box_4_position())

        #make list of all boxes position , [x][y] 0- box 1 1- box2 2- box 3 3- box 4
        all_boxes_position=np.array([object_box_position,object_box_2_position,object_box_3_position,object_box_4_position])
        
        
        #for i in range (0,4):
        #    print i 
        #    print all_boxes_position[i][0]
        #    print all_boxes_position[i][1]

        #print all_boxes_position


        object_disposer_orentation=np.array(self.get_object_box_orientation())

        #the reset application WITHOUT rewards!! keep in mind
        if object_disposer_position[0]> -39.1 and object_disposer_position[0]<27.4:
        #cheack if robot in bounderies of yellow line in stright parts.
            if (object_disposer_position[1]> -29.3 and object_disposer_position[1]< -22.41) or (object_disposer_position[1]>14.27 and object_disposer_position[1]<21.2):
                #print "====+++ IN  BOUNDERIES +++===="
                if self.print_in_of_lines==0:
                    print "====+++ IN  BOUNDERIES +++===="
                    self.print_in_of_lines=1
                    self.print_out_of_lines=0
                    if self.flag_box_out==1:
                        self.reward_for_back_to_lines=1

                if self.object_disposer_out_of_line==1:
                    self.object_disposer_returned_to_lines_reward=1
                self.object_disposer_out_of_line=0
            else:
                if self.print_out_of_lines==0: 
                    print "====+++ OUT OF BOUNDERIES - STRIGHT LINES +++===="
                    self.print_out_of_lines=1
                    self.print_in_of_lines=0
                self.object_disposer_out_of_line=1
                self.object_disposer_returned_to_lines_reward=0
                
                    
                #self._episode_done = True
        else:
            #zone --A--B--C--D--
            if object_disposer_position[0]<=-39.1:
                #between A and B
                if object_disposer_position[1]>14.27 and object_disposer_position[1]<22.5:
                    if object_disposer_position[0]<=-39.1:
                        upper_border=(0.03718*(object_disposer_position[1]**2))+(0.3011*object_disposer_position[1])+(-62.19)-3.9
                        lower_border=(-39.1)
                        if lower_border>object_disposer_position[0] and upper_border<object_disposer_position[0]:
                            if self.print_in_of_lines==0:
                                print "====+++ IN  BOUNDERIES +++===="
                                self.print_in_of_lines=1
                                self.print_out_of_lines=0
                                if self.flag_box_out==1:
                                    self.reward_for_back_to_lines=1

                            if self.object_disposer_out_of_line==1:
                                self.object_disposer_returned_to_lines_reward=1
                            self.object_disposer_out_of_line=0
                        else:
                            if self.print_out_of_lines==0: 
                                print "====+++ OUT OF BOUNDERIES - Part A-B +++===="
                                self.print_out_of_lines=1
                                self.print_in_of_lines=0
                            self.object_disposer_out_of_line=1
                            self.object_disposer_returned_to_lines_reward=0
                            
                            #self._episode_done = True

                #between B and C
                if object_disposer_position[1]> -19.7 and object_disposer_position[1]<=14.27 :
                    upper_border=(0.0186*(object_disposer_position[1]**2))+(0.1736*object_disposer_position[1])+(-61.66)
                    lower_border=0.031*(object_disposer_position[1]**2)+0.232*object_disposer_position[1]+(-55.31)
                    if object_disposer_position[1] < 7.04 and object_disposer_position[1] > (-12.29) :
                        lower_border=0.031*(object_disposer_position[1]**2)+0.232*object_disposer_position[1]+(-55.31)
                    else:
                        if object_disposer_position[1]>=7.04 :
                            lower_border=0.1244*(object_disposer_position[1]**2)-1.144*object_disposer_position[1]+(-50.25)
                        elif object_disposer_position[1]<=-12.29 :
                            lower_border=0.153*(object_disposer_position[1]**2)+3.906*object_disposer_position[1]+(-28.94)
                    if lower_border>object_disposer_position[0] and upper_border<object_disposer_position[0]:
                        if self.print_in_of_lines==0:
                            print "====+++ IN  BOUNDERIES +++===="
                            self.print_in_of_lines=1
                            self.print_out_of_lines=0
                            if self.flag_box_out==1:
                                self.reward_for_back_to_lines=1

                        if self.object_disposer_out_of_line==1:
                            self.object_disposer_returned_to_lines_reward=1
                        
                        self.object_disposer_out_of_line=0
                        
                    else:
                        if self.print_out_of_lines==0: 
                            print "====+++ OUT OF BOUNDERIES - Part B-C +++===="
                            self.print_out_of_lines=1
                            self.print_in_of_lines=0
                        self.object_disposer_out_of_line=1
                        self.object_disposer_returned_to_lines_reward=0
                        self.object_disposer_returned_to_lines_reward=0

                        #self._episode_done = True

                #beetwen C and D
                if object_disposer_position[1]>-29.3 and object_disposer_position[1]<=-19.7 :
                    lower_border=0.953*(object_disposer_position[1]**2)+35.16*object_disposer_position[1]+(275.9)
                    if lower_border>=-39.1:
                        upper_border=(0.03718*(object_disposer_position[1]**2))+(0.3011*object_disposer_position[1])+(-62.19)-3.9
                        lower_border=(-39.1)
                    else:
                        upper_border=(0.03718*(object_disposer_position[1]**2))+(0.3011*object_disposer_position[1])+(-62.19)-3.9
                        lower_border=0.953*(object_disposer_position[1]**2)+35.16*object_disposer_position[1]+(275.9)
                    if lower_border>object_disposer_position[0] and upper_border<object_disposer_position[0]:
                        if self.print_in_of_lines==0:
                                print "====+++ IN  BOUNDERIES +++===="
                                self.print_in_of_lines=1
                                self.print_out_of_lines=0
                                if self.flag_box_out==1:
                                    self.reward_for_back_to_lines=1

                        if self.object_disposer_out_of_line==1:
                            self.object_disposer_returned_to_lines_reward=1
                        self.object_disposer_out_of_line=0
                    else:
                        if self.print_out_of_lines==0: 
                            print "====+++ OUT OF BOUNDERIES - Part C-D +++===="
                            self.print_out_of_lines=1
                            self.print_in_of_lines=0
                        self.object_disposer_out_of_line=1
                        self.object_disposer_returned_to_lines_reward=0
                        #self._episode_done = True
            
            #zone --E--F--G--H--
            elif object_disposer_position[0]>=27.4:
                #between E and F
                if object_disposer_position[1]> 14.27 and object_disposer_position[1]<22.5 :
                    if object_disposer_position[0]>=27.4:
                        upper_border=(-0.3317*(object_disposer_position[1]**2))+(10.04*object_disposer_position[1])+(-34.02)
                        lower_border=(27.4)
                        if lower_border<object_disposer_position[0] and upper_border>object_disposer_position[0]:
                            if self.print_in_of_lines==0:
                                print "====+++ IN  BOUNDERIES +++===="
                                self.print_in_of_lines=1
                                self.print_out_of_lines=0
                                if self.flag_box_out==1:
                                    self.reward_for_back_to_lines=1

                            if self.object_disposer_out_of_line==1:
                                self.object_disposer_returned_to_lines_reward=1
                            self.object_disposer_out_of_line=0
                        else:
                            if self.print_out_of_lines==0: 
                                print "====+++ OUT OF BOUNDERIES - Part E-F +++===="
                                self.print_out_of_lines=1
                                self.print_in_of_lines=0    
                            self.object_disposer_out_of_line=1
                            self.object_disposer_returned_to_lines_reward=0

                            #self._episode_done = True

                #between F and G
                if object_disposer_position[1]> -19.5 and object_disposer_position[1]<=14.27 :
                    if object_disposer_position[1]<=14.27 and object_disposer_position[1]>9.85:
                        upper_border=(-0.02363*(object_disposer_position[1]**2))+(-0.2147*object_disposer_position[1])+(51.35)
                        lower_border=(-1.367*(object_disposer_position[1]**2))+(32.07*object_disposer_position[1])+(-152.6)
                    else:
                        if object_disposer_position[1]<=9.85 and object_disposer_position[1]>-16.0:
                            upper_border=(-0.02353*(object_disposer_position[1]**2))+(-0.2124*object_disposer_position[1])+(51.36)
                            lower_border=(-0.0598*(object_disposer_position[1]**2))+(-0.6872*object_disposer_position[1])+(43.19)
                        elif object_disposer_position[1]<=-16.0 and object_disposer_position[1]>-19.5:
                            upper_border=(0.04529*(object_disposer_position[1]**2))+(+2.346*object_disposer_position[1])+(74.96)
                            lower_border=(-0.1548*(object_disposer_position[1]**2))+(-4.205*object_disposer_position[1])+(12.68)
                    if lower_border<object_disposer_position[0] and upper_border>object_disposer_position[0]:
                        if self.print_in_of_lines==0:
                            print "====+++ IN  BOUNDERIES +++===="
                            self.print_in_of_lines=1
                            self.print_out_of_lines=0
                            if self.flag_box_out==1:
                                self.reward_for_back_to_lines=1

                        if self.object_disposer_out_of_line==1:
                            self.object_disposer_returned_to_lines_reward=1
                        self.object_disposer_out_of_line=0
                    else:
                        if self.print_out_of_lines==0: 
                            print "====+++ OUT OF BOUNDERIES - Part F-G +++===="
                            self.print_out_of_lines=1
                            self.print_in_of_lines=0
                        self.object_disposer_out_of_line=1
                        self.object_disposer_returned_to_lines_reward=0

                        #self._episode_done = True

                #between G and H
                if object_disposer_position[1]>-29.3 and object_disposer_position[1]<=-19.5 :
                    if object_disposer_position[0]>=27.4:
                        upper_border=(-0.1004*(object_disposer_position[1]**2))+(-3.362*object_disposer_position[1])+(19.20)
                        lower_border=(-0.5255*(object_disposer_position[1]**2))+(-19.38*object_disposer_position[1])+(-141.8)
                        if lower_border<27.4:
                            lower_border=27.4
                        if lower_border<object_disposer_position[0] and upper_border>object_disposer_position[0]:
                            if self.print_in_of_lines==0:
                                print "====+++ IN  BOUNDERIES +++===="
                                self.print_in_of_lines=1
                                self.print_out_of_lines=0
                                if self.flag_box_out==1:
                                    self.reward_for_back_to_lines=1

                            if self.object_disposer_out_of_line==1:
                                self.object_disposer_returned_to_lines_reward=1
                            self.object_disposer_out_of_line=0
                        else:
                            if self.print_out_of_lines==0: 
                                print "====+++ OUT OF BOUNDERIES - Part G-H +++===="
                                self.print_out_of_lines=1
                                self.print_in_of_lines=0
                            self.object_disposer_out_of_line=1
                            self.object_disposer_returned_to_lines_reward=0

                            #self._episode_done = True

        #crash between cars
        distance_line_follower_to_object_disposer = math.sqrt( ((line_follower_car_position[0]-object_disposer_position[0])**2)+((line_follower_car_position[1]-object_disposer_position[1])**2) )
        if distance_line_follower_to_object_disposer<=3.3:
            print "====+++ line follower and object disposer crashed! +++===="
            self._episode_done = True
            self.crash=1

        #crash between line follower car and object (box)
        #cheack if box infront of line follower car
        if ((np.isclose(object_box_position[0],line_follower_car_position[0],atol=1.75) ) and (np.isclose(object_box_position[1],line_follower_car_position[1],atol=1.15))) or ((np.isclose(object_box_position[1],line_follower_car_position[1],atol=1.75) ) and (np.isclose(object_box_position[0],line_follower_car_position[0],atol=1.15))):
            print " === line follower car SMASHED with the object box ===" 
            self.crash=1
            self._episode_done = True

        #object disposer robot collect the box.
        # if ((np.isclose(object_box_position[0],object_disposer_position[0],atol=1.55) ) and (np.isclose(object_box_position[1],object_disposer_position[1],atol=0.30))) or ((np.isclose(object_box_position[1],object_disposer_position[1],atol=1.55) ) and (np.isclose(object_box_position[0],object_disposer_position[0],atol=0.30))):
        #     if self.box_collected_flag==0:
        #         print " === object disposer robot COLLECTED the object box ==="
        #         self.box_collected_flag=1

        vector_object_disposert_to_object_box=[object_box_position[0]-object_disposer_position[0],object_box_position[1]-object_disposer_position[1]]
        distance_vector_object_disposert_to_object_box=math.sqrt((object_box_position[0]-object_disposer_position[0])**2+(object_box_position[1]-object_disposer_position[1])**2)
        norm_vector_object_disposert_to_object_box=[vector_object_disposert_to_object_box[0]/distance_vector_object_disposert_to_object_box,vector_object_disposert_to_object_box[1]/distance_vector_object_disposert_to_object_box]
        vector_x_axis=[1,0]
        vector_y_axis=[0,1]
        orientation_object_disposer_box_to_x_axis=np.absolute(np.arccos(np.dot(norm_vector_object_disposert_to_object_box,vector_x_axis)))
        orientation_object_disposer_box_to_y_axis=np.absolute(np.arccos(np.dot(norm_vector_object_disposert_to_object_box,vector_y_axis)))
        if orientation_object_disposer_box_to_x_axis>(np.pi/2.0):
           orientation_object_disposer_box_to_x_axis=np.absolute(orientation_object_disposer_box_to_x_axis-np.pi )

        if orientation_object_disposer_box_to_y_axis>(np.pi/2.0):
           orientation_object_disposer_box_to_y_axis=np.absolute(orientation_object_disposer_box_to_y_axis-np.pi )

        orient_x_object_disposer_robot=np.absolute(object_disposer_orentation[0])
        orient_y_object_disposer_robot=np.absolute(object_disposer_orentation[1])

        #===========================================================================
        j=9
        for i in range (0,4):
            j=i
            if all_boxes_position[i][0]> -39.1 and all_boxes_position[i][0]<27.4:
        #cheack if box in bounderies of yellow line in stright parts.
                if (all_boxes_position[i][1]> -29.3 and all_boxes_position[i][1]< -22.41) or (all_boxes_position[i][1]>14.27 and all_boxes_position[i][1]<21.2):
                    pass
                else:
                    
                    if self.box_1_out_print==0 and i ==0 and j==i:
                        print ("===Box number "+str(i+1)+"is OUT===")
                        self.box_1_out=1
                        self.box_1_out_print=1
                        self.box_1=1
                        self.flag_box_out=1
                        
                        
                    
                    if self.box_2_out_print==0 and i ==1 and j==i :
                        print ("===Box number "+str(i+1)+"is OUT===")
                        self.box_2_out=1
                        self.box_2_out_print=1
                        self.flag_box_out=1
                        
                    if self.box_3_out_print==0 and i ==2 and j==i :
                        print ("===Box number "+str(i+1)+"is OUT===")
                        self.box_3_out=1
                        self.box_3_out_print=1
                        self.flag_box_out=1
                        
                    if self.box_4_out_print==0 and i ==3 and j==i :
                        print ("===Box number "+str(i+1)+"is OUT===")
                        self.box_4_out=1
                        self.box_4_out_print=1
                        self.flag_box_out=1
            else:
                #zone --A--B--C--D--
                #lower_border=(-39.1)
                #upper_border_box=(0.03718*(all_boxes_position[i][1]**2))+(0.3011*all_boxes_position[i][1])+(-62.19)-3.9
                if all_boxes_position[i][0]<=-39.1:
                    #between A and B
                    if all_boxes_position[i][1]>14.27 and all_boxes_position[i][1]<22.5:
                        if all_boxes_position[i][0]<=-39.1:
                            upper_border_box=(0.03718*(all_boxes_position[i][1]**2))+(0.3011*all_boxes_position[i][1])+(-62.19)-3.9
                            lower_border_box=(-39.1)
                            if lower_border_box>all_boxes_position[i][0] and upper_border_box<all_boxes_position[i][0]:
                                pass
                            else:
                                if self.box_1_out_print==0 and i ==0 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_1_out=1
                                    self.box_1_out_print=1
                                    self.flag_box_out=1
                                    
                                    
                                if self.box_2_out_print==0 and i ==1 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_2_out=1
                                    self.box_2_out_print=1
                                    self.flag_box_out=1

                                if self.box_3_out_print==0 and i ==2 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_3_out=1
                                    self.box_3_out_print=1
                                    self.flag_box_out=1

                                if self.box_4_out_print==0 and i ==3 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_4_out=1
                                    self.box_4_out_print=1
                                    self.flag_box_out=1

                        #between B and C
                    if all_boxes_position[i][1]> -19.7 and all_boxes_position[i][1]<=14.27 :
                        upper_border_box=(0.0186*(all_boxes_position[i][1]**2))+(0.1736*all_boxes_position[i][1])+(-61.66)
                        lower_border_box=0.031*(all_boxes_position[i][1]**2)+0.232*all_boxes_position[i][1]+(-55.31)
                        if all_boxes_position[i][1] < 7.04 and all_boxes_position[i][1] > (-12.29) :
                            lower_border_box=0.031*(all_boxes_position[i][1]**2)+0.232*all_boxes_position[i][1]+(-55.31)
                        else:
                            if all_boxes_position[i][1]>=7.04 :
                                lower_border_box=0.1244*(all_boxes_position[i][1]**2)-1.144*all_boxes_position[i][1]+(-50.25)
                            elif all_boxes_position[i][1]<=-12.29 :
                                lower_border_box=0.153*(all_boxes_position[i][1]**2)+3.906*all_boxes_position[i][1]+(-28.94)
                            if lower_border_box>all_boxes_position[i][0] and upper_border_box<all_boxes_position[i][0]:
                                pass
                            else:
                                if self.box_1_out_print==0 and i ==0 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_1_out=1
                                    self.box_1_out_print=1
                                    self.flag_box_out=1
                                    
                                if self.box_2_out_print==0 and i ==1 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_2_out=1
                                    self.box_2_out_print=1
                                    self.flag_box_out=1

                                if self.box_3_out_print==0 and i ==2 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_3_out=1
                                    self.box_3_out_print=1
                                    self.flag_box_out=1

                                if self.box_4_out_print==0 and i ==3 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_4_out=1
                                    self.box_4_out_print=1
                                    self.flag_box_out=1

                    #beetwen C and D
                    lower_border_box=(-39.1)
                    upper_border_box=(0.03718*(all_boxes_position[i][1]**2))+(0.3011*all_boxes_position[i][1])+(-62.19)-3.9
                    if all_boxes_position[i][1]>-29.3 and all_boxes_position[i][1]<=-19.7 :
                        lower_border_box=0.953*(all_boxes_position[i][1]**2)+35.16*all_boxes_position[i][1]+(275.9)
                        if lower_border_box>=-39.1:
                            upper_border_box=(0.03718*(all_boxes_position[i][1]**2))+(0.3011*all_boxes_position[i][1])+(-62.19)-3.9
                            lower_border_box=(-39.1)
                        else:
                            upper_border_box=(0.03718*(all_boxes_position[i][1]**2))+(0.3011*all_boxes_position[i][1])+(-62.19)-3.9
                            lower_border_box=0.953*(all_boxes_position[i][1]**2)+35.16*all_boxes_position[i][1]+(275.9)
                            if lower_border_box>all_boxes_position[i][0] and upper_border_box<all_boxes_position[i][0]:
                                pass
                            else:
                                if self.box_1_out_print==0 and i ==0 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_1_out=1
                                    self.box_1_out_print=1
                                    self.flag_box_out=1
                                    
                                if self.box_2_out_print==0 and i ==1 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_2_out=1
                                    self.box_2_out_print=1
                                    self.flag_box_out=1

                                if self.box_3_out_print==0 and i ==2 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_3_out=1
                                    self.box_3_out_print=1
                                    self.flag_box_out=1

                                if self.box_4_out_print==0 and i ==3 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_4_out=1
                                    self.box_4_out_print=1
                                    self.flag_box_out=1

                #zone --E--F--G--H--
                elif all_boxes_position[i][0]>=27.4:
                    lower_border_box=(27.4)
                    #between E and F
                    if all_boxes_position[i][1]> 14.27 and all_boxes_position[i][1]<22.5 :
                        if all_boxes_position[i][0]>=27.4:
                            upper_border_box=(-0.3317*(all_boxes_position[i][1]**2))+(10.04*all_boxes_position[i][1])+(-34.02)
                            lower_border_box=(27.4)
                            if lower_border_box<all_boxes_position[i][0] and upper_border_box>all_boxes_position[i][0]:
                                pass
                            else:
                                if self.box_1_out_print==0 and i ==0 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_1_out=1
                                    self.box_1_out_print=1
                                    self.flag_box_out=1
                                    
                                if self.box_2_out_print==0 and i ==1 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_2_out=1
                                    self.box_2_out_print=1
                                    self.flag_box_out=1

                                if self.box_3_out_print==0 and i ==2 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_3_out=1
                                    self.box_3_out_print=1
                                    self.flag_box_out=1

                                if self.box_4_out_print==0 and i ==3 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_4_out=1
                                    self.box_4_out_print=1
                                    self.flag_box_out=1

                    #between F and G
                    if all_boxes_position[i][1]> -19.5 and all_boxes_position[i][1]<=14.27 :
                        if all_boxes_position[i][1]<=14.27 and all_boxes_position[i][1]>9.85:
                            upper_border_box=(-0.02363*(all_boxes_position[i][1]**2))+(-0.2147*all_boxes_position[i][1])+(51.35)
                            lower_border_box=(-1.367*(all_boxes_position[i][1]**2))+(32.07*all_boxes_position[i][1])+(-152.6)
                        else:
                            if all_boxes_position[i][1]<=9.85 and all_boxes_position[i][1]>-16.0:
                                upper_border_box=(-0.02353*(all_boxes_position[i][1]**2))+(-0.2124*all_boxes_position[i][1])+(51.36)
                                lower_border_box=(-0.0598*(all_boxes_position[i][1]**2))+(-0.6872*all_boxes_position[i][1])+(43.19)
                            elif all_boxes_position[i][1]<=-16.0 and all_boxes_position[i][1]>-19.5:
                                upper_border_box=(0.04529*(all_boxes_position[i][1]**2))+(+2.346*all_boxes_position[i][1])+(74.96)
                                lower_border_box=(-0.1548*(all_boxes_position[i][1]**2))+(-4.205*all_boxes_position[i][1])+(12.68)
                                if lower_border_box<all_boxes_position[i][0] and upper_border_box>all_boxes_position[i][0]:
                                    pass
                                else:
                                    if self.box_1_out_print==0 and i ==0 and j==i:
                                        print ("===Box number "+str(i+1)+"is OUT===")
                                        self.box_1_out=1
                                        self.box_1_out_print=1
                                        self.flag_box_out=1

                                        
                                    if self.box_2_out_print==0 and i ==1 and j==i:
                                        print ("===Box number "+str(i+1)+"is OUT===")
                                        self.box_2_out=1
                                        self.box_2_out_print=1
                                        self.flag_box_out=1

                                    if self.box_3_out_print==0 and i ==2 and j==i:
                                        print ("===Box number "+str(i+1)+"is OUT===")
                                        self.box_3_out=1
                                        self.box_3_out_print=1
                                        self.flag_box_out=1

                                    if self.box_4_out_print==0 and i ==3 and j==i:
                                        print ("===Box number "+str(i+1)+"is OUT===")
                                        self.box_4_out=1
                                        self.box_4_out_print=1
                                        self.flag_box_out=1
                    #between G and H
                    if all_boxes_position[i][1]>-29.3 and all_boxes_position[i][1]<=-19.5 :
                        if all_boxes_position[i][0]>=27.4:
                            upper_border_box=(-0.1004*(all_boxes_position[i][1]**2))+(-3.362*all_boxes_position[i][1])+(19.20)
                            lower_border_box=(-0.5255*(all_boxes_position[i][1]**2))+(-19.38*all_boxes_position[i][1])+(-141.8)
                            if lower_border_box<27.4:
                                lower_border_box=27.4
                            if lower_border_box<all_boxes_position[i][0] and upper_border_box>all_boxes_position[i][0]:
                                pass
                            else:
                                if self.box_1_out_print==0 and i ==0 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_1_out=1
                                    self.box_1_out_print=1
                                    self.flag_box_out=1
                                    
                                if self.box_2_out_print==0 and i ==1 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_2_out=1
                                    self.box_2_out_print=1
                                    self.flag_box_out=1

                                if self.box_3_out_print==0 and i ==2 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_3_out=1
                                    self.box_3_out_print=1
                                    self.flag_box_out=1

                                if self.box_4_out_print==0 and i ==3 and j==i:
                                    print ("===Box number "+str(i+1)+"is OUT===")
                                    self.box_4_out=1
                                    self.box_4_out_print=1
                                    self.flag_box_out=1

                            
        self.box_1=0
        if self.box_2_out==1 and self.box_3_out==1 and self.box_4_out==1 and self.box_1_out==1:
            self._episode_done = True

        


        


        #===========================================================================

        #print ("orient x object dis" ,orient_x_object_disposer_robot)
        #print ("orient box object dis" ,orientation_object_disposer_box_to_x_axis)

        # if (((np.isclose(object_box_position[0],object_disposer_position[0],atol=1.55) ) and (np.isclose(object_box_position[1],object_disposer_position[1],atol=0.29)) and (np.isclose(orient_x_object_disposer_robot,orientation_object_disposer_box_to_x_axis,atol=0.22))) or ((np.isclose(object_box_position[1],object_disposer_position[1],atol=1.55) ) and (np.isclose(object_box_position[0],object_disposer_position[0],atol=0.29)) and (np.isclose(orient_y_object_disposer_robot,orientation_object_disposer_box_to_y_axis,atol=0.22)))):
        #     if self.box_collected_flag==0:
        #         print " === object disposer robot COLLECTED the object box ==="
        #         self.box_collected_flag=1

         
                
        #if (((np.isclose(object_box_position[0],object_disposer_position[0],atol=1.57) ) and (np.isclose(object_box_position[1],object_disposer_position[1],atol=0.35)) and (np.isclose(orient_x_object_disposer_robot,orientation_object_disposer_box_to_x_axis,atol=0.28)))):
           # if self.box_collected_flag==0:
               # print " === object disposer robot COLLECTED the object box ==="
               # self.box_collected_flag=1
        #out of game boundeis so game over (reset)
        if (object_disposer_position[0]<-64) or (object_disposer_position[0]>64) or (object_disposer_position[1]>39) or (object_disposer_position[1]<-39):
            self.object_disposer_out_of_game_area=1 
            self._episode_done = True

        else:
            self.object_disposer_out_of_game_area=0

        #cheak if number of steps is to hight.
        if self.cumulated_steps > self.max_steps-2:
            self.steps_flag=1
            self._episode_done = True
            

        if self.white_area_steps_counter>self.white_area_steps_max-2:
            self.white_area_steps_flag=1
            self._episode_done=True
        




        return self._episode_done 

    def _compute_reward(self, observations, done):
        reward = 0.0
        if done:     
            if self.crash ==1:
                #reward=self.reward-1.0
                reward=-1.0
            if self.object_disposer_out_of_game_area==1:
                #reward=self.reward-2.5
                reward=-1.0 #-2.5
                print " === object disposer robot EXIT from area of the game ==="
            if self.box_1_out==1 and self.box_2_out==1 and self.box_3_out==1 and self.box_4_out==1:
                #reward=self.reward+3.0
                reward=1.0 #3.0
                print " === ALL BOXES OUT OF LINES ! ==="
            
            if self.steps_flag==1:
                #reward=self.reward-1.0
                reward=-0.5 #-1
                print " === Too Long Episode ! ==="

            if self.white_area_steps_flag==1:
                #reward=self.reward-1.0
                reward=-0.5 #-1
                print " === Too Much Time on White area ! ==="



                
            

            
            self.box_1_out_print=0
            self.box_2_out_print=0
            self.box_3_out_print=0
            self.box_4_out_print=0
            self.box_1_out=0
            self.box_2_out=0
            self.box_3_out=0
            self.box_4_out=0
            self.box_1_reward=0
            self.box_2_reward=0
            self.box_3_reward=0
            self.box_4_reward=0
            
            self.white_area_steps_flag=0
            self.white_area_steps_counter=0

            
            self.reward=0.0
            self.cumulated_steps=0
            self.steps_flag=0
            self.reward_for_back_to_lines=0
            self.flag_box_out=0              
        else:
            if self.object_disposer_out_of_line==1:
                #self.reward=self.reward-0.01
                reward=-0.001
                self.white_area_steps_counter=self.white_area_steps_counter+1
            
            if self.box_1_reward==0 and self.box_1_out==1:
                #self.reward=self.reward+2.0 #1.5
                reward=1.0 #2.0
                self.box_1_reward=1
            if self.box_2_reward==0 and self.box_2_out==1:
                #self.reward=self.reward+2.0 #1.5
                reward=1.0 #2.0
                self.box_2_reward=1 
            if self.box_3_reward==0 and self.box_3_out==1:
                #self.reward=self.reward+2.0 #1.5
                reward=1.0 #2.0
                self.box_3_reward=1 
            if self.box_4_reward==0 and self.box_4_out==1:
                #self.reward=self.reward+2.0 #1.5
                reward=1.0 #2.0
                self.box_4_reward=1
            
            if self.reward_for_back_to_lines==1:
                reward=0.5
                self.flag_box_out=0
                self.reward_for_back_to_lines=0
                print " === Reward for back to lines ! ==="
                

            
            


             
            #if self.object_disposer_returned_to_lines_reward==1:
             #   reward=reward+1
              #  self.object_disposer_returned_to_lines_reward=0   
                
            #negative reward from collection box moment until box out of lines.
            #if self.box_collected_flag == 1 and self.box_collected_reward_given==1:
                #reward=-0.0001
                #pass
        
        return reward


