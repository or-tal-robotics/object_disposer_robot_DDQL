import numpy as np 
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from openai_ros.openai_ros_common import ROSLauncher
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image


class TurtleBot2catchEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, ros_ws_abspath):
        
        self.bridge = CvBridge()
        rospy.logdebug("Start TurtleBot2catchEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # We launch the ROSlaunch that spawns the robot into the world
        
        #ROSLauncher(rospackage_name="dql_robot",
         #           launch_file_name="old_not_mine/put_robots_in_world.launch",
          #          ros_ws_abspath=ros_ws_abspath)


        ROSLauncher(rospackage_name="dql_robot",
                    launch_file_name="put_line_follower_car_in_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        ROSLauncher(rospackage_name="dql_robot",
                    launch_file_name="put_object_disposer_robot_in_world.launch",
                    ros_ws_abspath=ros_ws_abspath)



        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(TurtleBot2catchEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")




        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        
        
        #subscribe to front camera of line follower robot
        rospy.Subscriber("/line_follower_car/front_camera/image_raw",Image,self._camera_rgb_image_raw_callback_line_follower_car)
        #subscribe to front camera of object disposer robot
        rospy.Subscriber("/object_disposer_robot/front_camera/image_raw",Image, self._camera_rgb_image_raw_callback_object_disposer_car)
        rospy.Subscriber("/object_disposer_robot/top_camera/image_raw",Image, self._camera_rgb_image_raw_callback_object_disposer_car_top_camera)
                
        #publuish speed to line follower robot
        self._cmd_vel_pub_line_follower_car=rospy.Publisher('/line_follower_car/cmd_vel_car',
                        Twist, queue_size=1)
        self.twist=Twist()
        #publish speed to object_disposer car
        self._cmd_vel_pub_object_disposer_robot=rospy.Publisher('/object_disposer_robot/cmd_vel_car',
                        Twist, queue_size=1)
        
        rospy.Subscriber("/gazebo/model_states", ModelStates ,self._model_state_callback)

        

        

        

        self._check_publishers_connection()

        self.gazebo.pauseSim()
        
        rospy.logdebug("Finished TurtleBot2Env INIT...")
        


    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    
    def _model_state_callback(self,msg):
        models = msg.name
        #predator_idx = models.index('predator')
        #prey_idx = models.index('prey')
        
        object_disposer_robot_idx=models.index('object_disposer_robot')
        line_follower_car_idx=models.index('line_follower_car')
        object_box_idx=models.index('object_box')

        #self.predator_position = [msg.pose[predator_idx].position.x, msg.pose[predator_idx].position.y]
        #self.prey_position = [msg.pose[prey_idx].position.x, msg.pose[prey_idx].position.y]
        self.line_follower_car_position = [msg.pose[line_follower_car_idx].position.x, msg.pose[line_follower_car_idx].position.y]
        self.object_disposer_robot_position=[msg.pose[object_disposer_robot_idx].position.x, msg.pose[object_disposer_robot_idx].position.y]
        self.object_box_position=[msg.pose[object_box_idx].position.x, msg.pose[object_box_idx].position.y]
        self.object_box_orientation=[msg.pose[object_box_idx].orientation.x, msg.pose[object_box_idx].orientation.y]

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        rospy.logdebug("ALL SENSORS READY")


    #def _front_camera_rgb_image_raw_callback_prey(self, data):
     #   self.front_camera_rgb_image_raw_prey = self.bridge.imgmsg_to_cv2(data,"rgb8")
    
    #def _back_camera_rgb_image_raw_callback_prey(self, data):
     #   self.back_camera_rgb_image_raw_prey = self.bridge.imgmsg_to_cv2(data,"rgb8")

    
            
    #def _camera_rgb_image_raw_callback_predator(self, data):
     #       self.camera_rgb_image_raw_predator = self.bridge.imgmsg_to_cv2(data,"rgb8")

    def _camera_rgb_image_raw_callback_object_disposer_car(self, data):
            self.camera_rgb_image_raw_object_disposer_car = self.bridge.imgmsg_to_cv2(data,"rgb8")
    
    def _camera_rgb_image_raw_callback_object_disposer_car_top_camera(self, data):
            self.camera_rgb_image_raw_object_disposer_car_top_camera = self.bridge.imgmsg_to_cv2(data,"rgb8")

    def _camera_rgb_image_raw_callback_line_follower_car(self, data):
            self.camera_rgb_image_raw_line_follower_car = self.bridge.imgmsg_to_cv2(data,"rgb8")
            hsv = cv2.cvtColor(self.camera_rgb_image_raw_line_follower_car, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([ 10, 10, 10])
            upper_yellow = np.array([220, 245, 90])

            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            h, w, d = self.camera_rgb_image_raw_line_follower_car.shape
            search_top = 3*h/4
            search_bot = 3*h/4 + 20
            mask[0:search_top, 0:w] = 0
            mask[search_bot:h, 0:w] = 0

            M = cv2.moments(mask)
            if M['m00'] > 0:
                    #print "i did it!"
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.circle(self.camera_rgb_image_raw_line_follower_car, (cx, cy), 20, (0,0,255), -1)
#The proportional controller is implemented in the following four lines which
#is reposible of linear scaling of an error to drive the control output.
                    err = cx - w/2
                    #self.twist.linear.x = 10.0
                    #self.twist.angular.z = -float(err) / 12
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 0.0
                    
                    self._cmd_vel_pub_line_follower_car.publish(self.twist)
                    time.sleep(0.025)
    
    #def _right_camera_rgb_image_raw_callback_predator(self, data):
     #   self.right_camera_rgb_image_raw_predator = self.bridge.imgmsg_to_cv2(data,"rgb8")

    #def _LaserScan_callback_predator(self, data):
     #   self.LaserScan_predator = data.ranges

        
    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        #while (self._cmd_vel_pub_predator.get_num_connections() == 0 or self._cmd_vel_pub_prey.get_num_connections() == 0 ) and not rospy.is_shutdown():
        while (self._cmd_vel_pub_object_disposer_robot.get_num_connections() == 0) and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_base(self, linear_speed, angular_speed,sleep_time = 0.1, epsilon=0.05, update_rate=10, min_laser_distance=-1):
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("TurtleBot2 Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()

        self._cmd_vel_pub_object_disposer_robot.publish(cmd_vel_value)
        
    def get_camera_rgb_image_raw(self):
        return self.camera_rgb_image_raw_object_disposer_car

    def get_camera_rgb_image_raw_top_camera(self):
        return self.camera_rgb_image_raw_object_disposer_car_top_camera

            

  #  def get_prey_position(self):
  #      return self.prey_position

  #  def get_predator_position(self):
   #     return self.predator_position

    def get_object_disposer_robot_position(self):
        return self.object_disposer_robot_position

    def get_line_follower_car_position(self):
        return self.line_follower_car_position

    def get_object_box_position(self):
        return self.object_box_position

    def get_object_box_orientation(self):
        return self.object_box_orientation