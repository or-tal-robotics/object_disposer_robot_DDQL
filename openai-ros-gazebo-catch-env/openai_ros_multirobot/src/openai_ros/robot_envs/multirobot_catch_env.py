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
import matplotlib.pyplot as plt

from skimage import transform as tf


#parameters for map building.
MIN_RANGE=1
MAX_RANGE=50
MIN_ANGLE=-1.570796
NUM_OF_MEASURMENTS=720
MAP_SIZE_EDITION=10
IMAGE_SIZE=100

OBJ_COLOR=2500 #2500
OBJ_SEROUND_COLOR=2000 #2000
PASS_COLOR=1500 #1000
RANGE_COLOR=500 #1100

X_grid=50
Y_grid=100

def get_line(start, end):
        """Bresenham's Line Algorithm
        Produces a list of tuples from start and end
        """
        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
    
        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)
    
        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
    
        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
    
        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1
    
        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1
    
        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
    
        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        return points

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
        
        #subscribe to front and top camera of object disposer robot
        rospy.Subscriber("/object_disposer_robot/front_camera/image_raw",Image, self._camera_rgb_image_raw_callback_object_disposer_car)
        rospy.Subscriber("/object_disposer_robot/top_camera/image_raw",Image, self._camera_rgb_image_raw_callback_object_disposer_car_top_camera)
        
        #subscribe to laser scanner of object disposer robot
        rospy.Subscriber("/object_disposer_robot/scan", LaserScan, self._clbk_laser_object_disposer_robot, queue_size=1)
                
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
        object_box_2_idx=models.index('object_box_2')
        object_box_3_idx=models.index('object_box_3')
        object_box_4_idx=models.index('object_box_4')

        #self.predator_position = [msg.pose[predator_idx].position.x, msg.pose[predator_idx].position.y]
        #self.prey_position = [msg.pose[prey_idx].position.x, msg.pose[prey_idx].position.y]
        self.line_follower_car_position = [msg.pose[line_follower_car_idx].position.x, msg.pose[line_follower_car_idx].position.y]
        self.object_disposer_robot_position=[msg.pose[object_disposer_robot_idx].position.x, msg.pose[object_disposer_robot_idx].position.y]
        
        self.object_box_position=[msg.pose[object_box_idx].position.x, msg.pose[object_box_idx].position.y]
        self.object_box_orientation=[msg.pose[object_box_idx].orientation.x, msg.pose[object_box_idx].orientation.y]

        self.object_box_2_position=[msg.pose[object_box_2_idx].position.x, msg.pose[object_box_2_idx].position.y]
        self.object_box_2_orientation=[msg.pose[object_box_2_idx].orientation.x, msg.pose[object_box_2_idx].orientation.y]

        self.object_box_3_position=[msg.pose[object_box_3_idx].position.x, msg.pose[object_box_3_idx].position.y]
        self.object_box_3_orientation=[msg.pose[object_box_3_idx].orientation.x, msg.pose[object_box_3_idx].orientation.y]

        self.object_box_4_position=[msg.pose[object_box_4_idx].position.x, msg.pose[object_box_4_idx].position.y]
        self.object_box_4_orientation=[msg.pose[object_box_4_idx].orientation.x, msg.pose[object_box_4_idx].orientation.y]

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
    
    
    
    #laser scans data from object disposer robot,
    #it converts from laser scans to occupancy grid image.
    def _clbk_laser_object_disposer_robot(self,data):
        #print (data.ranges)
        angle=np.zeros(NUM_OF_MEASURMENTS)
        angle[0]=MIN_ANGLE
        radius=np.zeros(NUM_OF_MEASURMENTS)
        for i in range(1,NUM_OF_MEASURMENTS):
            angle[i]=angle[i-1]+data.angle_increment

        for i in range(0,NUM_OF_MEASURMENTS):
            if data.ranges[i]<=MAX_RANGE:
                radius[i]=data.ranges[i]
            else:
                radius[i]=MAX_RANGE-1
                #radius[i]=0
        self.radius=radius

        
        #print(radius)
        x=np.zeros(NUM_OF_MEASURMENTS)
        y=np.zeros(NUM_OF_MEASURMENTS)
        #check if it measurment or max range value 1-max range  0-measurment
        x_max_range=np.zeros(NUM_OF_MEASURMENTS)
        y_max_range=np.zeros(NUM_OF_MEASURMENTS)

        for i in range(0,NUM_OF_MEASURMENTS):
            if radius[i]==MAX_RANGE-1:
                x_max_range[i]=1
                y_max_range[i]=1
            
            x[i]=radius[i]*np.cos(angle[i])
            y[i]=radius[i]*np.sin(angle[i])
        #coordinate_matrix=np.zeros((MAX_RANGE,MAX_RANGE*2))
        coordinate_matrix=np.zeros((MAX_RANGE,MAX_RANGE*2))
        x_round=np.rint(x)
        y_round=np.rint(y)

        #for i in range (0,MAX_RANGE):
            #for j in range(0,MAX_RANGE*2):
                #a=np.array((i ,j, 0))
                #b=np.array((0 ,MAX_RANGE, 0))
                #if np.linalg.norm(a-b)<MAX_RANGE :
                    #coordinate_matrix[i,j]=RANGE_COLOR 


        coordinate_matrix[0,MAX_RANGE]=1000 #set robot location on map

        #set the location of the objects and the serounding area on map
        for i in range(0,NUM_OF_MEASURMENTS):
            
            #if (not (x[i]==0 and y[i]==0) and int(x_round[i])+2<MAX_RANGE and int(y_round[i])+2<MAX_RANGE*2) and x_max_range[i]==0 and y_max_range[i]==0 :
            if (not (x[i]==0 and y[i]==0) and int(x_round[i])+2<MAX_RANGE and int(y_round[i])+2<MAX_RANGE*2) and x_max_range[i]==0 and y_max_range[i]==0 :
                if y[i]>=0:
                    coordinate_matrix[int(x_round[i]),int(MAX_RANGE+y_round[i])]=OBJ_COLOR
                else:
                    coordinate_matrix[int(x_round[i]),int(MAX_RANGE+y_round[i])]=OBJ_COLOR

                if int(x_round[i])-1>=0:
                    coordinate_matrix[int(x_round[i]-1),int(MAX_RANGE+y_round[i])]=OBJ_SEROUND_COLOR
                    if int(MAX_RANGE+y_round[i])-1>=-MAX_RANGE:
                        coordinate_matrix[int(x_round[i]-1),int(MAX_RANGE+y_round[i]-1)]=OBJ_SEROUND_COLOR
                    if int(MAX_RANGE+y_round[i])+1<=MAX_RANGE:
                        coordinate_matrix[int(x_round[i]-1),int(MAX_RANGE+y_round[i]+1)]=OBJ_SEROUND_COLOR

                if int(x_round[i])+1<MAX_RANGE and int(y_round[i])+1<MAX_RANGE*2:
                    coordinate_matrix[int(x_round[i]+1),int(MAX_RANGE+y_round[i])]=OBJ_SEROUND_COLOR
                    if int(MAX_RANGE+y_round[i])-1>=-MAX_RANGE:
                        coordinate_matrix[int(x_round[i]+1),int(MAX_RANGE+y_round[i]-1)]=OBJ_SEROUND_COLOR
                    if int(MAX_RANGE+y_round[i])+1<=MAX_RANGE:
                        coordinate_matrix[int(x_round[i]+1),int(MAX_RANGE+y_round[i]+1)]=OBJ_SEROUND_COLOR

                if int(MAX_RANGE+y_round[i])-1>=-MAX_RANGE:
                    coordinate_matrix[int(x_round[i]),int(MAX_RANGE+y_round[i]-1)]=OBJ_SEROUND_COLOR
                if int(MAX_RANGE+y_round[i])+1<=MAX_RANGE:
                    coordinate_matrix[int(x_round[i]),int(MAX_RANGE+y_round[i]+1)]=OBJ_SEROUND_COLOR
            
            if x_max_range[i]==1 and y_max_range[i]==1 :
                coordinate_matrix[int(x_round[i]),int(MAX_RANGE+y_round[i])]=RANGE_COLOR
            

            

        

               
                


            

        t=0
        points_to_object=[]
        point_to_maybe=[]
        for i in range (0,MAX_RANGE):
            for j in range(0,MAX_RANGE*2):
                if (coordinate_matrix[i,j]==OBJ_COLOR) : 
                    points_to_object= get_line((i,j),(0,MAX_RANGE))
                    for xx in range(1,MAX_RANGE-1):
                        for yy in range(1,MAX_RANGE*2-1):
                            t=0
                            while points_to_object[t][0] :
                                if points_to_object[t][0]==xx and points_to_object[t][1]==yy :
                                    if coordinate_matrix[xx,yy]!=OBJ_SEROUND_COLOR:
                                        coordinate_matrix[xx,yy]=PASS_COLOR
                                    if coordinate_matrix[xx,yy+1]!=OBJ_SEROUND_COLOR:
                                        coordinate_matrix[xx,yy+1]=PASS_COLOR
                                    if coordinate_matrix[xx,yy-1]!=OBJ_SEROUND_COLOR:
                                        coordinate_matrix[xx,yy-1]=PASS_COLOR
                                t=t+1

                
                
        
                
                
        
        coordinate_matrix_resize=np.zeros((64,64))
        
        
        coordinate_matrix_resize=tf.resize(coordinate_matrix,(64,64))
        self.coordinate_matrix=coordinate_matrix_resize
        
        

        
        #plt.imshow(coordinate_matrix_resize,cmap='jet')
        #plt.draw()
        #plt.pause(0.0000000000000000000000000000001)
        #plt.cla()






        
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

    #new addition 10.3.20
    def get_laser_image(self):
        return self.coordinate_matrix

            

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

    def get_object_box_2_position(self):
        return self.object_box_2_position

    def get_object_box_2_orientation(self):
        return self.object_box_2_orientation

    def get_object_box_3_position(self):
        return self.object_box_3_position

    def get_object_box_3_orientation(self):
        return self.object_box_3_orientation

    def get_object_box_4_position(self):
        return self.object_box_4_position

    def get_object_box_4_orientation(self):
        return self.object_box_4_orientation