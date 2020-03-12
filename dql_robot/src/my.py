#!/usr/bin/env python
import numpy as np 
import rospy
import time
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
from PIL import Image as IMG


#parameters for map building.
MIN_RANGE=1
MAX_RANGE=50
MIN_ANGLE=-1.570796
NUM_OF_MEASURMENTS=720
MAP_SIZE_EDITION=10

import matplotlib.pyplot as plt

def build_colored_matrix(x,y):
    matrix_map=np.zeros(((2*MAX_RANGE+MAP_SIZE_EDITION),(2*(MAX_RANGE+MAP_SIZE_EDITION))))
    bw_image = np.zeros((2*(MAX_RANGE+MAP_SIZE_EDITION),(2*(MAX_RANGE+MAP_SIZE_EDITION))),np.uint8)
    robot_position=[0,MAX_RANGE+MAP_SIZE_EDITION] #60
    
    

    matrix_map[0][MAX_RANGE+MAP_SIZE_EDITION]=255
    bw_image[0][MAX_RANGE+MAP_SIZE_EDITION]=255
    #x = np.rint(x)
    #y = np.rint(y)
    x= np.array(x, dtype='int')
    y= np.array(y, dtype='int')
    #print y

    #print x
    for i in range(0,NUM_OF_MEASURMENTS):
        for j in range(0,NUM_OF_MEASURMENTS):
            if x[i]>=1 and x[i]<=50 and ((y[j]>=-50 and y[j]<=-1) or (y[j]<=50 and y[j]>=1)):
                matrix_map[x[i]+robot_position[0]][y[j]+robot_position[1]]=255
                bw_image[x[i]+robot_position[0]][y[j]+robot_position[1]]=255
                #bw_image=cv2.line(bw_image,(x[i]+robot_position[0],y[j]+robot_position[1]),(robot_position[0],robot_position[1]),255,1)
                #cv2.imshow("BW Image",bw_image)
                #cv2.waitKey(1)
            elif y[j]>50 or y[j]<-50:
                matrix_map[x[i]+robot_position[0]][y[j]+robot_position[1]]=100
                bw_image[x[i]+robot_position[0]][y[j]+robot_position[1]]=100
   
    #matrix_map[0:256, 0:256] = 0 # red patch in upper left
    
 
    #print matrix_map
    cv2.imshow("BW Image",bw_image)
    cv2.waitKey(1)
    
   
    
       
    #print matrix_map       
            
            #matrix_map[i+robot_position[0]][j+robot_position[1]]=


def clbk_laser(msg):
    # 720 / 5 = 144
    #print "=================================="
    #print(msg.ranges)
    angle=np.zeros(NUM_OF_MEASURMENTS)
    angle[0]=MIN_ANGLE
    radius=np.zeros(NUM_OF_MEASURMENTS)
    for i in range(1,NUM_OF_MEASURMENTS):
        angle[i]=angle[i-1]+msg.angle_increment

    for i in range(0,NUM_OF_MEASURMENTS):
        if msg.ranges[i]<=MAX_RANGE:
            radius[i]=msg.ranges[i]
        else:
            radius[i]=0
    



    
    #print angle
    #print(radius)
    
    #print msg.ranges[360]
   # print angle[360]
    
    x=np.zeros(NUM_OF_MEASURMENTS)
    y=np.zeros(NUM_OF_MEASURMENTS)

    for i in range(0,NUM_OF_MEASURMENTS):
        x[i]=radius[i]*np.cos(angle[i])
        y[i]=radius[i]*np.sin(angle[i])

    
    
   
    ##plt.scatter(x, y,alpha=0.6)
    ##plt.xlim(0,(MAX_RANGE+MAP_SIZE_EDITION))
    ##plt.ylim((-MAX_RANGE-MAP_SIZE_EDITION),(MAX_RANGE+MAP_SIZE_EDITION))
    #plt.axis("equal")
    ##plt.draw()
    ##plt.pause(0.0000000000000000000000000000001)
    ##plt.cla()
    build_colored_matrix(x,y) 

    
    
    #print robot_position
    #print matrix_map
    


    
        
    



rospy.init_node('my')
print "++++++++++++++++++++++++++++++++"
rospy.Subscriber("/object_disposer_robot/scan", LaserScan, clbk_laser, queue_size=1)
#plt.ion()
#plt.show()
rospy.spin()
 


