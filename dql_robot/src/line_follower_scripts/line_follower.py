#!/usr/bin/env python


#This Program is tested on Gazebo Simulator
#This script uses the cv_bridge package to convert images coming on the topic
#sensor_msgs/Image to OpenCV messages and then convert their colors from RGB to HSV
#then apply a threshold for hues near the color yellow to obtain the binary image
#to be able to see only the yellow line and then follow that line
#It uses an approach called proportional and simply means

import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import time

class Follower:

        def __init__(self):
                
                self.bridge = cv_bridge.CvBridge()
                #cv2.namedWindow("window", 1)

                self.image_sub = rospy.Subscriber('/line_follower_car/front_camera/image_raw',
                        Image, self.image_callback_line_follower)
                #print "check"
                self.sub=rospy.wait_for_message('/line_follower_car/front_camera/image_raw',
                        Image)
                #print "check"
                self.cmd_vel_pub = rospy.Publisher('/line_follower_car/cmd_vel_car',
                        Twist, queue_size=1)

                self.twist = Twist()

        def image_callback_line_follower(self, msg):
                #print "check"
                image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                #cv2.imshow("window", image)
                lower_yellow = numpy.array([ 10, 10, 10])
                upper_yellow = numpy.array([220, 245, 90])



                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

                h, w, d = image.shape
                search_top = 3*h/4
                search_bot = 3*h/4 + 20
                mask[0:search_top, 0:w] = 0
                mask[search_bot:h, 0:w] = 0

                M = cv2.moments(mask)
                #print "c1"
                #print M['m00']
                if M['m00'] > 0:
                        #print "i did it!"
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        cv2.circle(image, (cx, cy), 20, (0,0,255), -1)
#The proportional controller is implemented in the following four lines which
#is reposible of linear scaling of an error to drive the control output.
                        err = cx - w/2
                        self.twist.linear.x = 10.0
                        self.twist.angular.z = -float(err) / 12
                        self.cmd_vel_pub.publish(self.twist)
                        time.sleep(0.025)
                #cv2.imshow("window", image)
                #cv2.waitKey(3)

rospy.init_node('line_follower')
follower = Follower()
rospy.spin()
