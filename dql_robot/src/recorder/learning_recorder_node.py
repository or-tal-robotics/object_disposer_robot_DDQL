#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Int16

episode_count = 0
bridge = CvBridge()
ii_robot1 = 1
ii_robot2 = 1
ii_robot3 = 1
ii_prey = 1
jj = 1
out_robot1 = []
out_robot2 = []
out_robot3 = []
out_prey = []
out_gazebo = []
#pathOut_robot1 = '/home/or/openai_ws_catch/src/gym-openai-multirobot-catch/dql_robot/video/robot1/episode'
#pathOut_robot2 = '/home/or/openai_ws_catch/src/gym-openai-multirobot-catch/dql_robot/video/robot2/episode'
#pathOut_robot3 = '/home/or/openai_ws_catch/src/gym-openai-multirobot-catch/dql_robot/video/robot3/episode'
#pathOut_prey = '/home/or/openai_ws_catch/src/gym-openai-multirobot-catch/dql_robot/video/prey/episode'
pathOut_gazebo = '/home/lab/igal_ws/src/object_disposer_robot_DDQL/dql_robot/video'
record_frq = 10

def counter_cb(msg):
    global episode_count
    episode_count = msg.data

# def robot1_image_callback(msg):
#     global episode_count, bridge, pathOut_robot1, ii_robot1, out_robot1, record_frq
#     if (episode_count % record_frq) == 0:
#         img = bridge.imgmsg_to_cv2(msg,"rgb8")
#         file_name = pathOut_robot1+str(episode_count)+'.avi'
#         if ii_robot1 == 1:
#             print('Start recording robot1 camera!, episode:'+str(episode_count))
#             out_robot1 = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (640,480))
#             ii_robot1 = 0
#         img = cv2.resize(img,(640,480))
#         out_robot1.write(img)

#     if ((episode_count-1) % record_frq) == 0 and ii_robot1 == 0:
#         print("Finish recording robot camera!")
#         out_robot1.release()
#         ii_robot1 = 1

# def robot2_image_callback(msg):
#     global episode_count, bridge, pathOut_robot2, ii_robot2, out_robot2, record_frq
#     if (episode_count % record_frq) == 0:
#         img = bridge.imgmsg_to_cv2(msg,"rgb8")
#         file_name = pathOut_robot2+str(episode_count)+'.avi'
#         if ii_robot2 == 1:
#             print('Start recording robot2 camera!, episode:'+str(episode_count))
#             out_robot2 = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (640,480))
#             ii_robot2 = 0
#         img = cv2.resize(img,(640,480))
#         out_robot2.write(img)

#     if ((episode_count-1) % record_frq) == 0 and ii_robot2 == 0:
#         print("Finish recording robot camera!")
#         out_robot2.release()
#         ii_robot2 = 1

# def robot3_image_callback(msg):
#     global episode_count, bridge, pathOut_robot3, ii_robot3, out_robot3, record_frq
#     if (episode_count % record_frq) == 0:
#         img = bridge.imgmsg_to_cv2(msg,"rgb8")
#         file_name = pathOut_robot3+str(episode_count)+'.avi'
#         if ii_robot3 == 1:
#             print('Start recording robot3 camera!, episode:'+str(episode_count))
#             out_robot3 = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (640,480))
#             ii_robot3 = 0
#         img = cv2.resize(img,(640,480))
#         out_robot3.write(img)

#     if ((episode_count-1) % record_frq) == 0 and ii_robot3 == 0:
#         print("Finish recording robot camera!")
#         out_robot3.release()
#         ii_robot3 = 1

# def prey_image_callback(msg):
#     global episode_count, bridge, pathOut_prey, ii_prey, out_prey, record_frq
#     if (episode_count % record_frq) == 0:
#         img = bridge.imgmsg_to_cv2(msg,"rgb8")
#         file_name = pathOut_prey+str(episode_count)+'.avi'
#         if ii_prey == 1:
#             print('Start recording prey camera!, episode:'+str(episode_count))
#             out_prey = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (640,480))
#             ii_prey = 0
#         img = cv2.resize(img,(640,480))
#         out_prey.write(img)

#     if ((episode_count-1) % record_frq) == 0 and ii_prey == 0:
#         print("Finish recording robot camera!")
#         out_prey.release()
#         ii_prey = 1


def gazebo_image_callback(msg):
    global episode_count, bridge, pathOut_gazebo, jj, out_gazebo, record_frq
    if (episode_count % record_frq) == 0:
        img = bridge.imgmsg_to_cv2(msg,"rgb8")
        file_name = pathOut_gazebo+str(episode_count)+'.avi'
        if jj == 1:
            print('Start recording gazebo camera!, episode:'+str(episode_count))
            out_gazebo = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (640,480))
            jj = 0
        img = cv2.resize(img,(640,480))
        out_gazebo.write(img)

    if ((episode_count-1) % record_frq) == 0 and jj == 0:
        print("Finish recording gazebo camera!")
        out_gazebo.release()
        jj = 1


if __name__ == '__main__':
    print("Initializing recorder node....")
    rospy.init_node("learning_recorder_node")
    rospy.Subscriber('/episode_counter',Int16, counter_cb)
    #rospy.Subscriber("/robot1/camera/rgb/image_raw", Image, robot1_image_callback)
    #rospy.Subscriber("/robot2/camera/rgb/image_raw", Image, robot2_image_callback)
    #rospy.Subscriber("/robot3/camera/rgb/image_raw", Image, robot3_image_callback)
    #rospy.Subscriber("/prey/camera/image_raw", Image, prey_image_callback)
    
    rospy.Subscriber("/gazebo/image_raw", Image, gazebo_image_callback)
    rospy.wait_for_message("/gazebo/image_raw", Image)
    #rospy.wait_for_message("/robot1/camera/rgb/image_raw", Image)
    rospy.wait_for_message('/episode_counter',Int16)
    print("Finish initializing, Starts recording!")
    rospy.spin()

