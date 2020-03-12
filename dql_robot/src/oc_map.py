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

from matplotlib import colors


#parameters for map building.
MIN_RANGE=1
MAX_RANGE=50
MIN_ANGLE=-1.570796
NUM_OF_MEASURMENTS=720
MAP_SIZE_EDITION=10
IMAGE_SIZE=100

X_grid=50
Y_grid=100

import matplotlib.pyplot as plt

def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
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

def mat_maker(size_x,size_y,x,y):
    #set matrix size of [size_x X size_y]
    mat=np.zeros[(size_x,size_y)]
    #set robot position
    mat[0][size_y/2]=200
    x=(np.around(x))
    y=(np.around(y))
    for i in range(0,len(x)):
        x[i]=int(x[i])
        y[i]=int(y[i])
    

    return mat



def samplemat(size_x,size_y,x,y):
    """Make a matrix with all zeros and increasing elements on the diagonal"""
    aa = np.zeros((size_x+20,size_y+20))
    x=(np.around(x))
    y=(np.around(y))
    for i in range(0,len(x)):
        x[i]=int(x[i])+10
        y[i]=int(y[i])+10
        

    for i in range (0,len(x)):
        for j in range(0,len(y)):
            aa[int(x[i])][int(y[j]+60)]=200
    
    
    return aa

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

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

    #np.random.seed(1977)
    #x, y, z = np.random.random((3, 10))

# Bin the data onto a 10x10 grid
# Have to reverse x & y due to row-first indexing
    # x_lim=np.zeros(50)
    # x_lim[49]=MAX_RANGE+MAP_SIZE_EDITION
    # y_lim=np.zeros(50)
    # y_lim[49]=MAX_RANGE+MAP_SIZE_EDITION
    # y_lim[0]=-MAX_RANGE-MAP_SIZE_EDITION
    # zi, yi, xi = np.histogram2d(y_lim, x_lim, bins=(10,10), normed=False)
    # zi = np.ma.masked_equal(zi, 0)

    # #ax = plt.subplots()
    # #ax.pcolormesh(xi, yi, zi, edgecolors='black')
    # #scat = ax.scatter(x, y, s=200)
    # #fig.colorbar(scat)
    # #ax.margins(0.05)

    # #plt.show()
    
    #plt.rc('grid', linestyle="-", color='black')
    #plt.imshow()
    
    #plt.xlim(0,(MAX_RANGE+MAP_SIZE_EDITION))
    #plt.ylim((-MAX_RANGE-MAP_SIZE_EDITION),(MAX_RANGE+MAP_SIZE_EDITION))
     #plt.axis("equal")
     #plt.pcolormesh(xi,yi,zi, edgecolors='black')
    #plt.draw()
    #plt.pause(0.0000000000000000000000000000001)
    #plt.cla()
    #plt.grid(True)
    # #build_colored_matrix(x,y) 

# define grid

   

    
    # Display matrix
    #plt.ion()
    #cv2.imshow("ff",samplemat(IMAGE_SIZE, IMAGE_SIZE,x,y))
    #cv2.waitKey(1)
    #plt.matshow(samplemat(IMAGE_SIZE, IMAGE_SIZE,x,y), fignum=0)
    #plt.show()
    ##plt.draw()
    ##plt.pause(0.000000000000000000000000000000000000001)
    ##plt.cla()
    #plt.pause(0.0000000000000000000000000000001)
    #plt.close()
    
    #print robot_position
    #print matrix_map
    
    # make it by radius and angle
    #plt.hist2d(radius,angle)


    #plt.scatter(x,y)
    #plt.xlim(0,(MAX_RANGE+MAP_SIZE_EDITION))
    #plt.ylim((-MAX_RANGE-MAP_SIZE_EDITION),(MAX_RANGE+MAP_SIZE_EDITION))
    #plt.axis("equal")
    #plt.draw()
    #plt.grid(True)
    #plt.pause(0.0000000000000000000000000000001)
    #plt.cla()


    
    
    #plt.matshow(samplemat(IMAGE_SIZE, IMAGE_SIZE,x,y), fignum=0)
    #plt.show()
    #plt.draw()
    #plt.pause(0.0000000000000000000000000000001)
    #plt.cla()
    #plt.pause(0.0000000000000000000000000000001)
    #plt.close()


    #mat=np.zeros((X_grid,Y_grid))
    #delta_x=MAX_RANGE/X_grid
    #delta_y=MAX_RANGE/Y_grid
    #robot_pos=[0,Y_grid/2]
    #y_lim=[-50,50]
    #x_lim=[0,50]

    #inside=[]
    
    #for i in range(0,len(x)):
    #    x[i]=int(np.floor(x[i]))
    #    y[i]=int(np.floor(y[i]))
    #z=np.ones(len(x))
    #z_array = np.nan * np.empty((720,720))
    #z_array[y,x]  = z

    #print(z_array)


    # xx=0
    # yy=0

    # for x_count in range(0,len(x)):
    #     for y_count in range(0,len(y)):
    #         xx=int(np.around(x[x_count]))
    #         yy=int(np.around(y[y_count]))
    #         if (xx<x[x_count] and x[x_count]<xx+1) and (yy<np.abs(y[y_count]) and np.abs(y[y_count])<yy+1):
    #             if 0<y[y_count]:
    #                 mat[xx][Y_grid/2+yy]=100
    #             else:
    #                  mat[xx][Y_grid/2-yy]=100
            
    
    # plt.matshow(mat, fignum=0)
    # #plt.show()
    # plt.draw()
    # plt.pause(0.0000000000000000000000000000001)
    # plt.cla()
    
    #plt.close()        

    #convert to 2D matrix
    coordinate_matrix=np.zeros((MAX_RANGE,MAX_RANGE*2))
    x_round=np.rint(x)
    y_round=np.rint(y)
    coordinate_matrix[0,MAX_RANGE]=1000 #set robot location on map
    for i in range(0,NUM_OF_MEASURMENTS):
        if not (x[i]==0 and y[i]==0):
            if y[i]>=0:
                coordinate_matrix[int(x_round[i]),int(MAX_RANGE+y_round[i])]=2500
            else:
                coordinate_matrix[int(x_round[i]),int(MAX_RANGE+y_round[i])]=2500

            if int(x_round[i])-1>=0:
                coordinate_matrix[int(x_round[i]-1),int(MAX_RANGE+y_round[i])]=2000
                if int(MAX_RANGE+y_round[i])-1>=-MAX_RANGE:
                    coordinate_matrix[int(x_round[i]-1),int(MAX_RANGE+y_round[i]-1)]=2000
                if int(MAX_RANGE+y_round[i])+1<=MAX_RANGE:
                    coordinate_matrix[int(x_round[i]-1),int(MAX_RANGE+y_round[i]+1)]=2000

            if int(x_round[i])+1<=MAX_RANGE:
                coordinate_matrix[int(x_round[i]+1),int(MAX_RANGE+y_round[i])]=2000
                if int(MAX_RANGE+y_round[i])-1>=-MAX_RANGE:
                    coordinate_matrix[int(x_round[i]+1),int(MAX_RANGE+y_round[i]-1)]=2000
                if int(MAX_RANGE+y_round[i])+1<=MAX_RANGE:
                    coordinate_matrix[int(x_round[i]+1),int(MAX_RANGE+y_round[i]+1)]=2000

            if int(MAX_RANGE+y_round[i])-1>=-MAX_RANGE:
                coordinate_matrix[int(x_round[i]),int(MAX_RANGE+y_round[i]-1)]=2000
            if int(MAX_RANGE+y_round[i])+1<=MAX_RANGE:
                coordinate_matrix[int(x_round[i]),int(MAX_RANGE+y_round[i]+1)]=2000
    


    point_direction=0.0
    point_direction_x=0.0
    point_direction_xy=0.0
    point_direction_y=0.0
    nearest_direction=0.0
    coun=0
    # for i in range (0,MAX_RANGE):
    #     for j in range(0,MAX_RANGE*2):
    #         if coordinate_matrix[i,j] ==2500:
    #             xx=i
    #             yy=j
    #             xx_next=xx
    #             yy_next=yy
    #             point_direction=np.arctan2([j,MAX_RANGE],[i,0])
    #             #print(point_direction[0])
    #             while xx!=0 or yy!=MAX_RANGE :
                    
    #                 #print point_direction
    #                 if xx>0 :
    #                     xx_next=xx-1
    #                 if yy<MAX_RANGE:
    #                     yy_next=yy+1
    #                 if yy>MAX_RANGE:
    #                     yy_next=yy-1
    #                 if yy==MAX_RANGE:
    #                     yy_next=yy

    #                 point_direction_x=np.arctan2([yy,MAX_RANGE],[xx_next,0])
    #                 #print(point_direction_x[0])
    #                 point_direction_y=np.arctan2([yy_next,MAX_RANGE],[xx,0])
    #                 #print(point_direction_y[0])
    #                 point_direction_xy=np.arctan2([yy_next,MAX_RANGE],[xx_next,0])
    #                 #print(point_direction_xy[0])
    #                 nearest_direction=find_nearest([point_direction_xy,point_direction_y,point_direction_x],point_direction)
    #                 if nearest_direction[0]==point_direction_xy[0]:
    #                     coordinate_matrix[xx_next,yy_next]=1500
    #                     xx=xx_next
    #                     yy=yy_next
    #                     #print "3"

    #                 elif nearest_direction[0]==point_direction_x[0]:
    #                     coordinate_matrix[xx_next,yy]=1500
    #                     xx=xx_next
    #                     #print "1"
                        
    #                     #yy=yy_next
    #                 elif nearest_direction[0]==point_direction_y[0]:
    #                     coordinate_matrix[xx,yy_next]=1500
    #                     yy=yy_next
    #                     #print "2"
    #                     if np.abs(yy>xx):
    #                         if yy_next==MAX_RANGE or yy_next%2==0:
    #                             xx=xx_next
    #                     else:
    #                         coun+=coun
    #                         if coun==5:
    #                             coun==0
    #                             xx=xx_next
                        #xx=xx_next
                    
    ratio=0.0
    t=0
    points_to_object=[]
    point_to_maybe=[]
    for i in range (0,MAX_RANGE):
        for j in range(0,MAX_RANGE*2):
            if coordinate_matrix[i,j]==2500 : 
                point_direction=np.arctan2([j,MAX_RANGE],[i,0])
                points_to_object=get_line((i,j),(0,MAX_RANGE))
                for xx in range(0,MAX_RANGE):
                    for yy in range(0,MAX_RANGE*2):
                        t=0
                        while points_to_object[t][0] :
                            if points_to_object[t][0]==xx and points_to_object[t][1]==yy :
                                if coordinate_matrix[xx,yy]!=2000:
                                    coordinate_matrix[xx,yy]=1500
                                if coordinate_matrix[xx,yy+1]!=2000:
                                    coordinate_matrix[xx,yy+1]=1500
                                if coordinate_matrix[xx,yy-1]!=2000:
                                    coordinate_matrix[xx,yy-1]=1500
                            t=t+1

            
                        

                
                                
                    
                       
                    
                    

                        




                    



    # point_direction=0.0
    # point_direction_next_y=0.0
    # point_direction_next_x=0.0
    # point_direction_next_xy=0.0
    # nearest_direction=0.0
    # for i in range (0,MAX_RANGE):
    #     for j in range (0,MAX_RANGE*2):
    #         if coordinate_matrix[i,j] ==2500:
    #             xx=i
    #             yy=j
    #             xx_next=0
    #             yy_next=0
    #             while xx!=0 or yy!=MAX_RANGE:
    #                 point_direction=np.arctan2([yy,MAX_RANGE],[xx,0])
    #                 if xx>0:
    #                     xx_next=xx-1
    #                 if yy<MAX_RANGE:
    #                     yy_next=yy+1
    #                 if yy>MAX_RANGE:
    #                     yy_next=yy-1
    #                 if yy==MAX_RANGE:
    #                     yy_next=yy
    #                 point_direction_next_xy=np.arctan2([yy_next,MAX_RANGE],[xx_next,0])
    #                 point_direction_next_x=np.arctan2([yy,MAX_RANGE],[xx_next,0])
    #                 point_direction_next_y=np.arctan2([yy_next,MAX_RANGE],[xx,0])
    #                 nearest_direction=find_nearest([point_direction_next_xy,point_direction_next_x,point_direction_next_y],point_direction)
    #                 if nearest_direction[0]==point_direction_next_xy[0]:
    #                     coordinate_matrix[xx_next,yy_next]=1500
    #                     yy=yy_next
    #                     xx=xx_next
    #                 if nearest_direction[0]==point_direction_next_y[0]:
    #                     coordinate_matrix[xx,yy_next]=1500
    #                     yy=yy_next
    #                     xx=xx
    #                 if nearest_direction[0]==point_direction_next_x[0]:
    #                     coordinate_matrix[xx_next,yy]=1500
    #                     yy=yy
    #                     xx=xx_next


                  

                    
                    
                    



                


    # for i in range (0,MAX_RANGE):
    #     for j in range (0,MAX_RANGE*2):
    #         if coordinate_matrix[i,j] ==2500:
    #             xx=i
    #             yy=j
    #             while xx!=0 or yy!= MAX_RANGE:
    #                 if yy<MAX_RANGE:
    #                     if xx>0:
    #                         coordinate_matrix[xx-1,yy+1]=1500
    #                         yy=yy+1
    #                     else:
    #                         coordinate_matrix[xx,yy+1]=1500
    #                         yy=yy+1
    #                 if yy>MAX_RANGE:
    #                     if xx>0:
    #                         coordinate_matrix[xx-1,yy-1]=1500
    #                         yy=yy-1
    #                     else:
    #                         coordinate_matrix[xx,yy+1]=1500
    #                         yy=yy-1
    #                 if yy==MAX_RANGE and xx>0:
    #                     coordinate_matrix[xx-1,yy]=1500
    #                 if xx>0:
    #                     xx=xx-1
    
    # for i in range (0,MAX_RANGE):
    #     for j in range (0,MAX_RANGE*2):
    #         t=0
    #         k=0
    #         if coordinate_matrix[i,j]!=0:
    #             t=i
    #             k=j
    #             if k<t:
    #                 for tt in range (t,0,-1):
    #                     kk=k
    #                     if kk>MAX_RANGE:
    #                         coordinate_matrix[tt-1,kk-1]=1500
    #                         kk=kk-1
    #                     if kk<MAX_RANGE:
    #                         coordinate_matrix[tt-1,kk+1]=1500
    #                         kk=kk+1
    #                     if kk==MAX_RANGE:
    #                         coordinate_matrix[tt-1,kk]=1500




    #print coordinate_matrix
    plt.imshow(coordinate_matrix,cmap='jet')
    plt.draw()
    plt.pause(0.0000000000000000000000000000001)
    plt.cla()
    

             

    
       
    



rospy.init_node('my')
print "++++++++++++++++++++++++++++++++"
rospy.Subscriber("/object_disposer_robot/scan", LaserScan, clbk_laser, queue_size=1)
#plt.ion()
#plt.show()
rospy.spin()
 


