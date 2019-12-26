#!/usr/bin/env python

import gym
import numpy as np
import time
from gym import wrappers
# ROS packages required


from datetime import datetime
import sys
from std_msgs.msg import Int16
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os.path
from os import path

for i in range (0,100):
    if  not path.exists('results.csv'):
        with open('results.csv', 'w') as newFile:
            newFileWriter = csv.writer(newFile)
            newFileWriter.writerow(['Episode', 'Reward'])
            #newFileWriter.writerow([i, i])
    else:
            with open('results.csv', 'a') as newFile:
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([i, i])