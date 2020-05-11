#!/usr/bin/env python

import gym
import numpy as np
import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from imagetranformer import transform
from rl_common import ReplayMemory, update_state, learn
from dqn_model import DQN
import cv2
import tensorflow as tf
from datetime import datetime
import sys
from std_msgs.msg import Int16
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os.path
from os import path