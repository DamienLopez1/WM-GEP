import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems


import argparse
from os.path import join, exists

import random
import numpy as np
import cv2
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as Dataset
from torchvision import transforms
import scipy.stats
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window



from utils.misc import RolloutGenerator
#import VAEtorch
#import trainmdrnn


Device_Used = "cuda"
device = torch.device("cuda" if  Device_Used == "cuda" else "cpu")
#device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
args = parser.parse_args()


args.logdir = 'D:\steps1000'

#RolloutGenerator Params: 
# (mdir: model directory, device, time_limit: number of samples in goal space before exploration,
#number_goals: number of goals to set over lifetime of agent,
#Forward_model: 'M' = World Model, 'D' = Linear layers(do not use),
#hiddengoals: True = Goals set in World Model, False = goals as observations(basically IMGEPs),
#curiosityreward = True/False - not relevant in this implementation,
#static: True = static VAE and HiddenVAE, False = constantly evolving VAE and HiddenVAE)
generator = RolloutGenerator(args.logdir, device, 100,200,True,False,False)
    
generator.rollout(None,render = False) #run program










