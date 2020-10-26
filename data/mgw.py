"""
Generating data from the CarRacing gym environment.
!!! DOES NOT WORK ON TITANIC, DO IT AT HOME, THEN SCP !!!
"""
import argparse
from os.path import join, exists
import gym
import numpy as np

import random
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import cv2

def generate_data(rollouts, data_dir, noise_type): # pylint: disable=R0914
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."

    env = gym.make('MiniGrid-MultiRoom-N6-v0')
    env.reset()
    #env = RGBImgPartialObsWrapper(env) # Get pixel observations
    seq_len = 1000

    for i in range(rollouts):
        #env.reset()
        #print('env.action_space',env.action_space)
        #env.env.viewer.window.dispatch_events()
     
        s_rollout = []
        r_rollout = []
        d_rollout = []
        a_rollout = []
        
        t = 0
        while True:
            action = random.randint(0, env.action_space.n - 1)
            t += 1
            #env.render()
            s, r, done, _ = env.step(action)
            
            #tu = cv2.resize(s['image'],(64,64))
            tu = s['image']
            #env.env.viewer.window.dispatch_events()
            s_rollout += [tu]
            if t == 1000:
                d = True
            else: 
                d= False
        
           
            r_rollout += [r]
            d_rollout += [d]
            a_rollout += [[action]]
            if t ==1000:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy)
