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
from models import MDRNNCell, VAE

import torch


def generate_data(rollouts, data_dir, noise_type): # pylint: disable=R0914
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."
    
    mdir = 'D:\steps1000'

    vae_file, rnn_file, ctrl_file = \
                [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]
    
    assert exists(vae_file) and exists(rnn_file),\
        "Either vae or mdrnn is untrained."
    
    vae_state, rnn_state= [
        torch.load(fname, map_location={'cuda:0': str('cuda')})
        for fname in (vae_file, rnn_file)]
    
    for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
        print("Loading {} at epoch {} "
              "with test loss {}".format(
                  m, s['epoch'], s['precision']))
    
    vae = VAE(3, 32).to('cuda')
    vae.load_state_dict(vae_state['state_dict'])
    
    mdrnn = MDRNNCell(32, 1, 256, 5).to('cuda')
    mdrnn.load_state_dict({k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
        
    hidden = [torch.zeros(1, 256).to('cuda') for _ in range(2)]
    
    
    env = gym.make('MiniGrid-MultiRoom-N6-v0')
    env.reset()
    #env = RGBImgPartialObsWrapper(env) # Get pixel observations
    seq_len = 1000

    for i in range(rollouts):
        #env.reset() #uncomment this if a new environment must be produced every episode
        #env.env.viewer.window.dispatch_events()
        
        s_rollout = []
        r_rollout = []
        d_rollout = []
        a_rollout = []
        h_rollout = []
        
        t = 0
        while True:
            action = random.randint(0, env.action_space.n - 1)
            t += 1
            #env.render()
            s, r, done, _ = env.step(action)
            
            #tu = cv2.resize(s['image'],(64,64))
            tu = s['image']
            obs = torch.tensor(tu.flatten(),dtype = torch.float32).unsqueeze(0).to('cuda')
            
            reconx, latent_mu, logsigma = vae(obs)
    
            #print(hidden[0])
            act =  torch.Tensor([[action]]).to('cuda')
            _, _, _, _, _, hidden = mdrnn(act, latent_mu, tuple(hidden))
            
            
            
            #env.env.viewer.window.dispatch_events()
            s_rollout += [tu]
            if t == 125:
                d = True
            else: 
                d= False
        
            r_rollout += [r]
            d_rollout += [d]
            a_rollout += [[action]]
            
            
            
            
            h_rollout.append(np.array(hidden[0].cpu().detach().numpy()))
            
            if t ==125:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout),
                         hiddens = np.array(h_rollout).squeeze(1))
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
