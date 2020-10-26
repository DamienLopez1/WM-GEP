#!/usr/bin/env python
# coding: utf-8

# # Import Everything



""" Various auxiliary utilities """
import math
from os.path import join, exists
import torch
from torch import optim
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from models import MDRNN,MDRNNCell, VAE, Controller,Dtild,HiddenVAE
from models.mdrnn import gmm_loss

import gym
import gym.envs.box2d
import random
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import cv2
from itertools import chain
import scipy.stats
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
from torch.distributions.categorical import Categorical
from sklearn.metrics import confusion_matrix
import seaborn as sn



ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =    1, 32, 256, 64, 64

criterion = nn.MSELoss()

#Initialises KDE sampling
def sampling_method(kde):
    sample = kde.resample(size=1).squeeze()
    return sample.ravel()


#Below are functions used for data collection to train VAE, MDRNN and HiddenVAE


def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)



# # Rollout Class

class RolloutGenerator(object):
    """ Utility to generate rollouts.

    mdir: model directory i.e where models are stored, 
    device: cuda or cpu
    time_limit: number of samples in goal space before exploration,
    number_goals: number of goals to set over lifetime of agent,
    Forward_model: 'M' = World Model, 'D' = Linear layers(do not use),
    hiddengoals: True = Goals set in World Model, False = goals as observations(basically IMGEPs),
    curiosityreward = True/False - not relevant in this implementation,
    static: True = static VAE and HiddenVAE, False = constantly evolving VAE and HiddenVAE
    """
    def __init__(self, mdir, device, time_limit,number_goals,Forward_model,hiddengoals:bool, curiosityreward = bool,static = bool):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file,Dtild_file,hiddenvae_file =             [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl','dtild','hiddenvae']]

        assert exists(vae_file) and exists(rnn_file),            "Either vae or mdrnn is untrained."

        vae_state, rnn_state,hiddenvae_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file,hiddenvae_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state),('HiddenVAE',hiddenvae_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])
        
        self.HiddenVAE = HiddenVAE(256, LSIZE).to(device)
        self.HiddenVAE.load_state_dict(hiddenvae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
        
        self.mdrnnBIG = MDRNN(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnnBIG.load_state_dict(rnn_state["state_dict"])
        
        
        self.controller = Controller(256, 256, 6).to(device)
        
        
        
        
        self.env = gym.make('MiniGrid-MultiRoom-N6-v0')
        
        self.device = device
        self.number_goals = number_goals
        self.time_limit = time_limit
        
        
        self.vae_state = vae_state
        self.rnn_state = rnn_state
        self.hiddenvae_state = hiddenvae_state
        
        self.hiddengoals = hiddengoals
        self.curiosityreward = curiosityreward
        self.static = static
        self.Forward_model = Forward_model
        
        
        self.fmodel = Dtild(32,256,1, 32).to(device)
        
        
        

    def rollout(self, params, render= False):
        """ Executes rollouts for number of goals

        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.fmodel)
        optimizer = optim.Adam(params= self.fmodel.parameters(), lr=0.0001)
        
        
        
        MDRNNoptimizer = torch.optim.RMSprop(self.mdrnnBIG.parameters(), lr=1e-3, alpha=.9)
        #MDRNNoptimizer.load_state_dict(self.rnn_state["optimizer"])
        
        VAEOptimizer = optim.Adam(self.vae.parameters())
        VAEOptimizer.load_state_dict(self.vae_state["optimizer"])
        
        HiddenVAEOptimizer = optim.Adam(self.HiddenVAE.parameters())
        HiddenVAEOptimizer.load_state_dict(self.hiddenvae_state["optimizer"])
        
        zstate_list = []
        self.env.seed(1337) 
        #self.env.reset()
        obs = self.env.reset()
        
        
        obs = obs['image']
        expl_rate = 0.4
        
            
        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]
        
        _,latent_mu,logsigma,z = self.tolatent(obs)
        
        i = 0
        
        #Bootstrapping, collect 100 initial states for goal space
        while True:
           
            action = random.randrange(6)
            
            _,hidden,z,zh  = self.transform_obs_hidden(obs, hidden, action)
            _,_,hidden_latent = self.tohiddenlatent(hidden)
            obs, exreward, done, _ = self.env.step(action)
            obs = obs['image']
            
            if self.hiddengoals:
                zstate_list.append(np.array(hidden_latent.cpu().detach().numpy()))#if we use pure hidden
            else:
                zstate_list.append(np.array(z.cpu().detach().numpy()))#if we use latent_space
            
            i+=1
            if render:
                self.env.render()
            
            
            if i > self.time_limit:
                break
            
    
        s = obs
        loss_list = []
        WM_loss= []
        VAE_loss_per_rollout = []
        hiddenvae_loss_per_rollout = []
        rollout_reward = []
        visitationcount = []
        exreward_per_rollout = []
        visitationarray = np.zeros((25,25))
        final_loss = []
        
        #Goal Exploration
        for c in range(self.number_goals):
            #reset env, uncomment below if necessary to reset agent in enviroinment after every episode
            '''
            self.env.seed(1337)
            self.env.reset()
            s = self.env.reset()
            
            #reset obs and hidden state
            s = s['image']
            _,_,_,z = self.tolatent(s)
            hidden = [
            torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]
            '''
            
            print('Goal Number', c)
            zstate_list = np.array(zstate_list) 
            zstate_list = zstate_list.squeeze(1)
            kde = scipy.stats.gaussian_kde(zstate_list.T)
            
            z_goal = sampling_method(kde) #sample goal from goal space using KDE
            z_goal = torch.tensor([z_goal],dtype = torch.float32).to(self.device) #controller requires both as tensors
            
            
            if not self.hiddengoals:
                z_goal_obs = self.vae.decoder(z_goal)
                z_goal_obs = z_goal_obs.reshape(7,7,3)
                z_goal_obs = np.array(z_goal_obs.cpu().detach())

                plt9 = plt.figure('Zgoal')
                plt.cla()
                sn.heatmap(z_goal_obs[:,:,0],cmap = 'Reds', annot=True,cbar = False).invert_yaxis()
            
            total_hiddenvae_loss = 0
            total_vae_loss = 0
            total_reward = 0
            total_exreward = 0
            total_loss = 0
            goal_loss = []
            
            
           
            scur_rollout = []
            snext_rollout = []
            r_rollout = []
            d_rollout = []
            act_rollout = []
            
            zstate_list = zstate_list[:,np.newaxis,:]
            zstate_list = zstate_list.tolist()
            
            
            for goalattempts in range(100):
                if visitationarray[self.env.agent_pos[0],self.env.agent_pos[1]] ==0:
                    visitationarray[self.env.agent_pos[0],self.env.agent_pos[1]] += 1
                h = []
                for a in range(6):
                    if self.Forward_model == 'D':
                        h.append(self.fmodel(z.detach(),hidden[0].detach(),torch.tensor([[a]],dtype = torch.float32).to(self.device)))
                        
                    else:
                        #Perform a prediction of next state for every action. Add to a list spo comparison with goal can occur
                        z, hmus,hsigmas,hlogpi, zh, next_hidden, next_hidden_latent, next_hidden_mu, next_hidden_sigma = self.predict_next(s, hidden,a) 
                        h.append([hmus,hsigmas,hlogpi,next_hidden_latent,next_hidden_mu,next_hidden_sigma])
                        
                    
                
                if expl_rate > random.random():
                    m = random.randrange(6)
                else:
                    #choose action which will bring us closer to goal
                    m = self.gpcf(z_goal,h)
                
                
                
                z, hmus,hsigmas,hlogpi, zh, hidden, hidden_latent, hidden_mu, hidden_sigma = self.predict_next(s, hidden,m) #gets mean, standard deviation and  pi, next latent of prediction of next latent obs
                
                           
                if not self.hiddengoals:
                    if self.Forward_model == 'D':
                        predicted_next_obs = self.vae.decoder(h[m])
                        predicted_next_obs = predicted_next_obs.reshape(7,7,3)
                        p = np.array(predicted_next_obs.cpu().detach())
                    else:
                        predicted_next_obs = self.vae.decoder(zh)
                        predicted_next_obs = predicted_next_obs.reshape(7,7,3)
                        p = np.array(predicted_next_obs.cpu().detach())
                else:
                    predicted_next_obs = self.vae.decoder(zh)
                    predicted_next_obs = predicted_next_obs.reshape(7,7,3)
                    p = np.array(predicted_next_obs.cpu().detach())  
                    
                
                #Show predicted next observation
                if render:
                    plt5 = plt.figure('Predicted obs')
                    plt.cla()
                    sn.heatmap(p[:,:,0],cmap = 'Reds', annot=True,cbar = False).invert_yaxis()
                
                
                s,exreward,_,_ = self.env.step(m)#perform action , get next observation and external reward if any 
                total_exreward += exreward
                
                
                s = s['image']
                
                recons,next_mu,next_logsigma,next_z = self.tolatent(s) #transform observation to latent representation
                
                if self.hiddengoals:
                    reconhidden,hiddenmu,hiddenlogsigma  =  self.HiddenVAE(hidden[0].detach()) #transoform hidden state into latent representation if using goals in world model
               
               
                #Show actual observation
                if render:
                    plt6 = plt.figure('Actual obs')
                    plt.cla()
                    sn.heatmap(s[:,:,0],cmap = 'Reds', annot=True,cbar = False).invert_yaxis() 
                
                #Collect information for training World Model
                scur_rollout.append(np.array(z.cpu().detach()))
                snext_rollout.append(np.array(next_z.cpu().detach()))
                r_rollout.append([0.0])
                act_rollout.append([[np.float(m)]])
                d_rollout.append([0.0])
                
                
                if render:
                    self.env.render()
                
                
                if self.hiddengoals:
                    hiddenvae_loss = self.VAEloss(reconhidden,hidden[0].detach(),hiddenmu,hiddenlogsigma)
                    total_hiddenvae_loss += hiddenvae_loss
                    
                VAE_loss = self.VAEloss(recons,torch.tensor(s.flatten(),dtype = torch.float32).unsqueeze(0).to(self.device),next_mu,next_logsigma)
                total_vae_loss += VAE_loss
                
                #Curiosity reward is how far the next state was from the prediction 
                Curiosityreward = gmm_loss(next_z.detach(), hmus, hsigmas, hlogpi)/33
                
                #Uncomment below if requiring to add only completely new hidden states to goal space
                '''
                if Curiosityreward > 1.29: #only add this to the goal space if it was new: this promotes sampling goals which we are unsure about
                    if self.hiddengoals:
                        zstate_list.append(np.array(hidden_latent.cpu().detach().numpy()))#if we use pure hidden
                    else:
                        zstate_list.append(np.array(z.cpu().detach().numpy()))#if we use latent_space
                '''
                
                #add all states to goal space
                if self.hiddengoals:
                      zstate_list.append(np.array(hidden_latent.cpu().detach().numpy()))#if we use pure hidden
                else:
                      zstate_list.append(np.array(z.cpu().detach().numpy()))#if we use latent_space
                
               
                #if forward model is a linear layer then there are vastly different loss functions. This performs badly so is not recommended to use
                if self.Forward_model == 'D':
                    if self.hiddengoals:
                        goal_loss.append(self.LSEloss(hidden_latent,z_goal)) #how far away the achieved step is from the goal
                        floss = self.LSEloss(h[m],hidden_latent.detach())#difference between forward model prediction and next hidden
                    else:
                        goal_loss.append(criterion(next_z.detach(),z_goal).item()) #how far away the achieved step is from the goal
                        floss = criterion(h[m],next_z.detach())#difference between forward model prediction and next latent
                else:
                    if self.hiddengoals:
                        goal_loss.append(gmm_loss(z_goal,hidden_mu,hidden_sigma,torch.tensor([-1.0], dtype = torch.float32).to(self.device))/33) #how far away the achieved step is from the goal
                        floss = Curiosityreward #difference between forward model prediction and next hidden
                    else:
                        goal_loss.append(gmm_loss(z_goal,next_mu,next_logsigma.exp(),torch.tensor([-1.0], dtype = torch.float32).to(self.device))/33)
                        floss = Curiosityreward
                
                total_loss += floss
                
                #train forward model D if necessary
                if self.Forward_model == 'D':
                    optimizer.zero_grad()
                    floss.backward()
                    optimizer.step()
                
                #To see what goals look like at lowest observed distance throughout testing
                if goal_loss[-1] < 1.5: 
                    '''
                    plt84 = plt.figure('Actual obs')
                    plt.cla()
                    sn.heatmap(s[:,:,0],cmap = 'Reds', annot=True,cbar = False).invert_yaxis() 
                    
                    
                    plt85 = plt.figure('Zgoal')
                    plt.cla()
                    sn.heatmap(z_goal_obs[:,:,0],cmap = 'Reds', annot=True,cbar = False).invert_yaxis()
                    plt.show()
                    '''
                    reward = 4.0 #this reward is more of a place holder
                else:
                    reward = 0.0
                
                if self.curiosityreward:
                    reward = reward + Curiosityreward
                
                total_reward += reward
                
                
            final_loss.append(goal_loss[-1]) 
            
            #Using every single observation, action, next observation,terminality condition and reward seen in the episode, get the loss of the world model
            mdrnnlosses = self.get_loss(torch.tensor(scur_rollout).to(self.device), torch.tensor(act_rollout).to(self.device), torch.tensor(r_rollout).to(self.device),
                              torch.tensor(d_rollout).to(self.device), torch.tensor(snext_rollout).to(self.device), include_reward = False)
            
            #train world model
            MDRNNoptimizer.zero_grad()
            mdrnnlosses['loss'].backward()
            MDRNNoptimizer.step()
            
            WM_loss.append(mdrnnlosses['loss']) #append to world model loss graph
            
            #train VAE and HiddenVAE if representation learning is not static
            if not self.static:
                VAE_loss_per_rollout.append(total_vae_loss/(goalattempts+1)) #average VAE loss metric when non static representations are being used
                VAEOptimizer.zero_grad()
                VAE_loss_per_rollout[-1].backward()
                VAEOptimizer.step()
                
                if self.hiddengoals:
                    hiddenvae_loss_per_rollout.append(total_hiddenvae_loss/(goalattempts+1))  #average HiddenVAE loss metric when non static representations  of hiddens states are being used
                    HiddenVAEOptimizer.zero_grad()
                    hiddenvae_loss_per_rollout[-1].backward()
                    HiddenVAEOptimizer.step()
            
            
            
            if goalattempts % 10 == 0:   #every 10 goals update the MDRNN cell for use in predicting the next state
                self.mdrnn.load_state_dict(self.mdrnnBIG.state_dict())    
                
    
            
            loss_list.append(total_loss/(goalattempts+1))
            rollout_reward.append(total_reward)
            visitationcount.append(np.sum(visitationarray))
            exreward_per_rollout.append(total_exreward)          
            
            
        plot1 = plt.figure('Average Forward model loss')
        plt.plot(loss_list)
        plt7= plt.figure('WM_loss')
        plt.plot(WM_loss)
        plt4 = plt.figure('Distance to goal per step')
        plt.cla()
        plt.plot(goal_loss)
        rolloutrewardplot = plt.figure('Reward per rollout')
        plt.plot(rollout_reward)
        if not self.static:
            vaerolloutplot = plt.figure('VAE loss per rollout')
            plt.plot(VAE_loss_per_rollout)
            if self.hiddengoals:
                hiddenvaerolloutplot = plt.figure('HiddenVAE loss per rollout')
                plt.plot(hiddenvae_loss_per_rollout)
        plt8 = plt.figure('Visitation')
        plt.plot(visitationcount)
        pltexreward = plt.figure('Extrinsic Reward per rollout')
        plt.plot(exreward_per_rollout)
        pltgoalloss = plt.figure('Final Goal Loss per Episode')
        plt.plot(final_loss)
        plt.show() 
        
        input('stop')
        
            
            
    def transform_obs_hidden(self,obs, hidden,m):
        obs = torch.tensor(obs.flatten(),dtype = torch.float32).unsqueeze(0).to(self.device)
       
        action =  torch.Tensor([[m]]).to(self.device)
        reconx, latent_mu, logsigma = self.vae(obs)
        
        

        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(latent_mu)
        
        hmus, hsigmas, hlogpi, _, _, next_hidden = self.mdrnn(action, z, tuple(hidden))
        
        hlogpi = hlogpi.squeeze()
        mixt = Categorical(torch.exp(hlogpi)).sample().item()
        
        
        zh = hmus[:, mixt, :]  + hsigmas[:, mixt, :] * torch.randn_like(hmus[:, mixt, :])
        
        return action.squeeze().cpu().numpy(), next_hidden,z,zh
    
    def gpcf(self,z_goal,h):
        #Compares possible next state as a result of each action to goal. 
        #Chooses the action which will reduce the loss the most
        
        output = []
        for action in range(len(h)):
            if self.hiddengoals:
                if self.Forward_model == 'D':
                    output.append(self.LSEloss(h[action],z_goal))
                else:
                    #compares the goal with the next state using the NLL loss of the latent representation of the next hidden state
                    #h[action][4] = next_hidden_mu, h[action][5] =  next_hidden_sigma, logpi = -1
                    output.append(gmm_loss(z_goal,h[action][4],h[action][5],torch.tensor([-1.0], dtype = torch.float32).to(self.device))/33)
            else:
                if self.Forward_model == 'D':
                    output.append(criterion(h[action],z_goal).item())
                else:
                    #compares the goal with the next state using the NLL loss of the latent representation of the next hidden state
                    #h[action][0] = mu, h[action][1] = sigma, logpi = logpi
                    output.append(gmm_loss(z_goal, h[action][0], h[action][1], h[action][2])/33)
                
                
        return output.index(min(output))
    
    
   
    def get_loss(self, latent_obs, action, reward, terminal,
                 latent_next_obs, include_reward: bool):
            """ Compute losses.
        
            The loss that is computed is:
            (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
                 BCE(terminal, logit_terminal)) / (LSIZE + 2)
            The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
            approximately linearily with LSIZE. All losses are averaged both on the
            batch and the sequence dimensions (the two first dimensions).
        
            :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
            :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
            :args reward: (BSIZE, SEQ_LEN) torch tensor
            :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        
            :returns: dictionary of losses, containing the gmm, the mse, the bce and
                the averaged loss.
            """
            
            
            mus, sigmas, logpi, rs, ds = self.mdrnnBIG(action, latent_obs)
            gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
            bce = F.binary_cross_entropy_with_logits(ds, terminal)
            if include_reward:
                mse = F.mse_loss(rs, reward)
                scale = LSIZE + 2
            else:
                mse = 0
                scale = LSIZE + 1
            loss = (gmm + bce + mse) / scale
            return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)     
        
    def tolatent(self,obs):
        obs = torch.tensor(obs.flatten(),dtype = torch.float32).unsqueeze(0).to(self.device)
       
        reconx, latent_mu, logsigma = self.vae(obs)
        
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(latent_mu)
        
        return reconx,latent_mu,logsigma,z
    
    def tohiddenlatent(self,hidden):
        '''

        Parameters
        ----------
        hidden : tensor.
            hidden state of mdrnn
        Returns
        -------
        latent_mu : tensor
            mean of latent representation of hidden state.
        sigma : tensor
            standard deviation of latent representation of hidden state.
        zhidden : tensor
           latent representation of hidden state.

        '''
        _,latent_mu,logsigma  =  self.HiddenVAE(hidden[0].detach())
              
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        zhidden = eps.mul(sigma).add_(latent_mu)
        
        return latent_mu,sigma,zhidden
     
    def predict_next(self,obs, hidden,m):
        '''
        Parameters
        ----------
        obs : array(7x7x3)
            Observation of current state.
        hidden : tensor
            current hidden state.
        m : integer
            action to be taken or could be taken.

        Returns
        -------
        z : tensor
            Latent representation of current observation.
        hmus : tensor
            mean of gaussians of prediction of latent representation of next observation.
        hsigmas : tensor
             standard deviation of gaussians of prediction of latent representation of next observation.
        hlogpi : tensor
            Mixture proportion of gaussians of prediction of latent representation of next observation.
        zh : tensor
            predicition of next observation using categorical distribution.
        next_hidden : tenosr
            next hidden state given the action.
        next_hidden_latent : tensor
            latent representation of next hidden state.
        next_hidden_mu : tensor
            mean of latent representation of next hidden state.
        next_hidden_sigma : tensor
            standard deviation of latent representation of next hidden state.
        '''
        
        obs = torch.tensor(obs.flatten(),dtype = torch.float32).unsqueeze(0).to(self.device)
       
        action =  torch.Tensor([[m]]).to(self.device)
        reconx, latent_mu, logsigma = self.vae(obs)
        
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(latent_mu)
        
        hmus, hsigmas, hlogpi, _, _, next_hidden = self.mdrnn(action, z, tuple(hidden))
        
        hlogpi = hlogpi.squeeze()
        mixt = Categorical(torch.exp(hlogpi)).sample().item() 
        #gets prediction of next latent using categorical distribution over gaussians predicted with MDRNN
        zh = hmus[:, mixt, :]  + hsigmas[:, mixt, :] * torch.randn_like(hmus[:, mixt, :]) 
        
        
        next_hidden_mu,next_hidden_sigma,next_hidden_latent = self.tohiddenlatent(next_hidden)
        
        
        
        return z, hmus,hsigmas,hlogpi, zh, next_hidden, next_hidden_latent, next_hidden_mu, next_hidden_sigma       

    def LSEloss(self,yHat,y):
        return torch.sum((yHat - y)**2)
    
    def VAEloss(self,recon_x, x, mu, logsigma):
        """ VAE loss function """
        
        BCE = F.mse_loss(recon_x, x, size_average=False)
    
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        
        return BCE + KLD
