#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 00:45:09 2018

@author: Jet
"""

#from dumbNodes.radioNode import radioNode
#from myFunction import ismember   #
#import random
import numpy as np

#from actor import Actor
#from critic import Critic

import actor
import critic

import tensorflow as tf

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

class ac:
#    goodChans     = [ ]    
#    numStates     = [ ]
#    states        = [ ]
#    stateHist     = [ ]
#    stateTally    = [ ]
#    stateTrans    = [ ]
#    avgStateTrans = [ ]
#    
#    discountFactor   = 0.9
#    policyAdjustRate = 5         # Policy is adjusted at this step increment
#    
#            
#    policy           = [ ] 
#    policyHist       = [ ]        
#    # [Not transmitting, Good Channel no Interference, Good Channel Interference, 
#    # Bad Channel no Interference, Bad Channel Interference]
#    rewards          = [-200, 100, -200, 50, -100]   
#    # different duty cycle need different rewards   
#    rewardHist       = [ ]
#    rewardTally      = [ ]        
#    rewardTrans      = [ ]
#    cumulativeReward = [ ]
#    
    
    

    
#    def __init__(self,numChans,states,numSteps):
    def __init__(self,numChans):
#         states = np.array( states) 
#        self.actions = np.zeros((numChans+1,numChans))
#        for k in range(0,numChans):
#            self.actions[k+1,k] = 1
#        self.numChans      = numChans
#        self.numActions    = np.shape(self.actions)[0]
#        self.actionTally   = np.zeros(numChans+1)
#        self.actionHist    = np.zeros((numSteps,numChans))
#        self.actionHistInd = np.zeros(numSteps)
#        
#        self.goodChans     = np.ones(numChans)
#        
#        self.states        = states
#        self.numStates     = np.shape(states)[0]
#        
#        self.stateHist     = np.zeros((numSteps,numChans))
#        self.stateTally    = np.zeros(self.numStates)
#      
#        self.rewardHist    = np.zeros(numSteps)
#        self.rewardTally   = np.zeros(numChans+1)
#        self.cumulativeReward = np.zeros(numSteps)
#        self.rewardTrans   = np.zeros((self.numActions, self.numStates,self.numStates) )
#        
#        self.exploreHist   = [ ]
#        
#        self.type          = "ac"
#        self.hyperType     = "learning"
#        
#        self.policy        = np.zeros(numChans)
#               
#        self.n_actions     = numChans + 1   
#        self.n_features    = numChans 
        
        sess = tf.Session()

        self.actor_  = actor.Actor( self,  sess, numChans, numChans+1, lr = 0.001)
        self.critic_ = critic.Critic(self, sess, numChans , lr = 0.01) 
        
        sess.run(tf.global_variables_initializer())
        
        


        
    def getAction(self,observation):
        observation = np.array( observation)   # new for add in matlab
        observation = observation[np.newaxis, :]
        
        temp = self.actor_.choose_action(observation) 
        return temp
#        # !!! new define, convert action from a int to a array
#        action       = np.zeros(self.numChans) 
#        if temp > 0:
#            action[temp-1] = 1 
#        
#        self.actionHist[stepNum,:] = action                   
#        if not np.sum(action):
#            self.actionTally[0] +=    1
#            self.actionHistInd[stepNum] = 0
#        else:
#            self.actionHistInd[stepNum] = np.where(action == 1)[0] + 1
#            self.actionTally[1:] += action
#        
#        return action, temp  
    
    
#    def getReward(self,collision,stepNum, isWait):
#        
#        if isWait == True:
#             self.rewards  = [-50, 100, -200, 50, -100] 
#        action = self.actionHist[stepNum,:]
#        if not np.sum(action):
#            reward = self.rewards[0]
#            self.rewardTally[0] +=  reward
#        else:
#            if any(np.array(self.goodChans+action) > 1): 
#                if collision == 1:
#                    reward = self.rewards[2]
#                else:
#                    reward = self.rewards[1]             
#            else:
#                if collision == 1:
#                    reward = self.rewards[4]
#                else:
#                    reward = self.rewards[3]  
#                    
##            if stepNum > 5000:
##                reward *= stepNum*0.1   
##            else:
##                pass
# 
#            self.rewardTally[1:] += action * reward        
#        self.rewardHist[stepNum] = reward   
#        
#        if stepNum == 0:
#            self.cumulativeReward[stepNum] = reward
#        else:
#            self.cumulativeReward[stepNum] = self.cumulativeReward[stepNum-1] + reward
#        return reward  
    
    
#    def storeTransition(self, s, a, r, s_):
#        # self.dqn_.store_transition(s, a, r, s_)   
        
    def learn(self, s, a, r, s_):
        s  = np.array(s)
        s_ = np.array(s_)
        td_error = self.critic_.learn(s, r, s_)
        self.actor_.learn(s, a, td_error)
        

       
        
        
        