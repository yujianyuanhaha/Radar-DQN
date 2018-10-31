# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:27:51 2018

@author: Yue Xu

Blending: Jet Oct 24th.
"""
'''
Version 1  Thr ~= 0.9 and the optimal Thr should be 0.95
'''
# coding: utf-8


import numpy as np;
import tensorflow as tf;
import random


class drqn:
    
    exploreInit      = 1.0              # Initial exploration probability
    exploreDecay     = 0.01             # Percentage reduction in exploration chance per policy calculation
    # approximate reach 1% at 20000 steps
    exploreProbMin   = 0.01             # avoid the risk to stuck
    exploreHist      = [ ]
    
    def __init__(
           self,
           drqnNode,
           n_actions,  # int
           n_features, # num of channels, int
           learning_rate = 0.001,
           reward_decay  = 0.9,
           lamda         = 0.9,
           e_greedy      = 0.9,
           replace_target_iter = 400,
           batch_size = 1,
           e_greedy_increment = None,
    ):
        self.n_actions           = n_actions
        self.n_features          = n_features
        self.power               = None # input   1*n_features
        self.observation         = None # input from power  1*n_features
        self.reward_estimated    = None # input from power  1*n_features
        self.lr                  = learning_rate
        self.gamma               = reward_decay # discount factor
        self.lamda               = lamda # eligible trace factor
        self.epsilon_max         = e_greedy     # epsilon max value
        self.replace_target_iter = replace_target_iter  # num of steps that updating target-net's parameters
        self.memory_size         = 2000  #  size of memory box 
        self.temp_memory_step    = 0
        self.batch_size          = batch_size    # batch size
        self.n_hidden_units      = 40;
        self.continue_length     = 10
        self.epsilon_increment   = e_greedy_increment 
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max # epsilon_greedy algorithm
        # record training times and decide when to update target-network 
        self.learn_step_counter  = 0
        self.exploreProb         = self.exploreInit
        # initialize memory box [s, a, r, time, s_]  size = n_features[1 * 2^#C vector] + #action[0~#C] + #reward[0~100] +  n_features 
        self.memory              = np.zeros((self.memory_size, n_features*2 + 3)) 
        self.temp_memory         = np.zeros((self.continue_length, self.n_features))
        # build [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection(tf.global_variables, scope='target_net0')
        e_params = tf.get_collection(tf.global_variables, scope='eval_net0')
        #t_params = tf.get_collection('target_net_parameters')  # retrieve parameters of target-net ---List
        #e_params = tf.get_collection('eval_net_parameters')   # retrieve parameters of evaluate-net   ---List
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] # updatge each layers' parameters

        self.sess          = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his      = []  
        self.select_action = []
        self.collision     = 0
        self.collision_hit = []

# Build Network
# There are two networks: Evaluate-Network and Target-Network
# Evaluate Network: Updating parameters in each step
# Target-Network: Updating parameters after several steps
#  E_N and T_N can be seen as an approximated Function
    def _build_net(self):
    
    # Build Evaluate-Network:
    # This Network has 3 layers 
    # s,q_t,r,a,trace are tensor
        self.s        = tf.placeholder(tf.float32,[None, self.n_features * self.continue_length],name = 'state') # receive observation 
        self.q_target = tf.placeholder(tf.float32,[None,self.n_actions],name = 'Q_target') # receive Q_target
        self.r        = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward(memory + estiamted)
        self.a        = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.t        = tf.placeholder(tf.int32, [None, ], name='t')  # step
        self.s_       = tf.placeholder(tf.float32, [None, self.n_features * self.continue_length], name = 's_') # receive observation_next
        self.temp_batch_size = tf.placeholder(tf.int32, name='tbs')  # step
              
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net0'):            
            sIn          = tf.reshape(self.s, [-1, self.n_features])#[None, self.continue_length ,self.n_features]
            e_in         =  tf.layers.dense(sIn, self.n_hidden_units, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e00',reuse=tf.AUTO_REUSE)
            e_in         = tf.reshape(e_in, [self.temp_batch_size, self.continue_length, self.n_hidden_units])
            e_lstm_cell  = tf.nn.rnn_cell.LSTMCell(self.n_hidden_units, state_is_tuple=True)
            e_init_state = e_lstm_cell.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            e_outputs, e_final_state = tf.nn.dynamic_rnn(e_lstm_cell, e_in, initial_state=e_init_state, time_major=False)
            e_o          = tf.reshape(e_outputs, [self.batch_size, -1]) 

            e01          = tf.layers.dense(e_o, 25, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e01')
            #e02 = tf.layers.dense(e01, 40, tf.nn.relu, kernel_initializer=w_initializer,
                                 #bias_initializer=b_initializer, name='e02')
            #e03 = tf.layers.dense(e02, 30, tf.nn.relu, kernel_initializer=w_initializer,
                                 #bias_initializer=b_initializer, name='e03')
            self.q_eval0 = tf.layers.dense(e01, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q0')

    
    # Build Target-Network:
    # This Network has same architecture 
    
    #30 50 60 40 - 200 1600
        with tf.variable_scope('target_net1'):
            
            s_In         = tf.reshape(self.s, [-1, self.n_features])#[None, self.continue_length ,self.n_features]
            t_in         =  tf.layers.dense(s_In, self.n_hidden_units, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e00')
            t_in         = tf.reshape(t_in, [self.temp_batch_size, self.continue_length, self.n_hidden_units])            
            
            #s_In = tf.reshape(self.s_, [-1, self.continue_length, self.n_hidden_units])#[None, self.continue_length ,self.n_features]
            
            t_lstm_cell  = tf.nn.rnn_cell.LSTMCell(self.n_hidden_units, state_is_tuple=True)
            t_init_state = t_lstm_cell.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state

            t_outputs, t_final_state = tf.nn.dynamic_rnn(t_lstm_cell, t_in, initial_state=t_init_state, time_major=False)
            t_o = tf.reshape(t_outputs, [self.batch_size, -1])         
            
            t01 = tf.layers.dense(t_o, 25, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t01')
            #t02 = tf.layers.dense(t01, 40, tf.nn.relu, kernel_initializer=w_initializer,
                                 #bias_initializer=b_initializer, name='t02')
            #t03 = tf.layers.dense(t02, 30, tf.nn.relu, kernel_initializer=w_initializer,
                                 #bias_initializer=b_initializer, name='t03')
            self.q_next0 = tf.layers.dense(t01, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t04')
        
 
            
            
        with tf.variable_scope('q_target1'):
            q_target1 = self.r + self.gamma * tf.reduce_max(self.q_next0, axis=1, name='Qmax_s_1') 
            self.q_target = tf.stop_gradient(q_target1)
        
        with tf.variable_scope('q_eval0'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
        # something will happen here
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval0, indices=a_indices)   
            
        with tf.variable_scope('loss0'):
        #self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error1'))
        
        with tf.variable_scope('train0'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def flatInputS(self, In, t):
        length = len(In)
        out = []
        if length ==1:
            out.append(In)
            idx = np.argwhere(self.memory[:,self.n_features+2] == t-1)  
            if len(idx) == 0:    #for matlab
                return np.repeat(In, self.continue_length)                
            else:
                idx = int(idx[0,0])
            for i in range(self.continue_length-1):
                temp = self.memory[idx - i,:self.n_features].copy()
                temp = np.reshape(np.array(temp),[1,self.n_features])
                out.append(temp)
        else:
            for i in range(length):
                out.append(np.reshape(np.array(In[i,:]),[1,-1]))
                idx = int(np.argwhere(self.memory[:,self.n_features+2] == t[i]-1 ))
                for j in range(self.continue_length-1):
                    temp = self.memory[idx - i,:self.n_features].copy()
                    temp = np.reshape(np.array(temp),[1,self.n_features])
                    out.append(temp)
        
        out = np.reshape(np.array(out),[length,-1])    
            
        return out
    
    def flatInputSN(self, In, t):
        length = len(In)
        out = []
        if length ==1:
            out.append(In)
            idx = np.argwhere(self.memory[:,self.n_features+2] == t-1)
            if len(idx) == 0:    #for matlab
                out = np.repeat(In, self.continue_length)
                out = np.reshape(np.array(out),[length,-1]) 
#                out = np.matrix(out)
                return out   # reshape        
            else:
                idx = int(idx[0,0])
                
            for i in range(self.continue_length-1):
                temp = self.memory[idx - i,-self.n_features:].copy()
                temp = np.reshape(np.array(temp),[1,self.n_features])
                out.append(temp)
        else:
            for i in range(length):
                out.append(np.reshape(np.array(In[i,:]),[1,-1]))
                idx = int(np.argwhere(self.memory[:,self.n_features+2] == t[i]-1 ))
                for j in range(self.continue_length-1):
                    temp = self.memory[idx - i,-self.n_features:].copy()
                    temp = np.reshape(np.array(temp),[1,self.n_features])
                    out.append(temp)       
        out = np.reshape(np.array(out),[length,-1]) 
#        out = np.matrix(out) #
        

        
        return out

    def store_transition(self,s, a, r, s_, step):
        
        s = np.array( s)
        s_ = np.array( s_)
        
        if not hasattr(self, 'memory_counter'): # check whether self has 'memory_counter'
            self.memory_counter = 0            
        transition = np.zeros((1,self.n_features*2 + 3))
        transition[0,:self.n_features]  = s
        transition[0,self.n_features]   = a
        transition[0,self.n_features+1] = r
        transition[0,self.n_features+2] = step
        transition[0,-self.n_features:] = s_
        # store [s, a, r, trace, s_] 
        # because memory's size is fixed, so the new sample will replace the oldest one.
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition 
        self.memory_counter += 1


    def temp_batch(self,ob):
        index = self.temp_memory_step % self.continue_length
        self.temp_memory_step +=1
        self.temp_memory[index, :] = ob  
        tempbatch = self.temp_memory
        return tempbatch


    def choose_action(self,step, observation):

        observation = np.array( observation)   # new for add in matlab
        observation = observation[np.newaxis, :]
        
        temp_observation = observation[np.newaxis, :]
        

        
        self.exploreHist.append(self.exploreProb)
        if np.random.uniform() < self.exploreProb:   #
            # forward feed the observation and get q value for every actions
            action = np.random.randint(0, self.n_actions)      # ! different
        else:
            actions_value = self.sess.run(self.q_eval0, feed_dict={self.s: self.flatInputS(temp_observation,step),
                                                                  self.temp_batch_size: 1})
            action = np.argmax(actions_value) 
            self.learn_step_counter += 1  
            

        
                      
        self.select_action.append(action)
        return action





    def learn(self):
        
        self.exploreProb = self.exploreInit * \
            np.exp(-self.exploreDecay * self.learn_step_counter )
        if self.exploreProb <= self.exploreProbMin:
            self.exploreProb = self.exploreProbMin  
        #self.exploreHist.append(self.exploreProb)
    
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op) 
            # self.sess.run(_replace_target_parameters())
        
        if self.memory_counter > self.memory_size:
            temp = self.memory_counter % self.memory_size
            if temp <= 400:
                sample_index = np.random.choice(self.memory_size - self.continue_length-temp,size = self.batch_size,replace = False) +self.continue_length+temp
            else:
                sample_index = np.random.choice(temp - self.continue_length,size = self.batch_size,replace = False) +self.continue_length   
        else:
            sample_index = np.random.choice(self.memory_counter - self.continue_length,size = self.batch_size) +self.continue_length
        batch_memory = self.memory[sample_index,:]
        t = batch_memory[:, self.n_features + 2]
   
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
            ###  recall the structure of memory  [s_n_features,action,reward,trace,s_next_n_feature]
                #print(batch_memory[:, :self.n_features])
                self.s:               self.flatInputS(batch_memory[:, :self.n_features], t),
                self.a:               batch_memory[:, self.n_features],
                self.r:               batch_memory[:, self.n_features + 1],
                self.temp_batch_size: self.batch_size,
                self.s_:              self.flatInputSN(batch_memory[:, -self.n_features:], t),
        })
        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        
        return cost
