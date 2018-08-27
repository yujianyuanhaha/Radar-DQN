"""
This part of code is the Deep Q Network (DQN) brain.

view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: r1.2
"""

import numpy as np
#import tensorboard as tb
import tensorflow as tf

import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   

# to avoid the warning Your CPU supports instructions that this TensorFlow binary 
# was not compiled to use: AVX2 FMA

np.random.seed(1)
tf.set_random_seed(1)

# Deep Q Network off-policy
class dqn:
    
    exploreProb      = [ ]              # Current exploration probability
    exploreInit      = 1.0              # Initial exploration probability
    exploreDecay     = 0.001              # Percentage reduction in exploration chance per policy calculation
    exploreProbMin   = 0.1  # avoid the risk to stuck
    exploreHist      = [ ]    
    exploreDecayType = 'expo'           # either 'expo', 'step' or 'perf'
    exploreWindow    = 500              # only used with 'step'
    exploreMin       = 0.1             # only used with 'step'    
    explorePerf      = 10               # only used with 'perf' 
    explorePerfWin   = 100
    e_greedy            = 0.9              
    # triggers jump in explore prob to 1 if reward is below this over last explorePerfWin epoch   
        
    def __init__(
            self,
            dqnNode,
            n_actions,
            n_features,
            learning_rate       = 0.001,    # for neural network
            reward_decay        = 0.9,
            exploreDecayType    = 'expo',   
            replace_target_iter = 300,
            memory_size         = 1000,
            batch_size          = 32,
            e_greedy_increment  = None,
            output_graph        = True      # enable tensorboard                  
    ):    # allow dqnNode to call in its attribute
                
        self.n_actions           = n_actions
        self.n_features          = n_features
        self.lr                  = learning_rate
        self.gamma               = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size         = memory_size
        self.batch_size          = batch_size
                
        self.exploreDecayType    = exploreDecayType
        self.epsilon_max         = self.e_greedy  
        self.epsilon_increment   = e_greedy_increment 
        self.epsilon             = 0 if e_greedy_increment is not None else self.epsilon_max       
        self.exploreProb         = self.exploreInit
        self.learn_step_counter  = 0

        # initialize zero memory [s, a, r, s_]

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        
            
        # tensorflow
        with tf.variable_scope('eval_net', reuse=tf.AUTO_REUSE):
            with tf.name_scope('weight'):
                weight = t_params[0]
                tf.summary.histogram('e1' + '/weight', weight)
            with tf.name_scope('biases'):
                biases = t_params[1]
                tf.summary.histogram('e1' + '/biases', biases)
            
        self.sess = tf.Session()
        merged = tf.summary.merge_all()

      #  self.sess.run(merged)   #stuck
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        
        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("../logs/", self.sess.graph)        

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        # !!! the cal of n_features, the length of state is the size of n_feature        
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------


        with tf.variable_scope('eval_net', reuse=tf.AUTO_REUSE):
            e1 = tf.layers.dense(self.s, 50, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1,     50, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
#            e3 = tf.layers.dense(e2,     200, tf.nn.relu, kernel_initializer=w_initializer,
#                                 bias_initializer=b_initializer, name='e3')
#            e4 = tf.layers.dense(e3,     20, tf.nn.relu, kernel_initializer=w_initializer,
#                                 bias_initializer=b_initializer, name='e4')
            
            self.q_eval = tf.layers.dense(e2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')


        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net',reuse=tf.AUTO_REUSE):
            t1 = tf.layers.dense(self.s_, 50, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 50, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
#            t3 = tf.layers.dense(t2, 200, tf.nn.relu, kernel_initializer=w_initializer,
#                                 bias_initializer=b_initializer, name='t3')
#            t4 = tf.layers.dense(t3, 20, tf.nn.relu, kernel_initializer=w_initializer,
#                                 bias_initializer=b_initializer, name='t4')
            self.q_next = tf.layers.dense(t2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t5')
        with tf.variable_scope('q_target',reuse=tf.AUTO_REUSE):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval',reuse=tf.AUTO_REUSE):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss',reuse=tf.AUTO_REUSE):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):

        s = np.array( s)
        s_ = np.array( s_)

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = np.array( observation)   # new for add in matlab
        observation = observation[np.newaxis, :]

        if np.random.uniform() < 1.0 - self.exploreProb:   #
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            # size of observation = s / n_feature
            action = np.argmax(actions_value) 
            self.learn_step_counter += 1            
        else:
            action = random.randint(0, self.n_actions-1)      
#        print action
        return action


    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
           # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })   # be sharp, man

        self.cost_his.append(cost)

        if self.exploreDecayType == 'expo':
            self.exploreProb = self.exploreInit * \
                np.exp(-self.exploreDecay * self.learn_step_counter )
            if self.exploreProb <= self.exploreProbMin:
                self.exploreProb = self.exploreProbMin                
            self.learn_step_counter += 1
        elif self.exploreDecayType == 'incre':
            self.epsilon = self.epsilon + self.epsilon_increment \
                    if self.epsilon < self.epsilon_max else self.epsilon_max            
            self.exploreProb  = 1 -  self.epsilon


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
        

if __name__ == '__main__':
    #from dqnNode import dqnNode
    # dqnNode_ = dqnNode(2,[[0,0],[0,1],[1,0],[1,1]],1000,5)
    # t = dqnNode(numChans,states,numSteps, nodeTypes[k])
    # dqnNode_ = None
    dqn_ = dqn(None, 3,2, output_graph=True)
    dqn_2 = dqn(0, 3,2, output_graph=True)
    print "order matters"