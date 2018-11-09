Ersin - On the Use of MDPs in Cognitive Radar: An Application to Target Tracking  
Date: 7/27/2018  
Author: Jianyuan (Jet) Yu  
Contact: jianyuan@vt.edu   
Affiliate: Wireless, ECE, Virginia Tech

Table of Contents
=================

* [Codes Description](#codes-description)
    * [overview](#overview)
    * [parameters](#parameters)
* [Config Neural Networks](#config-neural-networks)
* [Notice](#notice)


# Codes Description

## overview
mdp process combine of **offline training** and **online evaluation**, while dqn could be directly online training and evaluation.

## parameters
1. _`t`_ or _`TimeSteps`_ - time duration, default set as ```150```,
2. _`NumTrainingRuns`_ -  offline training case number, default set as ```6000```.
3. _`NumBands`_ - num of bands, notice to **increase the interference pattern as well when you adjust the NumBands**, like NumBands = 5 with   [1 0 0 0 0 0 0 0 0 0], while NumBands = 10 with   [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ].

# Config Neural Networks
These descriptions are also updated in related .py files.
1. parameter description table.  

| parameter name    | meaning   | default value      | extra   |
|----------------|-----------|----------|----------------|
| _`exploreDecay`_             |explore rate decay rate, for adjust choose random action         | 0.001        | -              |  
| _`exploreProbMin`_          |explore prob min         | 0.01       | -              |
| _`learning_rate`_            | learning rate of RMS Optimizer         | 0.001        | -              |
| _`reward_decay`_            | reward decay when calculate Q value         | 0.9       | -              |
| _`replace_target_iter`_            | iteration count to refresh iteration network         | 300        | -              |
| _`memory_size`_            | the count of tuple (s,a,r,s_)         | 1000        | -              |
|_`batch_size`_            | batch  size when calculate the neural network         | 32        | -              |

2. config layers  
e.g. from default *eval_net* 50X50( two-layer, first layer 50, second layer 50) to 
```
    e1 = tf.layers.dense(self.s, 50, tf.nn.relu, kernel_initializer=w_initializer,
                            bias_initializer=b_initializer, name='e1')
    e2 = tf.layers.dense(e1,     50, tf.nn.relu, kernel_initializer=w_initializer,
                            bias_initializer=b_initializer, name='e2')    
    self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='q')
```
new layer like 50X50X200X20  
```
    e1 = tf.layers.dense(self.s, 50, tf.nn.relu, kernel_initializer=w_initializer,
                            bias_initializer=b_initializer, name='e1')
    e2 = tf.layers.dense(e1,     50, tf.nn.relu, kernel_initializer=w_initializer,
                            bias_initializer=b_initializer, name='e2')
    e3 = tf.layers.dense(e2,     200, tf.nn.relu, kernel_initializer=w_initializer,
                            bias_initializer=b_initializer, name='e3')
    e4 = tf.layers.dense(e3,     20, tf.nn.relu, kernel_initializer=w_initializer,
                            bias_initializer=b_initializer, name='e4')
    
    self.q_eval = tf.layers.dense(e4, self.n_actions, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='q')
```
Besides, we can also config *target_net* layers.


# Config DRQN
| parameter name    | meaning   | default value      | extra   |
|----------------|-----------|----------|----------------|
| _`continue_length`_             | todo        | 10        | -              |  
| _`n_hidden_units`_          |hidden layer units         | 40       | -              |
And notice DRQN generally require slower _`exploreDecay`_ than DQN.  

# Notice
1. Ersin code default generates file at the local directory, it would result in a weird error like `script not found` if these file increase to hundreds. Hence we prefer to set the directory somewhere else.
2. restart Matlab(or `Open additional instance of Matlab` of right click matlab icon when you PC got enough memory) and reload .py file each time when editing the `dqn.py` file.
3. restart Matlab(or `Open additional instance of Matlab` of right click matlab icon when you PC got enough memory) and reload .py file each time when NumBand changes.(the tensorflow REUSE bugs to be solved later)
4. apply _`int32()`_ when pass value from .m to .py
5. apply _`np.array()`_ when receive array value from .m to .py
6. [PyException](https://www.mathworks.com/matlabcentral/answers/170466-python-from-2014b-matlab-debug-challanges-where-is-python-stdout-stderr) with try-catch sentence may provide inner information of bugs during debugging.