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
1. ```t``` or ```TimeSteps``` - time duration, default set as ```150```,
2. ```NumTrainingRuns``` -  offline training case number, default set as ```6000```.
3. ```NumBands``` - num of bands, notice to **increase the interference pattern as well when you adjust the NumBands**, like NumBands = 5 with   [1 0 0 0 0 0 0 0 0 0], while NumBands = 10 with   [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ].

# Config Neural Networks
These descriptions are also updated in related .py files.
1. parameter description table.  

| parameter name    | meaning   | default value      | extra   |
|----------------|-----------|----------|----------------|
| exploreDecay             |explore rate decay rate, for adjust choose random action         | 0.001        | -              |  
| exploreProbMin          |explore prob min         | 0.01       | -              |
| learning_rate            | learning rate of RMS Optimizer         | 0.001        | -              |
| reward_decay            | reward decay when calculate Q value         | 0.9       | -              |
| replace_target_iter            | iteration count to refresh iteration network         | 300        | -              |
| memory_size            | the count of tuple (s,a,r,s_)         | 1000        | -              |
|batch_size            | batch  size when calculate the neural network         | 32        | -              |

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

# Notice
1. Ersin code default generates file at the local directory, it would result in a weird error like "script not found" if these file increase to hundreds. Hence we prefer to set the directory somewhere else.
2. restart Matlab and reload .py file each time when editing the ```dqn.py``` file.
3. restart Matlab and reload .py file each time when NumBand changes.(the tensorflow REUSE bugs to be solved later)
4. apply ```int32()``` when pass value from .m to .py
5. apply ```np.array()``` when receive array value from .m to .py
6. [PyException](https://www.mathworks.com/matlabcentral/answers/170466-python-from-2014b-matlab-debug-challanges-where-is-python-stdout-stderr) with try-catch sentence may provide inner information of bugs during debugging.