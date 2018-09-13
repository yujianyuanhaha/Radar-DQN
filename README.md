Ersin - On the Use of MDPs in Cognitive Radar: An Application to Target Tracking  
Date: 7/27/2018  
Author: Jianyuan (Jet) Yu  
Contact: jianyuan@vt.edu   
Affiliate: Wireless, ECE, Virginia Tech


Table of Contents
=================

   * [Overview](#overview)
   * [Setup](#setup)
   * [How to run](#how-to-run)
   * [Notice](#notice)
   * [News](#news)
   * [Bugs](#bugs)
   * [Further Works](#further-works)

-------------------------------------------------------------------------
# Overview
The work replaces the mdp solver in Ersin work [git link](git@git.ece.vt.edu:cognitive-radar/code.git) as dqn(deep q network) solver. The codes are mostly written in .m except for the ```dqn.py``` file, and python files are mostly based on open source [RL with Tensorflow by Morvan Zhou](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow).   
It is NOT preferable to run codes in this blended way due to the low efficient computation and difficult to adjust the parameters within the dqn solver. Moreover, these codes may bring the new compatible problem when moving to ARC platform.


# Setup
1. version: matlab 2018a, python 2.7, tensorflow 1.6.0
2. add in python & tensorflow into matlab path:  
in matlab terminal 
``` 
py.sys.path
insert(py.sys.path,int32(0),'/PATH_of_Tensorflow')
```  
the ```PATH_of_Tensorflow``` depend on where you install them, e.g. in my mac OS, they look like
```
insert(py.sys.path,int32(0),'/anaconda2/lib/python2.7/')
insert(py.sys.path,int32(0),'/anaconda2/lib/python2.7/site-packages')
```
after that, type in
```
py.importlib.import_module('dqn')
```
it will work if no error pop up.  







# How to run
## Mac OS 
**first setup**  refer to setup step 1-2
1. start matlab from terminal(by the [guide](https://stackoverflow.com/questions/45733111/importing-tensorflow-in-matlab-via-python-interface)) from the PATH where /bin of Matlab is, in my Mac OS, it is like change to path ```/Application/MATLAB_R2018a.app/bin```
2. after Matlab launch, change path to where ```dqn.py``` file is locate, type in 
    ```
    py.importlib.import_module('dqn')
    ``` 
to load the python module.  
3. execute ```RunSimulations.m``` file, notice replace the 'dqn' as 'mdp' solver(last second option) recurrent to Ersin mdp settings.
It seems Matlab2018a easily crash down, thus we save the figure asap. To further save the command windows text(like running time or errors) by command [diary](https://www.mathworks.com/help/matlab/ref/diary.html), by doing so we can look up the logs of terminal by the ```diary``` file.

---
## Ubuntu 
(Matlab version 2018a) User  **first setup** refer to setup step 1-2 as well, and then follow below  
1. downgrade tensorflow to version 1.5.0 to avoid kernel dead error in ubuntu, in terminal  
    ```
    conda install -c conda-forge tensorflow=1.5.0
    ```
2. replace(actually work as update) the ```libstdc++.so.6```,```libstdc++.so.6.0.22``` of matlab as the those ```libstdc++.so.6```,```libstdc++.so.6.0.22``` of anaconda.  
in terminal, type in command as below. The second command just archived the old lib, and third copy the new one from anaconda, similiar to the other lib.   
    ```
    cd /usr/local/MATLAB/R2018a/sys/os/glnxa64/
    sudo mv libstdc++.so.6 libstdc++.so.6.old
    sudo cp ~/anaconda2/lib/libstdc++.so.6 .
    sudo mv libstdc++.so.6.0.22 libstdc++.so.6.0.22.old
    sudo cp ~/anaconda2/lib/libstdc++.so.6.24 .
    ```

3. start matlab from terminal(same as Mac OS step 1)  
4. after Matlab launch, change path to where ```dqn.py``` file is locate(same as Mac OS step 2)
5. in Matlab, type in command
    ```
    py.sys.setdlopenflags(int32(10))
    ```
6. load the python module (same as Mac OS step 2)
7. execute ```RunSimulations.m``` file.  
After the first setup pass, next time we start the matlab from terminal, change path to  where ```dqn.py```  is, and run ```RunSimulations.m```.  
~~1. change directory to where ```dqn.py```  is~~  
~~2. for ubuntu in matlab type in  ```py.sys.setdlopenflags(int32(10))```, for Mac OS skip this step~~  
~~3. then in matlab type in  ``` py.importlib.import_module('dqn')```~~
-------------------------------------------------------------------------



# Running on ARC
Recommand to run on ARC when large computation, we can lauch MATLAB in ARC as below. Notice matlab(free with campus license) is installed on ```cascade``` while not on ```huckleberry```, and we running codes on GUI (meaning remote GUI helper app like ```XQuartz``` is needed).
1. If you are out of campus, login in *VT VPN*.
2. login in cascade like ```ssh -Y -C yourname@cascades2.arc.vt.edu```, where ```-Y``` for macOS and ```-X```
for Windows/Linux,  
3. after login, find where matlab is by ```module spider matlab```,
4. select right version by ```module load matlab/R2018a```,
5. load exact python 2.7 and tensorflow 1.5.0 version by create **conda virtual environment**.    
    ```
        module purge
        module load Anaconda/2.3.0
        module load gcc
        conda create --name RadarDQN python=2.7
        source activate RadarDQN
        pip install --ignore-installed --upgrade  https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.5.1-cp27-none-linux_x86_64.whl

    ```
    where *RadarDQN* is self-defined name.  

6. then type ```matlab``` to start, after that ```XQuartz``` is launched and then Matlab is launched.
7. after that, remember to add in the PATH of tensorflow by
    ``` insert(py.sys.path,int32(0),'~/.conda/envs/RadarDQN/lib/python2.7/site-packages/') ```, and it would work if no error pop after the test ```py.importlib.import_module('dqn')```.   
8. excute local codes, or get the updated codes by ```scp``` or ```git pull```.  
I will update the guidence once *mcc* compiler go through and *pbs* on ARC works.  

Notice:
1. Before ```git pull```, make ``` git clone ``` first. An easier way is clone in the **https** way rather than ssh, while for ssh way, please [set up public key](https://help.github.com/articles/error-permission-denied-publickey/#platform-linux) ahead. Moreover, every time you do ```git pull```, you may need to remove all ```.pyc``` fiels generated  late time.
2. For ```scp``` command, double check the PATH of the remote server. e.g. ``` scp note.txt jianyuan@cascades1.arc.vt.edu:/home/jianyuan```.

# Run Maltab  in terminal without GUI
Matlab support running in terminal without GUI both for Mac and Ubuntu.
For Mac, excute ```/MATLAB_PATH/matlab -nodesktop -nosplash -r "FILE_NAME.m" ``` e.g ```/Applications/MATLAB_R2018a.app/bin/matlab -nodesktop -nosplash -r "RunSimulations.m" ```. After that, go on type in ```FILE_NAME``` after ```>>``` show up.  
While for Ubuntu, after load matlab by ```module load matlab/R2018a```, excute like ```matlab -nodesktop -nosplash -r "FILE_NAME.m" ```, where only difference is path name is not necessary needed.  
By default, plots will pop up when codes run to the end, and type ``` edit FILE_NAME.m``` can edit related files.

# Codes Description

## overview
mdp process combine of **offline training** and **online evaluation**, while dqn could be directly online training and evaluation.

## parameters
1. ```t``` or ```TimeSteps``` - time duration, default set as ```150```,
2. ```NumTrainingRuns``` -  offline training case number, default set as ```6000```.



# Notice
1. Ersin code default generate file at local directory, it would result in weired error like "script not found" if these file increase to hunreds. Hence we prefer to set the directory somewhere else.
2. restart Matlab and reload .py file each time when editing the ```dqn.py``` file.
3. restart Matlab and reload .py file each time when NumBand changes.(the tensorflow REUSE bugs to be solved later)
4. apply ```int32()``` when pass value from .m to .py
5. apply ```np.array()``` when receive array value from .m to .py
6. [PyException](https://www.mathworks.com/matlabcentral/answers/170466-python-from-2014b-matlab-debug-challanges-where-is-python-stdout-stderr) with try-catch sentence may provide inner information of bugs during debugging.

# News
1. 9/6 add in dpg(deep policy gradient) method, which apt for episode model.
2. 9/1 dqn get same result as mdp.
3. 7/17 the codes running through.


# Bugs
1. ~~unmatch of ```State``` and ```NumBands```, i.e. the NumBands in 5 in the default case, while the State is 1 times 4 vector instead of 1 times 5.~~
2. ~~the ```CurrentActionNumber``` outrange, probably comes from the first one, I just did roll over.~~



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


# Further Works
TODO