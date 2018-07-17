# Ersin - On the Use of MDPs in Cognitive Radar: An Application to Target Tracking
Date: 7/27/2018  
Author: Jianyuan (Jet) Yu  
Contact: jianyuan@vt.edu   
Affiliate: Wireless, ECE, Virginia Tech

-------------------------------------------------------------------------
# Overview
The work replaces the mdp solver in Ersin work as dqn(deep q network) solver. The codes are mostly written in .m except for the ```dqn.py``` file.  
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
refer to setup step 1-2
1. start matlab from terminal(by the [guide](https://stackoverflow.com/questions/45733111/importing-tensorflow-in-matlab-via-python-interface)) from the PATH where /bin of Matlab is, in my Mac OS, it is like change to path ```/Application/MATLAB_R2018a.app/bin```
2. after Matlab launch, change path to where ```dqn.py``` file is locate, type in 
```
py.importlib.import_module('dqn')
``` 
to load the python module.  
3. execute ```RunSimulations.m``` file

-------------------------------------------------------------------------

# Notice
1. restart Matlab and reload .py file each time when editing the ```dqn.py``` file.
2. restart Matlab and reload .py file each time when NumBand changes.(the tensorflow REUSE bugs to be solved later)
3. apply ```int32()``` when pass value from .m to .py
4. apply ```np.array()``` when receive array value from .m to .py
5. [PyException](https://www.mathworks.com/matlabcentral/answers/170466-python-from-2014b-matlab-debug-challanges-where-is-python-stdout-stderr) with try-catch sentence may provide inner information of bugs during debugging.

# News
1. 7/17 the codes running through.


# Bugs 🐞
1. unmatch of ```State``` and ```NumBands```, i.e. the NumBands in 5 in the default case, while the State is 1 times 4 vector instead of 1 times 5.
2. the ```CurrentActionNumber``` outrange, probably comes from the first one, I just did roll over.


# Further Work
TODO