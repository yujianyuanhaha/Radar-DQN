Ersin - On the Use of MDPs in Cognitive Radar: An Application to Target Tracking  
Date: 7/27/2018  
Author: Jianyuan (Jet) Yu  
Contact: jianyuan@vt.edu   
Affiliate: Wireless, ECE, Virginia Tech


Table of Contents
=================

   * [Overview](#overview)
   * [Setup](#setup)
   * [Codes](#codes)
   * [News](#news)
   * [Bugs](#bugs)
   * [Further Works](#further-works)

-------------------------------------------------------------------------
# Overview
The work replaces the mdp solver in Ersin work [git link](git@git.ece.vt.edu:cognitive-radar/code.git) as dqn(deep q network) solver. The codes are mostly written in .m except for the ```dqn.py``` file, and python files are mostly based on open source [RL with Tensorflow by Morvan Zhou](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow).   
It is NOT preferable to run codes in this blended way due to the low efficient computation and difficult to adjust the parameters within the dqn solver. Moreover, these codes may bring the new compatible problem when moving to ARC platform.

# Setup
Refer to [READMEconfig.md](https://github.com/yujianyuanhaha/Ersin/blob/master/READMEconfig.md)

# Codes
Refer to [READMEcodes.md](md)


# News
1. 10/22 adding DRQN
2. 9/6 add in dpg(deep policy gradient) method, which apt for episode model.
3. 9/1 dqn get same result as mdp.
4. 7/17 the codes running through.


# Bugs
1. ~~unmatch of ```State``` and ```NumBands```, i.e. the NumBands in 5 in the default case, while the State is 1 times 4 vector instead of 1 times 5.~~
2. ~~the ```CurrentActionNumber``` outrange, probably comes from the first one, I just did roll over.~~


# Further Works 🚧
1. fix DoubleDQN PriDQN DuelDQN
2. add DRQN