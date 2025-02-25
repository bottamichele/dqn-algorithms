# DQN-Algorithms

## About
This repository is my personal library which contains the implementation in Python languange of 
Deep Q-Networks (DQN) and its some variants with the goal of use one of these algorithms 
on my other projects.

This library includes:
- Deep Q-Networks (MLP and CNN version)
- Double Deep Q-Networks (MLP and CNN version)
- Dueling Deep Q-Networks (MLP and CNN version)
- Dueling Double Q-Networks (MLP and CNN version)
- support for target network
- uniform memory replay
- proportional prioritized memory replay

# Installation
This library must be installed locally to be used. You need to do the following steps to install it:
1. download this repository either from this page or by running `git clone https://github.com/bottamichele/dqn-algorithms`
2. run `pip install <path_repository>` (***<path_repository>*** is path of the repository which you downloaded)
   to install the library with its all dependecies and compiling its C modules.

**DISCLAIMAR:** the compilation of C modules is tested only on Python 3.10.13 and numpy 1.26.3. 
                However, different versions should be work.

## Get started
To understand how to use this library a good start point is to go to the folder `test`, 
which is used for test the library, contains important things how to setup agents and train them
by using DQN or its any variant.

## License
This library is licensed under the MIT License. For more information about, 
see [LICENSE](https://github.com/bottamichele/dqn-algorithms/blob/main/LICENSE).
