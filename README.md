# DQN-Algorithms

## About
This repository is my personal library which contains the implementation in Python languange of 
Deep Q-Networks (DQN) with its some variants and its goal is to use one of these algorithms 
in my other projects.

This library includes:
- Deep Q-Networks (MLP and CNN version)
- Double Deep Q-Networks (MLP and CNN version)
- Dueling Deep Q-Networks (MLP and CNN version)
- Dueling Double Q-Networks (MLP and CNN version)
- support for target network
- uniform memory replay
- proportional prioritized memory replay
- N-step learning support

# Installation
This library must be installed locally to be used. You need to do the following steps to install it:
1. download this repository either from this page or by running `git clone https://github.com/bottamichele/dqn-algorithms`
2. run `pip install <path_repository>` (***<path_repository>*** is path of the downloaded repository)
   to install the library with all its dependecies and compiling its C modules.

**WARNING:** the compilation of C modules is only tested with Python v3.10.13 and v3.10.16 and numpy v1.26.3 and v2.1.2. 
             However, different versions should be work.

## License
This library is licensed under the MIT License. For more information about, 
see [LICENSE](https://github.com/bottamichele/dqn-algorithms/blob/main/LICENSE).
