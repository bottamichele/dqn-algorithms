import numpy as np

from setuptools import setup, Extension

#Modules
sumtree_module = Extension("dqn_algorithms._c_modules.sum_tree", ["./dqn_algorithms/_c_modules/src/sum_tree.c"], [np.get_include()])

#Setup
setup(ext_modules=[sumtree_module])