import numpy as np

from setuptools import setup, Extension

#Modules
sumtree_module = Extension("sum_tree", ["./dqn/c_modules/src/sum_tree.c"], [np.get_include()])

#Setup
setup(ext_modules=[sumtree_module])