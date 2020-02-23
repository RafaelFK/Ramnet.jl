#!/usr/bin/env python

"""
setup.py file for class wrapping example
"""

from distutils.core import setup, Extension


ram_node_module = Extension(
  '_ram_node',
  sources=['./wrapper/src/ram_node_wrap.cxx', './lib/ram_node.cpp', './lib/binary.cpp'],
  extra_compile_args=["-std=c++14", "-I./include"]
)

setup (
  name        = 'ram_node',
  version     = '0.1',
  author      = "Rafael F. Katopodis",
  description = """Simple implementation of a RAM-based neuron""",
  ext_modules = [ram_node_module],
  py_modules  = ["ram_node"],
)