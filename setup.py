#!/usr/bin/env python

"""
setup.py file for class wrapping example
"""

from setuptools import setup, Extension


ram_node_module = Extension(
  '_ram_node',
  sources=['./wrapper/src/ram_node_wrap.cxx', './lib/ram_node.cpp', './lib/binary.cpp'],
  extra_compile_args=["-std=c++14", "-I./include"]
)

dense_ram_node_module = Extension(
  '_dense_ram_node',
  sources=['./wrapper/src/dense_ram_node_wrap.cxx', './lib/dense_ram_node.cpp', './lib/binary.cpp'],
  extra_compile_args=["-std=c++14", "-I./include"]
)

sparse_ram_node_module = Extension(
  '_sparse_ram_node',
  sources=['./wrapper/src/sparse_ram_node_wrap.cxx', './lib/sparse_ram_node.cpp', './lib/binary.cpp'],
  extra_compile_args=["-std=c++14", "-I./include"]
)

ram_discriminator_module = Extension(
  '_ram_discriminator',
  sources=[
    './wrapper/src/ram_discriminator_wrap.cxx',
    './lib/dense_ram_node.cpp',
    './lib/ram_discriminator.cpp',
    './lib/binary.cpp'
  ],
  extra_compile_args=["-std=c++14", "-I./include"]
)

setup (
  name        = 'ram_node',
  version     = '0.1',
  author      = "Rafael F. Katopodis",
  description = """Simple implementation of a RAM-based neuron""",
  ext_modules = [
    ram_node_module,
    dense_ram_node_module,
    sparse_ram_node_module,
    ram_discriminator_module
  ],
  py_modules  = ["ram_node"],
)