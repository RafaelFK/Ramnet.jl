#!/usr/bin/env python

"""
setup.py file for class wrapping example
"""

from setuptools import setup, Extension, find_packages
from pathlib import Path

def abs_str_path(relative_str_path):
  return str(Path(relative_str_path).resolve())


ram_node_module = Extension(
  '_ram_node',
  sources=[
    abs_str_path('../src/ram_node_wrap.cxx'),
    abs_str_path('../../lib/ram_node.cpp'),
    abs_str_path('../../lib/binary.cpp')
  ],
  extra_compile_args=["-std=c++14", "-I../../include"]
)

dense_ram_node_module = Extension(
  '_dense_ram_node',
  sources=[
    abs_str_path('../src/dense_ram_node_wrap.cxx'),
    abs_str_path('../../lib/dense_ram_node.cpp'),
    abs_str_path('../../lib/binary.cpp')
  ],
  extra_compile_args=["-std=c++14", "-I../../include"]
)

sparse_ram_node_module = Extension(
  '_sparse_ram_node',
  sources=[
    abs_str_path('../src/sparse_ram_node_wrap.cxx'),
    abs_str_path('../../lib/sparse_ram_node.cpp'),
    abs_str_path('../../lib/binary.cpp')
  ],
  extra_compile_args=["-std=c++14", "-I../../include"]
)

ram_discriminator_module = Extension(
  '_ram_discriminator',
  sources=[
    abs_str_path('../src/ram_discriminator_wrap.cxx'),
    abs_str_path('../../lib/dense_ram_node.cpp'),
    abs_str_path('../../lib/ram_discriminator.cpp'),
    abs_str_path('../../lib/binary.cpp')
  ],
  extra_compile_args=["-std=c++14", "-I../../include"]
)

setup (
  name        = 'ramnet',
  version     = '0.1',
  author      = "Rafael F. Katopodis",
  description = """Simple implementation of a RAM-based neuron""",
  ext_modules = [
    dense_ram_node_module,
    sparse_ram_node_module,
    ram_discriminator_module
  ]
)