from setuptools import setup, Extension
import torch
from torch.utils import cpp_extension

setup(name='l1inftyB_cpp',
      ext_modules=[cpp_extension.CppExtension('l1inftyB_cpp', ['l1inftyB.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})