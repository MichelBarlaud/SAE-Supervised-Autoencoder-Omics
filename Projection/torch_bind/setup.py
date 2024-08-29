from setuptools import setup, Extension
import numpy
from torch.utils import cpp_extension
# Setup for PC

#setup(name='projections',
 #     ext_modules=[cpp_extension.CppExtension('projections', ['projections.cpp'])],
  #    cmdclass={'build_ext': cpp_extension.BuildExtension}, include_dirs = ['.','../code/include'])
  
#Setup for MAC M3 processor
  
setup(name='projections',
      ext_modules=[cpp_extension.CppExtension('projections', ['projections.cpp'], extra_compile_args = ['-mmacosx-version-min=10.13'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension}, include_dirs = ['.','../code/include'])
