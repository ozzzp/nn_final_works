from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Distutils import build_ext

ext_modules = [Extension("Othello_ext", sources=["Othello_ext.pyx"], include_dirs=[np.get_include()])]

setup(
    name='Othello_ext',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
