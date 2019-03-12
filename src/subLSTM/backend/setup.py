from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='subLSTM',
    ext_modules=[CppExtension('subLSTM', ['subLSTM.cpp'])],
    cmdclass={'build_ext': BuildExtension}
)