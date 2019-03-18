from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='sublstm',
    ext_modules=[CppExtension('sublstm', ['subLSTM.cpp'])],
    cmdclass={'build_ext': BuildExtension}
)