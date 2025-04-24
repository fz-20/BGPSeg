#python3 setup.py install
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

setup(
    name='boundaryops',
    author='Zheng Fang',
    ext_modules=[
        CUDAExtension('boundaryops_cuda', [
            'src/boundaryops_api.cpp',
            'src/boundaryquery/boundaryquery_cuda.cpp',
            'src/boundaryquery/boundaryquery_cuda_kernel.cu',
            ],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
