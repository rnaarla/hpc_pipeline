from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="hpc-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "hpc-orchestrator=orchestrator:main"
        ]
    },
    ext_modules=[
        CUDAExtension(
            name="custom_kernels",
            sources=[
                "optimization/kernels.cpp",
                "optimization/custom_kernels.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3", "-march=native", "-ffast-math"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_80,code=sm_80",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-maxrregcount=128",
                    "--extra-device-vectorization"
                ]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)