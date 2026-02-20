from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "quila_core",
        ["src/python/bindings/bindings.cpp"],
        include_dirs=["src"],
        libraries=["quila_core"],
        library_dirs=["build"],
    ),
]

setup(
    name="quila",
    version="0.1.0",
    author="Sintellix",
    description="Quila (QualiaTrace LM)",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "protobuf>=3.21.0",
        "pybind11>=2.11.0",
        "fastapi>=0.100.0",
        "websockets>=11.0",
        "deepspeed>=0.10.0",
        "hnswlib>=0.7.0",
        "zstandard>=0.21.0",
    ],
)
