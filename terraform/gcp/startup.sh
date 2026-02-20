#!/bin/bash
set -e

# Install CUDA 12.0
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get install -y cuda-12-0

# Install dependencies
apt-get install -y cmake git python3-pip

# Clone and build Quila
cd /opt
git clone https://github.com/your-org/quila.git
cd quila
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Install Python dependencies
pip3 install -r requirements.txt

# Start API server
cd src/python/api
nohup python3 server.py &
