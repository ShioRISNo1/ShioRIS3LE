#!/usr/bin/env bash
# Setup script for ShioRIS3 AutoSegmentation environment on Linux
set -e

# System dependencies
sudo apt update
sudo apt install -y build-essential cmake qt6-base-dev libopencv-dev libdcmtk-dev
sudo apt install -y python3 python3-pip python3-venv nvidia-cuda-toolkit

# Python environment
python3 -m venv ~/shioris3_ai_env
source ~/shioris3_ai_env/bin/activate
pip install --upgrade pip
pip install TotalSegmentator torch torchvision onnxruntime-gpu

# Download models for offline use
TotalSegmentator --download_models
