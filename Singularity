# vim :set ts=8 sw=4 sts=4 et:

bootstrap: docker
from: nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04

%post
    export DEBIAN_FRONTEND=noninteractive
    echo 'deb http://archive.ubuntu.com/ubuntu xenial universe' >> /etc/apt/sources.list
    apt-get update
    apt-get install -y python build-essential gfortran libatlas-base-dev python-pip python-dev python3 python3-pip vim curl unzip cabextract git
    apt-get clean
    
    pip install --upgrade pip
    pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
    pip install torchvision nltk numpy scipy scikit-learn

%test
    python -c 'import torch'
