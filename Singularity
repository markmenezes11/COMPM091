# vim :set ts=8 sw=4 sts=4 et:

bootstrap: docker
from: nvidia/cuda:8.0-cudnn7-runtime-ubuntu16.04

%post
    export DEBIAN_FRONTEND=noninteractive
    echo 'deb http://archive.ubuntu.com/ubuntu xenial universe' >> /etc/apt/sources.list
    apt-get update
    apt-get install -y python python-pip python3 python3-pip
    apt-get clean

    pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
    pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl

%test
    python -c 'import torch'
    python2 -c 'import torch'
    python3 -c 'import torch'
