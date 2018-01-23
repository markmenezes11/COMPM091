# Context Vectors (CoVe) - Keras Port, Ported to Python2

This folder contains a variation of the [Keras port](https://github.com/rgsachin/CoVe) of [CoVe](https://github.com/salesforce/cove) (see `LICENSE`).

The script `port_python3_to_python2.py` takes the [Keras port](https://github.com/rgsachin/CoVe) of [CoVe](https://github.com/salesforce/cove) and converts it into a Python2-compatible version.

You should not have to re-run the script since the ported model has already been generated, but if you do, here are some instructions...

## Requirements

To generate the Python2-ported model, you will need Python2, with `keras==2.1.3`, `h5py` and `tensorflow` libraries (installed via `pip` or otherwise).

The Singularity image provided by this repository also has all of these requirements installed, so if you are using Singularity it should be easy.

## Run

To run it, you will need the Keras_CoVe.h5 file from the [Keras CoVe](https://github.com/rgsachin/CoVe) repository. The output is Keras_CoVe_Python2.h5 (which is already contained in this folder so you should not have to re-run the script).

Place Keras_CoVe.h5 in this folder and run `python port python3_to_python2`, using a Python2 install of Python.

## Test

To test the vectors produced, you can run `python port_test_python2.py` and `python3 port_test_python3.py` and compare the vectors produced. They should be the same. You will need SentEval and its requirements (see the README in the parent folder).

## Reference 
1. [MT-LSTM from the paper Learned in Translation: Contextualized Word Vectors (McCann et. al. 2017)](https://arxiv.org/abs/1708.00107)
2. [MT-LSTM PyTorch implementation from which weights are ported](https://github.com/salesforce/cove)
3. [GloVe](https://nlp.stanford.edu/projects/glove/)
