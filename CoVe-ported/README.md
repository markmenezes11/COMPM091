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
- [MT-LSTM PyTorch implementation from which weights are ported](https://github.com/salesforce/cove)
- [Conneau, Alexis, Kiela, Douwe, Schwenk, Holger, Barrault, Loïc, and Bordes, Antoine. Supervised learning of universal sentence representations from natural language inference data. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 670–680. Association for Computational Linguistics, 2017.](https://arxiv.org/pdf/1705.02364.pdf)
- [McCann, Bryan, Bradbury, James, Xiong, Caiming, and Socher, Richard. Learned in translation: Contextualized word vectors. In Advances in Neural Information Processing Systems 30, pp. 6297–6308. Curran Associates, Inc., 2017.](https://arxiv.org/pdf/1708.00107.pdf)
- [Pennington, Jeffrey, Socher, Richard, and Manning, Christopher D. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pp. 1532–1543, 2014.](https://nlp.stanford.edu/pubs/glove.pdf)
- Martin Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mane, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viegas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.
- Pytorch. [online]. Available at: https://github.com/pytorch/pytorch.
- Chollet, Francois et al. Keras. [online]. Available at: https://github.com/keras-team/keras, 2015.
- Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In Conference on Empirical Methods in Natural Language Processing (EMNLP 2013).
