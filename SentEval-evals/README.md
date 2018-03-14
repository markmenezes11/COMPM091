# Evaluating Sentence Representations Using SentEval

This folder contains scripts and test results from evaluating InferSent (Conneau et al. (2017)), CoVe (McCann et al. (2017)) and GloVe (Pennington et al. (2014)) sentence representations using SentEval for evaluation on various transfer tasks.

These instructions assume that you have followed the Install instructions in the README in the parent folder (root folder of this reposotory), and that you have either locally installed the requirements, or that you are currently running the Singularity container. For common problems, see the `Troubleshooting` section at the bottom of this page. Also see the GitHub issues sections of the InferSent, SentEval and CoVe repositories.

## Folder Structure

- In each folder, there is an `eval.py` script. This is the script used to run the SentEval evaluation on the given sentence embeddings.
- The folders also contain some test results/logs from running the evaluation scripts on various transfer tasks.
- `InferSent` folders are for InferSent (Conneau et al. (2017)) sentence embeddings. `CoVe` folders are for Context Vectors (CoVe) (McCann et al. (2017)) sentence embeddings. `GloVe` folders are for GloVe (Pennington et al. (2014)) sentence embeddings using GloVe word embeddings for each word. `GloVe+CoVe` contains sentence embeddings that are the concatenation of GloVe and CoVe, in the form `[GloVe(w); CoVe(w)]` for each word `w`. `Random` contains randomised fixed-size embeddings.
- Folders with `_full` in their name use the sentence embedding in its full form (i.e. every word embedding in the sentence embedding is kept), with some padding to reach a maximum sentence length. Folers with `_max` use max pooling on the sentence representation to give a fixed size embedding. Folers with `_mean` use mean pooling on the sentence representation to give a fixed size embedding.
- Folders with `_lower` in their name lowercase the data in each transfer task before using it. 
- The `qsub_helper` scripts are used for submitting HPC job submissions using `qsub`.

## Run

See `python eval.py -h` before running the scripts properly, for details on what the parameters are, as you will have to set all of the paths and flags that you need. Also open the scripts in an editor if you want to see an example of what the parameters are set to by default.

To run an evaluation script, `cd` into the folder you want and then:
```
python eval.py
```

## Troubleshooting / Common Problems

\* If your CUDA and cuDNN versions do not match, you might have to create your own Singularity file, changing the `from:` line and also the line that installs PyTorch. You can then host it in a similar way, or build it in another way, then pull/run it.

\**  Keep an eye on the InferSent/SentEval download scripts because curl sometimes gives an SSL error on UCL's cluster machines. If you can't update curl on your machine, you could either use the Singularity image (which has a working version of curl) or download the file(s) on a different machine and use `scp` to send them across.

\*** You may have problems when running things later, where Singularity looks at installations in your `/home` folder and these conflict with things that Singularity has installed within the container. You might have to remove some `/home` installs to get it to work.

\**** Newer PyTorch versions have a few differences in how you call certain functions - if you have some weird PyTorch errors, this is likely the reason. This code has been tested with PyTorch version 0.2.0_3. At this time of writing, the examples in the InferSent, SentEval and CoVe repositories are all out of date and need minor tweaks to get them to work with the latest PyTorch version. See their GitHub issues sections for more details.

\***** You might need to put `CUDA_VISIBLE_DEVICES=<device-number>` before the command to run the script. For example, `CUDA_VISIBLE_DEVICES=0 python eval.py` makes sure the script uses GPU 0.

## References

- [Conneau, Alexis, Kiela, Douwe, Schwenk, Holger, Barrault, Loïc, and Bordes, Antoine. Supervised learning of universal sentence representations from natural language inference data. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 670–680. Association for Computational Linguistics, 2017.](https://arxiv.org/pdf/1705.02364.pdf)
- [McCann, Bryan, Bradbury, James, Xiong, Caiming, and Socher, Richard. Learned in translation: Contextualized word vectors. In Advances in Neural Information Processing Systems 30, pp. 6297–6308. Curran Associates, Inc., 2017.](https://arxiv.org/pdf/1708.00107.pdf)
- [Pennington, Jeffrey, Socher, Richard, and Manning, Christopher D. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pp. 1532–1543, 2014.](https://nlp.stanford.edu/pubs/glove.pdf)
- Martin Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mane, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viegas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.
- Pytorch. [online]. Available at: https://github.com/pytorch/pytorch.
- Chollet, Francois et al. Keras. [online]. Available at: https://github.com/keras-team/keras, 2015.
- Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In Conference on Empirical Methods in Natural Language Processing (EMNLP 2013).
