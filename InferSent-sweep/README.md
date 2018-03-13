# Parameter Sweeping InferSent

This folder contains code for performing a parameter sweep on InferSent, including training the InferSent model based on many different parameters, and evaluating the model using SentEval. The sweep script can either use qsub job submissions for running jobs in parallel or it can be run locally.

These instructions assume that you have followed the Install instructions in the README in the parent folder (root folder of this reposotory), and that you have either locally installed the requirements, or that you are currently running the Singularity container. For common problems, see the `Troubleshooting` section at the bottom of this page. Also see the GitHub issues sections of the InferSent, SentEval and CoVe repositories.

## File Structure

# TODO

## Run

*Note: A lot of the following scripts may use paths that do not exist on your system, so you will have to change these paths or set them as parameters when running them.*

# TODO

######################################################################################################################

### Evaluating Sentence Representations Using SentEval

These scripts, and text files containing their test results, can be found in the `SentEval-evals` folder. They require a GPU to run.

The scripts are organised into folders depending on what sentence representations they evaluate:
- The `SentEval-evals/CoVe` folder contains a script that uses SentEval to evaluate sentence representations of the form `[CoVe(w)]`, for each word `w` in the sentence (McCann et al., 2017).
- The `SentEval-evals/CoVe_lower` folder contains a script that uses SentEval to evaluate sentence representations of the form `[CoVe(w)]`, for each word `w` in the lower-cased sentence (McCann et al., 2017).
- The `SentEval-evals/GloVe` folder contains a script that uses SentEval to evaluate sentence representations of the form `[GloVe(w)]`, for each word `w` in the sentence (Pennington et al., 2014).
- The `SentEval-evals/GloVe_lower` folder contains a script that uses SentEval to evaluate sentence representations of the form `[GloVe(w)]`, for each word `w` in the lower-cased sentence (Pennington et al., 2014).
- The `SentEval-evals/GloVe+CoVe_lower` folder contains a script that uses SentEval to evaluate sentence representations of the form `[GloVe(w)+CoVe(w)]`, for each word `w` in the sentence (McCann et al., 2017 and Pennington et al., 2014).
- The `SentEval-evals/GloVe+CoVe_lower` folder contains a script that uses SentEval to evaluate sentence representations of the form `[GloVe(w)+CoVe(w)]`, for each word `w` in the lower-cased sentence (McCann et al., 2017 and Pennington et al., 2014).
- The `SentEval-evals/InferSent` folder contains a script that uses SentEval to evaluate InferSent sentence representations (Conneau et al., 2017).

To run one of the scripts, `cd` into the folder you want and then:
```
python eval.py
```

You can provide a specific transfer task with the `--transfertask` argument if you do not want SentEval to run all of them. Supported transfer tasks: STS12, STS13, STS14, STS15, STS16, MR, CR, MPQA, SUBJ, SST2, SST5, TREC, MRPC, SNLI, SICKEntailment, SICKRelatedness, STSBenchmark, ImageCaptionRetrieval. For example:
```
python eval.py --transfertask SICKEntailment
```

**You will very likely have to set the paths and GPU ID correctly, which can also be provided as arguments. See `python eval.py -h` for more details.**

There are also some test results in the above folders from running the scripts.

### Parameter Sweeping InferSent

The `InferSent-sweep` folder contains a program that runs a grid-search parameter sweep on InferSent sentence representations, using SentEval to evaluate them. The main script in that folder is `sweep.py`.

To tell it what parameters to sweep, you must change the arrays inside the script (in the `Parameters to sweep` section). A GPU is required to run this script.
```
python sweep.py
```

**You will very likely have to set the paths and GPU ID correctly, which can also be provided as arguments. See `python eval.py -h` for more details.**

You can also set the mode using the `--mode` argument, to specify whether you want it to run on your local PC or on the HPC cluster using qsub for job submissions. Again, you can find more details using `python sweep.py -h`. There is lots and lots of info in there...

### Training InferSent on SNLI and MultiNLI Datasets

You can also run the training on a single set of parameters using `train.py` in the `InferSent-sweep` folder, which will output a single model depending on the arguments you give it. See `python train.py -h` for more details.

For MultiNLI, you'll have to make a new folder with the MultiNLI datasets you downloaded with InferSent, and rename the files to match the same names that are used in the SNLI folder. In general, `matched` should become `dev` sets, and `mismatched` should become `test` sets. Then, you will need to make sure all the paths are correct in the arguments you provide as arguments to `train.py`.

### Replication of the CoVe Biattentive Classification Network (BCN)

The `CoVe-BCN` folder contains an attempt at replicating the Biattentive Classification Network (BCN) explained in Section 5 of the CoVe research paper (McCann et al., 2017). The description of the BCN missed out some important details, so this was a best attempt at replicating it, making various assumptions along the way.

You will need to provide it with a dataset to train on (see the `DATASET` section in the code), consisting of 2 input sentences and an output class (if there is only one input sentence, then simply duplicate it). You will also need to make sure the paths to the CoVe model and GloVe embeddings are correct, which are passed using the `--covepath` and `--glovepath` arguments (as well as other arguments). See `python cove_bcn.py -h` for more details.

You can download GloVe embeddings with:
```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

Run it with:
```
python cove_bcn.py
```

 See `python cove_bcn.py -h` for more details.

######################################################################################################################

## Troubleshooting / Common Problems

\* If your CUDA and cuDNN versions do not match, you might have to create your own Singularity file, changing the `from:` line and also the line that installs PyTorch. You can then host it in a similar way, or build it in another way, then pull/run it.

\**  Keep an eye on the InferSent/SentEval download scripts because curl sometimes gives an SSL error on UCL's cluster machines. If you can't update curl on your machine, you could either use the Singularity image (which has a working version of curl) or download the file(s) on a different machine and use `scp` to send them across.

\*** You may have problems when running things later, where Singularity looks at installations in your `/home` folder and these conflict with things that Singularity has installed within the container. You might have to remove some `/home` installs to get it to work.

\**** Newer PyTorch versions have a few differences in how you call certain functions - if you have some weird PyTorch errors, this is likely the reason. This code has been tested with PyTorch version 0.2.0_3. At this time of writing, the examples in the InferSent, SentEval and CoVe repositories are all out of date and need minor tweaks to get them to work with the latest PyTorch version. See their GitHub issues sections for more details.

## References

- [Conneau, Alexis, Kiela, Douwe, Schwenk, Holger, Barrault, Loïc, and Bordes, Antoine. Supervised learning of universal sentence representations from natural language inference data. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 670–680. Association for Computational Linguistics, 2017.](https://arxiv.org/pdf/1705.02364.pdf)
- [McCann, Bryan, Bradbury, James, Xiong, Caiming, and Socher, Richard. Learned in translation: Contextualized word vectors. In Advances in Neural Information Processing Systems 30, pp. 6297–6308. Curran Associates, Inc., 2017.](https://arxiv.org/pdf/1708.00107.pdf)
- [Pennington, Jeffrey, Socher, Richard, and Manning, Christopher D. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pp. 1532–1543, 2014.](https://nlp.stanford.edu/pubs/glove.pdf)
- Martin Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mane, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viegas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.
- Pytorch. [online]. Available at: https://github.com/pytorch/pytorch.
- Chollet, Francois et al. Keras. [online]. Available at: https://github.com/keras-team/keras, 2015.
- Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In Conference on Empirical Methods in Natural Language Processing (EMNLP 2013).
