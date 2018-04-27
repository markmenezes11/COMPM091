# On the Importance of the Choice of Downstream Models for Evaluating Sentence Representations

This repository contains the code created/used for my MEng Computer Science Undergraduate Final Year Individual Projct (COMPM091) at University College London.

My project primarily involved looking at sentence representations, specifically InferSent (Conneau et al., 2017) and CoVe (McCann et al., 2017) and using SentEval (Conneau et al., 2017) and the Biattentive Classification Network (McCann et al., 2017) to evaluate them.

The code for InferSent, SentEval and CoVe were cloned from the following repositories:
- InferSent: https://github.com/facebookresearch/InferSent
- SentEval: https://github.com/facebookresearch/SentEval
- CoVe: https://github.com/salesforce/cove and https://github.com/rgsachin/CoVe

In the `libs.zip` file, you will find snapshots of these repositories - the versions of them that were used in my project, at this time of writing. If you have problems with any newer versions of these libraries, it is therefore recommended that you use the versions contained in `libs.zip`. 

*Note: A CUDA-enabled GPU is required, and at least 16GB RAM is recommended to run most of the scripts. A small minority of scripts will need more than this, hence why they were run on UCL's HPC cluster.*




## Install

There are two ways to install. You can use the provided Singularity container, which already contains all of the required software, or you can install the requirements locally. It is highly recommended that you use the Singularity container.

For the Singularity method, use steps 1, 2a and 3. For local install, use steps 1, 2b and 3.

For common problems, see the `Troubleshooting` section at the bottom of this page.

### Step 1) Clone InferSent and SentEval

You can either clone them from here:
- InferSent: https://github.com/facebookresearch/InferSent
- SentEval: https://github.com/facebookresearch/SentEval

Or, in case any changes are made to these repositories that stop this code from working, snapshots of these repositories used at this time of writing have been provided in the `libs.zip` file, but obviously they may not be the latest version. A modified version of SentEval which uses far less memory can be found in the `SentEval-modified` folder, but only use this if you really have to. Read the README in that folder first. It works identically to the version found in `libs.zip` but contains various performance tweaks. So, instead of cloning directly from the InferSent and SentEval repositories, you can use the versions provided in `libs.zip` or the version of SentEval provided in the `SentEval-modified` folder.

### Step 2a) Singularity Container (Recommended)

Pull and run the latest Singularity container for this repository from Singularity Hub. 
```
singularity run --nv shub://markmenezes11/COMPM091
```
`--nv` runs it in nvidia mode. See `singularity run -h` for more options. You can also use the same command above to pull any later versions of the container in future, making sure to delete the old `.simg` file first.

The container uses nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04. Make sure your host machine's CUDA and cuDNN versions match this*. If they don't, you might need to tweak the `Singularity` file and re-build the image locally.

### Step 2b) Local Install

The `Singularity` file contains a list of commands in the `%post` section to install the required libraries. You will need to run each of these commands. Python 2.7 was used for the majority of this work, and so you should only need to do it for Python 2.7 (i.e. ignore the `pip3` commands) and use Python 2.7 when running the scripts later. You will also need CUDA and cuDNN to make use of your GPU. Depending on where/how you are running this, you might need to run the commands as `sudo` or `--user`. You will also need to find/use a different URL for PyTorch and TensorFlow if you are not using Linux and CUDA 8. 

### Step 3) Download Datasets

Run all of the required scripts** and curl commands** to download the required datasets. These scripts can be found in the repositories you just cloned (see Step 1), and instructions can be found in their READMEs.

## Note for Users of UCL's HPC Cluster

If using UCL's HPC cluster, it is recommended to follow the Singularity instructions above. Additionally, assuming it hasn't been moved, some outputs of running some of my code, as well as the Singularity image and the InferSent and SentEval libraries with all of the requirements already downloaded, should be in `/cluster/project2/ishi_storage_1/mmenezes`. If so, you can save time by just using the following Singularity command which will bind my folder to `/mnt/mmenezes` in the Singularity image for you, with all of the requirements pre-installed and pre-downloaded: 
```
singularity run --nv --bind /cluster/project2/ishi_storage_1:/mnt /cluster/project2/ishi_storage_1/mmenezes/markmenezes11-COMPM091-master.simg
```
When following the "Run" instructions later, you can then use the above command instead of `singularity run --nv shub://markmenezes11/COMPM091`. 

## Run

For "Run" instructions, see the relevant README(s) in the subfolder(s) within this repository. They all require all of the above installation instructions to be competed first. The folders containing the runnable scripts are as follows:
- The `CoVe-BCN` folder contains my replication of the Biattentive Classification Network model by McCann et al. (2017). It also includes a script for running a parameter sweep on this model, and evaluaitng both InferSent and CoVe on different transfer tasks.
- The `CoVe-ported` folder contains a script (and some test scripts) for porting the Python 3 version of the Keras port of CoVe into a Python 2 compatible version. It is unlikely that you will need to run any of these scripts.
- The `InferSent-sweep` folder contains code for performing a parameter sweep on InferSent, including training the InferSent model based on many different parameters, and evaluating the model using SentEval. The sweep script can either use qsub job submissions for running jobs in parallel or it can be run locally.
- The `SentEval-evals` folder contains scripts and test results from evaluating InferSent (Conneau et al. (2017)), CoVe (McCann et al. (2017)) and GloVe (Pennington et al. (2014)) sentence representations using SentEval for evaluation on various transfer tasks.
- The `Utils` folder contains helper scripta, for example for displaying RAM usage, for giving summaries based on lots of results files, and for manipulating datasets.

Also have a look at the `qsub_helper` scripts in these folders if you want to use `qsub` to submit HPC jobs. They can mostly be run using `qsub <qsub-params> <qsub_helper-script> <command>`. The <command> bit can use singularity by running the `singularity exec` command instead of `singularity run`. For example:

```
qsub -cwd -o $PWD/se_log_STS12.txt -e $PWD/se_error_STS12.txt ../qsub_helper.sh singularity exec --nv --bind /cluster/project2/ishi_storage_1:/mnt /cluster/project2/ishi_storage_1/mmenezes/markmenezes11-COMPM091-master.simg python eval.py --transfertask STS12
```




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
- Xin Li, Dan Roth, Learning Question Classifiers. COLING'02, Aug., 2002.
- E. M. Voorhees and D. M. Tice. The TREC-8 question answering track evaluation. In TREC, volume 1999, page 82, 1999.

Libraries and algorithms are referenced in the files they are used.