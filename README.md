# Training and Evaluating Sentence Representations 

This repository contains the code created/used for my MEng Computer Science Undergraduate Final Year Individual Projct (COMPM091).

My project primarily involved looking at sentence representations, specifically InferSent (Conneau et al., 2017) and CoVe (McCann et al., 2017) and using SentEval (Conneau et al., 2017) to evaluate them. They originate from these papers:
- Conneau, A et al. (Jul 2017). "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data". [online]. Available at: https://arxiv.org/pdf/1705.02364.pdf
- McCann, B et al. (Aug 2017). "Learned in Translation: Contextualized Word Vectors". [online]. Available at: https://arxiv.org/pdf/1708.00107.pdf

The code for InferSent, SentEval and CoVe were cloned from the following repositories:
- InferSent: https://github.com/facebookresearch/InferSent
- SentEval: https://github.com/facebookresearch/SentEval
- CoVe: https://github.com/salesforce/cove and https://github.com/rgsachin/CoVe

In the `libs.zip` file, you will find snapshots of these repositories - the versions of them that were used in my project, at this time of writing.




## Install

There are two ways to install. You can use the provided Singularity container, which already contains all of the required software, or you can install the requirements locally.

For common problems, see the `Troubleshooting` section at the bottom of this page.

#### Singularity (Recommended)

Step 1) Clone the following repositories to a location of your choice (*Note: In case any changes are made to these repositories that stop this code from working, snapshots of these repositories used at this time of writing have been provided in the `libs.zip` file, but obviously they may not be the latest version*):
- InferSent: https://github.com/facebookresearch/InferSent
- SentEval: https://github.com/facebookresearch/SentEval

Step 2) Pull and run the latest Singularity container for this repository from Singularity Hub. `--nv` runs it in nvidia mode. See `singularity run -h` for more options. The container uses nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04. Make sure your host machine's CUDA and cuDNN versions match this*. You can also use the same command (below) to pull any later versions of the container in future, making sure to remove the .simg file first:
```
singularity run --nv shub://markmenezes11/COMPM091
```
(*Note: The `Singularity` file used to create this container can be found in the root folder of this repository, in case you want to re-build the container locally.*)

Step 3) With the Singularity container running, run all of the required scripts** and curl commands** to download the data. These scripts can be found in the repositories you just cloned (see Step 1), and instructions can be found in their READMEs.

#### Local Install

Step 1) Clone the following repositories to a location of your choice (*Note: In case any changes are made to these repositories that stop this code from working, snapshots of these repositories used at this time of writing have been provided in the `libs.zip` file, but obviously they may not be the latest version*):
- InferSent: https://github.com/facebookresearch/InferSent
- SentEval: https://github.com/facebookresearch/SentEval

Step 2) Install all of the necessary requirements (including CUDA, cuDNN, Python 2, Python libraries, etc.) listed in the READMEs of the three repositories you just cloned (see Step 1). Alternatively, see the `Singularity` file for a nice list of commands to install the requirements.

Step 3) Run all of the required scripts** and curl commands** to download the required datasets. These scripts can be found in the repositories you just cloned (see Step 1), and instructions can be found in their READMEs.

#### UCL HPC Cluster

If using UCL's HPC cluster, it is recommended to follow the Singularity instructions above. Additionally, assuming it hasn't been moved, some outputs of running some of my code, as well as the Singularity image and the InferSent and SentEval libraries with all of the requirements already downloaded, should be in `/cluster/project2/ishi_storage_1/mmenezes`. If so, you can save time by just using the following Singularity command which will bind my folder to `/mnt/mmenezes` in the Singularity image for you, with all of the requirements pre-installed and pre-downloaded: 
```
singularity run --nv --bind /cluster/project2/ishi_storage_1:/mnt /cluster/project2/ishi_storage_1/mmenezes/markmenezes11-COMPM091-master.simg
```


## Run

The instructions assume that you have either locally installed the requirements, or that you are currently running the Singularity container. For common problems, see the `Troubleshooting` section at the bottom of this page. Also see the GitHub issues sections of the InferSent, SentEval and CoVe repositories.

(*Note: A lot of the following scripts may use paths that do not exist on your system, so you will have to change these paths or set them as parameters when running them.*)

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

Run it with:
```
python cove_bcn.py
```

You will need to provide it with a dataset to train on (see the `DATASET` section in the code), consisting of 2 input sentences and an output class (if there is only one input sentence, then simply duplicate it). You will also need to make sure the paths to the CoVe model and GloVe embeddings are correct. See `python cove_bcn.py -h` for more details.




## Troubleshooting

\* If your CUDA and cuDNN versions do not match, you might have to create your own Singularity file, changing the `from:` line and also the line that installs PyTorch. You can then host it in a similar way, or build it in another way, then pull/run it.

\**  Keep an eye on the InferSent/SentEval download scripts because curl sometimes gives an SSL error on UCL's cluster machines. If you can't update curl on your machine, you could either use the Singularity image (which has a working version of curl) or download the file(s) on a different machien and use `scp` to send them across.

\*** You may have problems when running things later, where Singularity looks at installations in your /home folder and these conflict with things that Singularity has installed within the container. You might have to remove some /home installs to get it to work.

\**** Newer PyTorch versions have a few differences in how you call certain functions - if you have some weird PyTorch errors, this is likely the reason. This code has been tested with PyTorch version 0.2.0_3. At this time of writing, the examples in the InferSent, SentEval and CoVe repositories are all out of date and need minor tweaks to get them to work with the latest PyTorch version. See their GitHub issues sections for more details.




## References

- Conneau, A  et al. (Jul 2017). "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data". [online]. Available at: https://arxiv.org/pdf/1705.02364.pdf
- McCann, B et al. (Aug 2017). "Learned in Translation: Contextualized Word Vectors". [online]. Available at: https://arxiv.org/pdf/1708.00107.pdf
- Pennington, J, Socher, R and Manning, C. (2014). "GloVe: Global Vectors for Word Representation". [online]. Available at: https://nlp.stanford.edu/pubs/glove.pdf
