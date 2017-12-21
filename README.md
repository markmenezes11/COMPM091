# Training and Evaluating Sentence Representations 

InferSent-master, SentEval-master and cove-master were cloned from these repositories:
- InferSent: https://github.com/facebookresearch/InferSent
- SentEval: https://github.com/facebookresearch/SentEval
- CoVe: https://github.com/salesforce/cove
        
They originate from these papers:
- Conneau et al. (Jul 2017). "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data". [online]. Available at: https://arxiv.org/pdf/1705.02364.pdf
- McCann et al. (Aug 2017). "Learned in Translation: Contextualized Word Vectors". [online]. Available at: https://arxiv.org/pdf/1708.00107.pdf

## Install

There are two ways to install. You can use the provided Singularity container, which already contains all of the required software, or you can install the requirements locally.

For common problems, see the `Troubleshooting` section at the bottom of this page.

### Singularity (Recommended)

1) Clone the following repositories somewhere accessible (e.g. parent folder outside of where you cloned this repository). Note: Snapshots of these repositories have been provided as zip files in the `libs` folder, but they may not be the latest version:
- InferSent: https://github.com/facebookresearch/InferSent
- SentEval: https://github.com/facebookresearch/SentEval
- CoVe: https://github.com/salesforce/cove

2) Pull and run the latest Singularity container for this repository from Singularity Hub. `--nv` runs it in nvidia mode. See `singularity run -h` for more options. The container uses nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04. Make sure your host machine's CUDA and cuDNN versions match this*: The `Singularity` file used to create this container can be found in the root folder of this repository. You can also use the same command to pull any later versions of the container in future, making sure to remove the .simg file first.
```
singularity run --nv shub://markmenezes11/COMPM091
```

3) With the Singularity container running, run all of the required scripts to download the data. These scripts can be found in the three repositories you just cloned (see step 1), and instructions can be found in their READMEs.

### Local Install

1) Clone the following repositories somewhere accessible (e.g. parent folder outside of where you cloned this repository). Note: Snapshots of these repositories have been provided as zip files in the `libs` folder, but they may not be the latest version:
- InferSent: https://github.com/facebookresearch/InferSent
- SentEval: https://github.com/facebookresearch/SentEval
- CoVe: https://github.com/salesforce/cove

2) Install all of the necessary requirements (including CUDA, cuDNN, etc.) listed in the READMEs of the three repositories you just cloned (see step 1). Alternatively, the `Singularity` file used to create the Singularity container can be found in the root folder of this repository, and contains a nice list of commands to install the requirements.

3) Run all of the required scripts** and curl commands** to download the required datasets. These scripts and curl commands can be found in the three repositories you just cloned (see step 1), and instructions can be found in their READMEs.




## Run

See the READMEs in the InferSent, SentEval and CoVe repositories to find out how to run the rest of the code such as the pre-made examples from Conneau et al. and McCann et al.****

For common problems, see the `Troubleshooting` section at the bottom of this page.

In this repository, you will find various scripts I have created to conduct on CoVe and InferSent, using SentEval, highlighted below.

The instructions assume that you either have locally installed the requirements above, or that you are currently running the Singularity container.

### Evaluating CoVe Using SentEval

These scripts, and text files containing their test results, can be found in the `cove/SentEval-evals` folder.

Note: You will probably need to change the COVE_PATH, SENTEVAL_PATH and SENTEVAL_DATA_PATH variables in these files to wherever you cloned the above repositories.  

To run the script that tests an unedited CoVe model:
```
cd cove/SentEval-evals
python cove.py
```

If a script has`lower` in its name (e.g. `cove_lower.py`), it means the tests are run with everything lower cased; `fst` means only the first 300 CoVe vectors are used; `lst` means only the last 300 CoVe vectors are used; `utf8` means all the words were first encoded to utf8.

### Evaluating InferSent Using SentEval

Similar to above, you can find a folder containing InferSent evaluations in `InferSent/SentEval-evals`.

### Training InferSent on SNLI and MultiNLI Datasets

For MultiNLI, you'll have to make a new folder with the MultiNLI datasets you downloaded with InferSent, and rename the files to match the same names that are used in the SNLI folder. In general, `matched` should become `dev` sets, and `mismatched` should become `test` sets.  

Then, specify the correct path as a parameter when calling the python script below:

```
cd InferSent
python train.py
```

See `python train.py -h` for more details.

### Parameter Sweeping InferSent

The InferSent parameter sweep script can be found in `InferSent/sweep.py`. To tell it what parameters to sweep, you must change the arrays inside the script (in the `Parameters to sweep` section). A GPU is required to run this script.

```
cd InferSent
python sweep.py
```

To change paths, GPU ID, etc., they can be given as arguments. See `python sweep.py -h` for more details.

`sweep_default.py` does a single sweep on default parameters. It uses the AllNLI dataset, which you will need to create by combining the SNLI and MultiNLI datasets: https://github.com/facebookresearch/InferSent/issues/24
 
There is also `sweep_qsub_gpu.sh` which is the script to use for job submission via qsub on a cluster machine (`qsub sweep_qsub_gpu.sh`). 




## Troubleshooting

\* If your CUDA and cuDNN versions do not match, you might have to create your own Singularity file, changing the `from:` line and also the line that installs PyTorch. You can then host it in a similar way, or build it in another way, then pull/run it.

\**  Keep an eye on these scripts because curl sometimes gives an SSL error on UCL's cluster machines. As a workaround I just downloaded it on my local PC and scp'd it to the clusters, but there's probably a better way.

\*** You may have problems when running things later, where Singularity looks at installations in your /home folder and these conflict with things that Singularity has installed within the container. You might have to remove some /home installs to get it to work.

\**** Newer PyTorch versions have a few differences in how you call certain functions - if you have some weird PyTorch errors, this is likely the reason. This code has been tested with PyTorch version 0.2.0_3. At this time of writing, the examples in the InferSent, SentEval and CoVe repositories are all out of date and need minor tweaks to get them to work with the latest PyTorch version. See their GitHub issues sections for more details. 