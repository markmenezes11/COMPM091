# Training and Evaluating Sentence Representations 

InferSent-master, SentEval-master and cove-master were cloned from these repositories:
- https://github.com/facebookresearch/InferSent
- https://github.com/facebookresearch/SentEval
- https://github.com/salesforce/cove
        
They originate from these papers:
- http://www.aclweb.org/anthology/D/D17/D17-1071.pdf
- https://arxiv.org/pdf/1708.00107.pdf

I have made a few minor changes to get around various errors caused by outdated cod, mainly because the required arguments to a few torch functions have been updated since. I have also created a Singularity file that can be used to run a Singularity container (like Docker) that already has all of the required software installed.



## Install

There are two ways to install. You can use the provided Singularity container, or install the requirements locally.

For common problems, see the `Troubleshooting` section at the bottom of this page.

### Singularity

1) Run the bash scripts* and curl downloads* mentioned in the READMEs the InferSent, SentEval and CoVe repositories (see relevant folders) to download the required datasets. You don't need Python or any of its libraries for this, just simple things like curl, unzip and cabextract.
2) Pull and run the latest Singularity container for this repository from Singularity Hub. "--nv" runs it in nvidia mode. The container uses nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04. Make sure your host machine's CUDA and cuDNN versions match this**:
```
singularity run --nv shub://markmenezes11/COMPM091
```
2) a) You can also Use the above command to pull any later versions of the container in future, making sure to remove the .simg file first.

### Local Install

1) Install all of the necessary requirements listed on the READMEs the InferSent, SentEval and CoVe repositories (see relevant folders).
2) Run the bash scripts* and curl downloads* mentioned in the READMEs the InferSent, SentEval and CoVe repositories (see relevant folders) to download the required datasets.



## Run

See the READMEs in the relevant folders to find out how to run the rest of the code such as the pre-made examples from Conneau et al. and McCann et al.

For common problems, see the `Troubleshooting` section at the bottom of this page.

Aside from that, I have created various other scripts for other experiments, highlighted below.

The instructions assume that you either have locally installed the requirements above, or that you are currently running the Singularity container.

### Evaluating CoVe Using SentEval

You can run the script that evaluates CoVe using SentEval by doing the following:
```
cd SentEval-master/examples
python cove.py
```
There also many other python scripts in SentEval-master/examples whose filename begines with `cove`. They all test CoVe under different conditions. For example, `lower` in `cove_lower.py` means the tests are run with everything lower cased; `fst300` means only the first 300 CoVe vectors are used; `lst300` means only the last 300 CoVe vectors are used.

The outputs from running some of the scripts in SentEval-master/examples can be found in various text files in that folder.

### Training InferSent on SNLI and MultiNLI Datasets

For MultiNLI, you'll have to make a new folder with the MultiNLI datasets you downloaded with InferSent, and rename the files to match the same names that are used in the SNLI folder. In general, `matched` should become `dev` sets, and `mismatched` should become `test` sets.  

(TODO: More info and scripts to be added here)

```
cd InferSent-master
python train_nli.py
```



## Troubleshooting

\*  Keep an eye on these scripts because curl sometimes gives an SSL error on UCL's cluster machines. As a workaround I just downloaded it on my local PC and scp'd it to the clusters, but there's probably a better way.

\** If your CUDA and cuDNN versions do not match, you might have to create your own Singularity file, changing the `from:` line and also the line that installs PyTorch. You can then host it in a similar way, or build it in another way, then pull/run it.

\*** You may have problems when running things later, where Singularity looks at installations in your /home folder and these conflict with things that Singularity has installed within the container. You might have to remove some /home installs to get it to work.