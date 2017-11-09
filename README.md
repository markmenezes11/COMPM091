# Initial Experiment: Evaluating CoVe using SentEval
This contains the code required to evaluate CoVe using SentEval.

InferSent-master, SentEval-master and cove-master were cloned from these repositories:
- https://github.com/facebookresearch/InferSent
- https://github.com/facebookresearch/SentEval
- https://github.com/salesforce/cove
        
They originate from these papers:
- http://www.aclweb.org/anthology/D/D17/D17-1071.pdf
- https://arxiv.org/pdf/1708.00107.pdf

I have made a few minor changes to get around various errors caused by outdated code (mainly because the required arguments to a few torch functions have been updated since).

Install all of the necessary requirements listed on the READMEs of those three repositories, and run the necessary bash scripts to download the required data. Keep an eye on these scripts because curl sometimes gives an SSL error on UCL's cluster machines. As a workaround I just downloaded it on my local PC and scp'd it to the clusters, which worked. 

You can run the script that evaluates CoVe using SentEval by doing the following:
```
cd SentEval-master/examples
python cove.py
```

See the READMEs in the relevant folders to find out how to run the rest of the code such as the pre-made examples from Conneau et al. and McCann et al.