# Similarity Computation for Graphs

* Doan & Machanda et al. Interpretable Graph Similarity Computation via Differentiable Optimal Alignment of Node Embeddings (GOTSim). SIGIR 2021. [[Paper](https://people.cs.vt.edu/~reddy/papers/SIGIR21.pdf)] [[Slides](https://github.com/khoadoan/GraphOTSim/blob/main/resources/SIGIR21-fp0937-slides.pdf)] [[Video](https://www.youtube.com/watch?v=IWxxsuFPsgs)]


## Setup the environment

This repository requires python 3.7+ and conda environment. Please refer to `requirements.txt` file for the dependenencies.

## Training and Evaluation GOTSim

GOTSim's training and evaluation processes are encapsulated inside the script `train_gotsim.py'. To train and evaluate using the provided 5-fold evaluation, simply run:

```
export PYTHONPATH=external/:python/:$PYTHONPATH
python python train_gotsim.py --basedir exp/final/PTC_ged/GOTSim/ --dataset data/PTC_ged/
```

