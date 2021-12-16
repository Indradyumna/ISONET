# ISONET

Interpretable Neural Subgraph Matching for Graph Retrieval

This directory contains code necessary for running ISONET experiments.

#Requirements

Recent versions of Pytorch,Pytorch Geometric, numpy, scipy, sklearn, networkx and matplotlib are required.
You can install all the required packages using  the following command:

	$ conda create --name <env> --file requirements.txt

#Datasets
Please download the Dataset files from https://rebrand.ly/subgraph-isonet and replace the current dummy Dataset folder.
This contains the original datasets, the dataset splits and other intermediate data dumps for reproducing tables and plots.  


#Run Eperiments

The command lines to used for training models are listed commands.txt
Command lines specify the exact hyperparameter settings used to train the models. 

#Reproduce plots and figures

The notebooks folder contains .ipynb files which reproduce all the tables and figures presented in the paper. 

Notes:
 - GPU usage is required
 - source code files are all in subgraph folder.
