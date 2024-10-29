# ISONET

[Interpretable Neural Subgraph Matching for Graph Retrieval (AAAI 22)](https://cdn.aaai.org/ojs/20784/20784-13-24797-1-2-20220628.pdf)

This directory contains code necessary for running ISONET experiments.

#Requirements

Recent versions of Pytorch,Pytorch Geometric, numpy, scipy, sklearn, networkx and matplotlib are required.
You can install all the required packages using  the following command:

	$ conda create --name <env> --file requirements.txt

#Datasets
Please download the Dataset files from https://rebrand.ly/subgraph-isonet and replace the current dummy Dataset folder.
This contains the original datasets, the dataset splits and other intermediate data dumps for reproducing tables and plots.  


#Run Experiments

The command lines used for training models are listed commands.txt
Command lines specify the exact hyperparameter settings used to train the models. 

#Reproduce plots and figures

The notebooks folder contains .ipynb files which reproduce all the tables and figures presented in the paper. 

Notes:
 - GPU usage is required
 - source code files are all in subgraph folder.

Reference
---------

If you find the code useful, please cite our paper:

	@inproceedings{roy2022interpretable,
	  title={Interpretable neural subgraph matching for graph retrieval},
	  author={Roy, Indradyumna and Velugoti, Venkata Sai Baba Reddy and Chakrabarti, Soumen and De, Abir},
	  booktitle={Proceedings of the AAAI conference on artificial intelligence},
	  volume={36},
	  number={7},
	  pages={8115--8123},
	  year={2022}
	}

Indradyumna Roy, Indian Institute of Technology - Bombay  
indraroy15@cse.iitb.ac.in
