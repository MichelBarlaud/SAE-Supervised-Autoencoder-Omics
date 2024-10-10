# Supervised Autoencoder

This is the code from : *Accurate Diagnosis with a confidence score using the latent space of a new Supervised Autoencoder for clinical metabolomic studies.*

In this repository, you will find the code to replicate the statistical study described in the paper.
  
When using this code, please cite:

> Michel Barlaud and Frederic Guyard.
Learning a sparse generative non-parametric supervised autoencoder.
Proceedings of the International Conference on Acoustics, Speech and Signal
Processing, Toronto, Canada, June 2021.

and 

Michel Barlaud, Guillaume Perez, and Jean-Paul Marmorat.
Linear time bi-level l1,infini projection ; application to feature selection and
sparsification of auto-encoders neural networks.
http://arxiv.org/abs/2407.16293, 2024



## Table of Contents
***
1. [Repository Contents](repository-contents)
2. [Installation](#installation)
3. [How to use](#how-to-use)
  
### **Repository Contents**
|File/Folder | Description |
|:---|:---:|
|`script_autoencoder.py`|Main script to train and evaluate the SAE|

|`scripts to illustrate the paper _Linear time bi-level l1,infini projection |

|`datas`|Contains the  databases used in the paper|

|`functions`|Contains dedicated functions for the three main scripts|
    
### **Installation** 
---

To run this code, you will need :
- A version of python, 3.8 or newer. If you are new to using python, we recommend downloading anaconda ([here](https://www.anaconda.com/products/individual)) and using Spyder (available by default from the anaconda navigator) to run the code.
- [Pytorch](https://pytorch.org/get-started/locally/).
- The following packages, all of which except captum and shap are **usually included in the anaconda distribution** : [numpy](https://numpy.org/install/), [matplotlib](https://matplotlib.org/stable/users/installing/index.html), [scikit-learn](https://scikit-learn.org/stable/install.html), [pandas](https://pandas.pydata.org/getting_started.html), [shap](https://pypi.org/project/shap/), [captum](https://captum.ai/#quickstart). To install any package, you can use anaconda navigator's built-in environment manager.

See `requirements.txt` for the exact versions on which this code was developed.

### **How to use**

Everything is ready, you can just run the script you want using, for example, the run code button of your Spyder IDE. Alternatively, you can run the command `python [script_name].py` in the Anaconda Prompt from the root of this folder (i.e. where you downloaded and unzipped this repository).

Each script will produce results (statistical metrics, top features...) in a results folder.

You can change the database used, and other parameters, near the start of each script.
