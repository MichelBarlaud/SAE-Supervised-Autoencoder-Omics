# Linear time bi-level l1,infini projection 


In this repository, you will find the code to replicate the statistical study described in the paper.
  

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
|`script_test_timer.py`|Main script to train and evaluate the Linear time bi-level l1,infini projection |

|`setup.py`|Script to install the projection on a PC (i9 processor) or a Macbook (M3 processor)|

|`test.py`|Test if the code is well installed|

|`projections.cpp`|Contains the CPP implementation|
    
### **Installation** 
---

To run this code, you will need :
- A version of python, 3.8 or newer. If you are new to using python, we recommend downloading anaconda ([here](https://www.anaconda.com/products/individual)) and using Spyder (available by default from the anaconda navigator) to run the code.
- [Pytorch](https://pytorch.org/get-started/locally/).

- To install this package, please run the following command:  python setup.py install --user


### **How to use**

Everything is ready, you can just run the script you want using, for example, the run code button of your Spyder IDE. Alternatively, you can run the command `python [script_name].py` in the Anaconda Prompt from the root of this folder (i.e. where you downloaded and unzipped this repository).

