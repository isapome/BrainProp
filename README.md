# Training deep networks with a biologically plausible algorithm

_Implementation of BrainProp, a biologically plausible learning rule that can train deep neural networks on image-classification tasks (MNIST, CIFAR10, CIFAR100, Tiny ImageNet)._ 


## BrainProp: How the brain can implement reward-based error backpropagation

This repository is the official implementation of "BrainProp: How the brain can implement reward-based error backpropagation".
<!--- (https://arxiv.org/abs/{...}) --->
In the paper we show that by training only one output unit at a time we obtain a biologically plausible learning rule able to train deep neural networks on state-of-the-art machine learning classification tasks. The architectures used range from 3 to 8 hidden layers.


<!---  📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials --->


## Requirements

The current version of the code requires a recent (as of June 2020) version of tensorflow-gpu, CUDA and cuDNN and it was specifically tested on the following versions of the packages:

* Python 3.6.6
* pip 20.1.1
* CUDA 10.1.243
* cuDNN 7.6.5.32

To install the required libraries and modules:

```setup
pip install -r Requirements.txt
```

#### Datasets
* MNIST, CIFAR10 and CIFAR100 are available through keras. 
* Tiny ImageNet can be downloaded from the [official page of the challenge](https://tiny-imagenet.herokuapp.com) or extracted by running: 
```tinyimagenet
python tinyimagenet.py
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;in the directory where the file "tiny-imagenet-200.zip" is located.

## Training and Evaluation

To train the model(s) in the paper, run this command:

```train
python main.py <input_file>
```
 the training will stop when the validation accuracy has not increased for 45 epochs, otherwise until 500 epochs are reached.
 
If the parameter `save_weights` is set to `True`, an h5 file with the weights will be saved and its name will be added to the input file. The model can then be evaluated by doing:
 
 ```eval
 python main.py <input_file> -l
 ```
 
The input files included allow to train models on:

* 3 fully connected layers:
  * MNIST: *inputs_MNIST.py*
* 2 locally connected layers and 1 fully connected layer:
  * MNIST: *inputs_loccon_MNIST.py*
  * CIFAR10: *inputs_loccon_C10.py*
  * CIFAR100: *inputs_loccon_C100.py*
* 7 convolutional layers and 1 fully connected layer:
  * CIFAR10: *inputs_C10.py*
  * CIFAR100: *inputs_C100.py*
  * Tiny ImageNet: *inputs_TIN.py*

All the hyperparameters (as specified in the paper) are included in the input files. 

<!---  📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. --->


<!---## Evaluation
To evaluate my model on ImageNet, run:
```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```
> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).--->


## Pre-trained Models

Some pre-trained models are included. Specifically networks trained with BrainProp with the deep architecture on CIFAR10, CIFAR100 and Tiny ImageNet.

<!--- You can download pretrained models here:
- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 
> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models. --->


## Results

Our model achieves the following performance (averaged over 10 different seeds, the mean and standard deviation are indicated):
<!--- ### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet) --->

| BrainProp            |  Top 1 Accuracy [%] |
|  ------------------  |  ----------------   |
| MNIST - dense        |     98.68(0.07)     |
| CIFAR10 - deep       |     88.88(0.27)     | 
| CIFAR100 - deep      |     59.58(0.46)     |
| Tiny ImageNet - deep |     47.50(1.30)     |

For the complete tables and figures, please refer to the paper. 

<!--- 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. --->


<!--- ## Contributing
> 📋Pick a licence and describe how to contribute to your code repository. --->
