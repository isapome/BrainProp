# Training deep networks with a biologically plausible algorithm

_Implementation of BrainProp, a biologically plausible learning rule that can train deep neural networks on image-classification tasks (MNIST, CIFAR10, CIFAR100, Tiny ImageNet)._ 


## BrainProp: How the brain can implement reward-based error backpropagation

This repository is the official implementation of "Attention-Gated Brain Propagation: How the brain can implement reward-based error backpropagation", [NeurIPS 2020 proceedings](https://proceedings.neurips.cc//paper/2020/file/1abb1e1ea5f481b589da52303b091cbb-Paper.pdf).

In the paper we show that by training only one output unit at a time we obtain a biologically plausible learning rule able to train deep neural networks on state-of-the-art machine learning classification tasks. The architectures used range from 3 to 8 hidden layers.


<!---  📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials --->


## Requirements

The current version of the code requires a recent (as of June 2020) version of tensorflow-gpu, CUDA and cuDNN and it was specifically tested on the following versions of the packages:

* Python 3.6.6
* pip 20.1.1
* CUDA 10.1.243
* cuDNN 7.6.5.32

To install the required libraries and modules (after having created a virtual environment with the versions of Python and pip indicated above):

```setup
pip install -r requirements.txt
```

#### Datasets
* MNIST, CIFAR10 and CIFAR100 are automatically available through keras. 
* Tiny ImageNet can be downloaded from the [official page of the challenge](https://tiny-imagenet.herokuapp.com) or extracted by running: 
```tinyimagenet
python tinyimagenet.py
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;in the directory where the file "tiny-imagenet-200.zip" is located.

## Training and Evaluation

To train the model(s) in the paper, run this command:

```
python main.py <dataset> <architecture> <algorithm>
```
 the training will stop when the validation accuracy has not increased for 45 epochs, otherwise until 500 epochs are reached.
 
The possible `<dataset>` - `<architecture>` combinations are:

* `MNIST` - {`dense`, `loccon`, `conv`}
* `CIFAR10` - {`loccon`, `conv`, `deep`}
* `CIFAR100` - {`loccon`, `conv`, `deep`}
* `TinyImageNet` - `deep`

  For the details of the architectures, please refer to the paper. 
 
For `<algorithm>`, set `BrainProp` for BrainProp or `EBP` for error-backpropagation.

Add the flag `-s` to save a plot of the accuracy, the trained weights (at the best validation accuracy) and the history file of the training. 

To load and evaluate a saved model:
<!--- If the parameter `save_weights` is set to `True`, an h5 file with the weights will be saved and its name will be added to the input file. The model can then be evaluated by doing: --->
 
 ```
 python main.py <dataset> <architecture> <algorithm> -l <weightfile.h5>
 ```
 
Three pre-trained models (on the deep network with BrainProp) on CIFAR10 (`CIFAR10_BrainProp_weights.h5`), CIFAR100 (`CIFAR100_BrainProp_weights.h5`) and Tiny ImageNet (`TIN_BrainProp_weights.h5`) are included.

All the hyperparameters (as specified in the paper) are automatically set depending on which architecture is chosen. 

<!---  📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. --->


<!---## Evaluation
To evaluate my model on ImageNet, run:
```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```
> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).--->


 <!--- ## Pre-trained Models --->

 <!--- Some pre-trained models are included. Specifically networks trained with BrainProp with the deep architecture on CIFAR10, CIFAR100 and Tiny ImageNet. --->

<!--- You can download pretrained models here:
- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 
> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models. --->


## Results

All the experiments ran on one node with a NVIDIA GeForce 1080Ti card.

Our algorithm achieved the following performances (averaged over 10 different seeds, the mean and standard deviation are indicated):
<!--- ### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet) --->

| BrainProp            |  Top 1 Accuracy [%] |  Epochs [#]  | Seconds/Epoch |
|  ------------------  |  ----------------   |  ----------- | ------------- |
| MNIST - `conv`        |     99.31(0.04)     |    63(18)    | 3  |
| CIFAR10 - `deep`       |     88.88(0.27)     |    105(4)    | 8  |
| CIFAR100 - `deep`      |     59.58(0.46)     |    218(22)   | 8  |
| Tiny ImageNet - `deep` |     47.50(1.30)     |    328(75)   | 47 |

For the `dense` and `conv` simulations the speed was 3s/epoch, while for `loccon` the speed ranged between 45- and 60s/epoch.

For the complete tables and figures, please refer to the paper. 

<!--- 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. --->


<!--- ## Contributing
> 📋Pick a licence and describe how to contribute to your code repository. --->
