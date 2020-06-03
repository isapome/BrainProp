# Training deep networks with a biologically plausible learning rule

_Implementation of BrainProp, a biologically plausible learning rule that can train deep neural networks on image-classification tasks (MNIST, CIFAR10, CIFAR100, Tiny ImageNet)._ 


## BrainProp: How the brain can implement reward-based error backpropagation

This repository is the official implementation of "BrainProp: How the brain can implement reward-based error backpropagation".
<!--- (https://arxiv.org/abs/{...}) --->
In the paper we show that by training only one output unit at a time we obtain a biologically plausible learning rule able to train deep neural networks on state-of-the-art machine learning classification tasks. The architectures we used range from 3 to 8 hidden layers. For the more shallow 


> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials


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
* Tiny ImageNet needs to be extracted by running: 
```tinyimagenet
python tinyimagenet.py
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;in the directory where the file "tiny-imagenet-200.zip" is located.

## Training

To train the model(s) in the paper, run this command:

```train
python main.py <input_file.py>
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

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.


## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.


## Results

Our model achieves the following performance on :


### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
