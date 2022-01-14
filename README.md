# MLP-Mixer
### Unoffical Implementation of MLP-Mixer, easy to use with terminal. Train and test easly.

https://arxiv.org/abs/2105.01601

![N|Solid](https://warehouse-camo.ingress.cmh1.psfhosted.org/18dc7ec6217146811d48b4b4c9aa51721f9df623/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f6d61782f323430302f312a4471727a6e454b7a525f78422d434568704f617633412e706e67)

MLP-Mixer is an architecture based exclusively on multi-layer perceptrons (MLPs).

According to paper, Model offers:
- Better accuracy than CNNs and Transformers
- Lower time complexity than CNNs and Transformers
- Lower parameters than CNNs and Transformers
## Quick Start
Clone the repo and install the requirements.txt in a Python>=3.8 environment.

```
git clone https://github.com/Oguzhanercan/MLP-Mixer
cd MLP-Mixer
pip install -r requirements.txt
```
### Dataset
There are 2 options for dataset. You can use pre-defined datasets listed below
- CIFAR10
- Mnist
- Fashion Mnist

or you can use your own dataset. Organize your folder structure as:
```
      data---
            |
            --0
               |
                --img0.png
                .
                .
                --img9999.png
            |
            -- 1
                |
                --img0.png
                .
                .
                --img9999.png
            .
            .

```
0 and 1 represents folders that contains images belongs only one particular class. There is no limit for classes or images.

### Train
Open a terminal at the same directory of clone. Then run the code below.
```
python main.py --mode train --dataset CIFAR10 --save True --device cuda --epochs 20 --valid_per 0.2 
```
You can customize the model hyperparameters, all arguments listed below
"Arguments:
- dataset : Categorical Option --- Choose the dataset, Options: CIFAR10, Mnist, Fashion Mnist, Custom
- train_path : Path --- Enter the train path, if you are using custom dataset mode
- test_path : Path --- Enter the test path, if you are using custom dataset mode
- batch_size : integer number ---
- im_size : integer number --- Enter the biggest dimension of image, Example : for 48x32x3 enter 48
- valid_per : float number between 0,1 --- Validation percantage, train dataset will be splitted as train and validation
- epochs : integer number --- Number of epochs to train
- learning_rate : float number  --- Learning rate for optimizer
- beta1 : float number between 0,1 --- Beta1 value for adam optimizer
- beta2 : float number between 0,1 --- Beta2 value for adam optimizer
- n_classes : integer number --- Number of classes that dataset has
- cuda : True or false --- if you have cuda compute capability enviroment True suggested
- -eveluate_per_epoch : integer number --- Prints the state information per epoch
- save_model : True or False --- If true model parameters will be saved.
- model_params : Path --- If you have pretrained parameters, enter the path


###### Custom dataset mode should include following arguments: mode,dataset,train_path,n_classes,im_size

