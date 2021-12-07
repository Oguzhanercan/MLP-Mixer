# MLP-Mixer
### Unoffical Implementation of MLP-Mixer, easy to use with terminal. Train and test easly.

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
- dataset
- train_path
- test_path
- batch_size
- im_size
- valid_per
- epochs
- learning_rate
- beta1
- beta2
- n_classes
- cuda
- -eveluate_per_epoch
- save_model
- model_path

###### Custom dataset mode should include following arguments: mode,dataset,train_path,n_classes,im_size

