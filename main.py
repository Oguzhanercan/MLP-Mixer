import argparse
import os
import data_works
import train
import mlp_mixer as mm
import test 
import torch.nn as nn
import torch
def main(args):
    

    print("Unofficial implementation of MLP Mixer, for more information about model follow the link: https://arxiv.org/abs/2105.01601")
    if args.mode == None:
        print(parser.description)
        print("Enter arguments, mode and dataset arguments must be entered. Example --mode train --dataset CIFAR10")
        arguments = input()
        os.system(os.getcwd() + "/main.py " + arguments)
    
    model = mm.Mixer(num_classes= args.n_classes, img_size = args.im_size)
    if args.model_params != "none":
        try:
            model.load_state_dict(torch.load(args.load_params))
        except:
            print("Invalid path for model parameters")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate,betas = (args.beta1,args.beta2),weight_decay=args.weight_decay)
    
    
    
    if args.mode == "train":
        trainloader, validloader = data_works.get_dataloader(args)
        train.train(args,trainloader,validloader,model,optimizer,criterion)
    elif args.mode == "test":
        testloader = data_works.get_dataloader(args)
        result = test.test(args,testloader,model,criterion)
        print("Test accuracy : " + str(result))
    else:
        print("invalid mode")
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments: --dataset\n --train_path\n--test_path\n--batch_size\n--im_size\nvalid_per\n--epochs"
    "\n--learning_rate\n--beta1\n--beta2\n--n_classes\n--cuda\n--eveluate_per_epoch\n--save_model\n--model_path \n CUSTOM DATASET MODE: \n Custom dataset mode should include following arguments: mode,dataset,train_path,n_classes,im_size ")



    parser.add_argument("--mode",
                        "-m",
                        help = "train or test",
                        type = str,
                        default= None
    )

    parser.add_argument("--dataset",
                        "-ds",
                        help = "CIFAR10,MNIST,MNIST FASHION is avaliable, anathor option is custom. Please examine the description for custom dataset. Options: CIFAR10-MNIST-MNIST_F-CUSTOM",
                        type = str,
                        default= None
    )


   
    parser.add_argument("--train_path",
                        "-train",
                        help="Path to train data",
                        type = str,
    )

    parser.add_argument("--test_path",
                        "-test",
                        help="Path to test data",
                        type = str,
    )

    parser.add_argument("--batch_size",
                        "-bs",
                        help="batch size",
                        type = int,
                        default = 4
    )
    parser.add_argument("--num_workers",
                        "-nm",
                        help="Number of workers to load data",
                        type = int,
                        default = 2
    )
    parser.add_argument("--im_size",
                        "-im",
                        help="Size of image, for 224x224 images enter 224",
                        type = int,
                        default = 32
    )

    parser.add_argument("--valid_per",
                        "-vp",
                        help="The percantage of validation, must be between (0,1). 0.2 = %20",
                        type = float,
                        default = 0.2
    )

    
    parser.add_argument("--epochs",
                        "-e",
                        help="number of epochs",
                        type = int,
                        default = 20
    )

    
    parser.add_argument("--learning_rate",
                        "-ilr",
                        help="initial learning rate for Adam",
                        type = float,
                        default = 0.001
    )

    parser.add_argument("--weight_decay",
                        "-wd",
                        help="Weight decay for Adam",
                        type = float,
                        default = 0.3,
    )
    parser.add_argument("--beta1",
                        "-b1",
                        help="Beta1 for Adam",
                        type = float,
                        default = 0.001,)

    parser.add_argument("--beta2",
                        "-b2",
                        help="initial learning rate for Adam",
                        type = float,
                        default = 0.001,)


    parser.add_argument("--device",
                        "-dvc",
                        help="cuda or cpu",
                        type = str,
                        default = "cpu"
    )


    parser.add_argument("--n_classes",
                        "-ncl",
                        help="number of classes",
                        type = int,
                        default = 10
    )

   
    parser.add_argument("--evaluate_per_epoch",
                        "-epe",
                        help="evaluating at nth epoch",
                        type = int,
                        default = 2
    )

    parser.add_argument("--save",
                        "-ml",
                        help="Save model parameters, True or False ",
                        type = bool,
                        default = False
    )


    parser.add_argument("--model_params",
                        "-mlp",
                        help="path of .pt file to load",
                        type = str,
                        default = "none"
    )
   

    args = parser.parse_args()
main(args)