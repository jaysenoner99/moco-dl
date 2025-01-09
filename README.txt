Instruction to reproduce the experimental results:

Dataset:

MiniImageNet Dataset: https://www.kaggle.com/datasets/arjunashok33/miniimagenet


To setup MiniImageNet, download the dataset from the link above and put it into the "Dataset/CLEAR" directory.
Then run the split_dataset.py script.


MoCo pretraining:

--First of all, setup a virtual environment in the project directory, activate it and launch the command:

  pip install -r requirements.txt

  to install all the required dependencies.


--To pretrain a MoCo model on MiniImageNet, run the following command:

  python3 main.py [--name] [--lr] [--epochs] [--schedule] [--cos] [--batch-size] [--moco-dim] [--moco-k] [--moco-m] [--moco-t] 
                  [--knn-k] [--knn-t] [--resume] [--results-dir]


  --name : String
        Name of the project. Defaults to "".

  --lr : float
        Initial Learning rate: Defaults to 0.03

  --epochs: int
        Number of training epochs. Defaults to 200

  --schedule: List<int>
        Milestones for the MultiStepLR learning rate scheduler. Defaults to [120,160]

  --cos: bool
        If True, use the CosineAnnealingLR learning rate scheduler. Defaults to False

  --batch-size: int 
        Number of samples in the minibatch. Defaults to 128.

  --wd: float
        Weight decay of the SGD optimizer. Defaults to 1e-4

  --moco-dim: int 
        Output dimension of the final linear layer in the ResNet backbone network. Defaults to 128

  --moco-k: int
        Dimension of the dictionary(queue) in MoCo. Defaults to 4096

  --moco-m: float
        MoCo momentum of updating key encoder. Defaults to 0.999

  --moco-t: float
       Softmax temperature for calculating the InfoNCE loss in MoCo. Defaults to 0.07

  --knn-k: int
       Number of neighbors to be considered in the KNN monitor. Defaults to 200.
  
  --knn-t: float
       Softmax temperature in the KNN monitor. Defaults to 0.1
       
  --resume: String
      Path of the checkpoint that needs to be resumed. Defaults to ""

  --results-dir: String
      Path of the directory where the main results will be saved. Defaults to "./results/"


The obtained pretrained models will be saved under the "./trained_models/" directory.


MoCo Linear Evaluation on MiniImageNet and CIFAR10.

--To evaluate the pretrained MoCo models in the task of classification on MiniImageNet/CIFAR10 with the linear evaluation protocol,
  run the following command:

  python3 eval.py [--lr] [--epochs] [--momentum] [--schedule] [--batch-size] [--moco-dim] [--moco-k] [--moco-m] [--moco-t] 
                  [--cifar] [--miniin] [--results-dir] [--path]

  --lr: float
      Learning rate of SGD optimizer. Defaults to 0.5 

  --epochs: int
      Number of training epochs. Defaults to 100

  --momentum: float
      Momemtum of SGD optimizer. Defaults to 0.9

  --schedule: List<int>
      Milestones for the MultiStepLR learning rate scheduler. Defaults to [60,80]

  --cifar: bool
      If True, evaluate the pretrained model by linear probing on CIFAR10

  --miniin: bool
      If True, evaluate the pretrained model by linear probing on MiniImageNet

  --results-dir: String
      Path of the directory where the results will be saved. Defaults to "./eval_results/"

  --path: String
      Path of the pretrained model to be evaluated. Defaults to "./trained_models/"

The parameters that were not listed in this sections are the same of the previous section and have the same default values.

This project logs relevant metrics using Comet ML. The results can be accessed by reaching the following URL:

  https://www.comet.com/jaysenoner99/deep-learning/view/new/panels


