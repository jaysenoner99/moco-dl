# MoCo Pretraining on MiniImageNet

This repository implements a PyTorch-based framework for training a Momentum Contrast (MoCo) model on the MiniImageNet dataset. The pretrained model is evaluated via linear probing on MiniImageNet and CIFAR-10.

## Dataset

The training and evaluation datasets used in this project are:
- [MiniImageNet](https://www.kaggle.com/datasets/arjunashok33/miniimagenet)
- CIFAR-10 (downloaded automatically by the PyTorch dataset utilities)

To set up MiniImageNet, download the dataset from the link above and place it into the `Dataset/CLEAR` directory. Then, run the `split_dataset.py` script.

## Installation

### Requirements
Ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/jaysenoner99/moco-dl.git
cd moco-dl
```

## MoCo Pretraining

### Setting Up the Virtual Environment
First, set up a virtual environment in the project directory, activate it, and install dependencies:

```bash
pip install -r requirements.txt
```

### Training MoCo
To pretrain a MoCo model on MiniImageNet, run:

```bash
python3 main.py [--name] [--lr] [--epochs] [--schedule] [--cos] [--batch-size] [--moco-dim] [--moco-k] [--moco-m] [--moco-t] \
                [--knn-k] [--knn-t] [--resume] [--results-dir]
```

#### Arguments:
- `--name`: Name of the project (default: "").
- `--lr`: Initial learning rate (default: 0.03).
- `--epochs`: Number of training epochs (default: 200).
- `--schedule`: Milestones for the MultiStepLR learning rate scheduler (default: [120,160]).
- `--cos`: Use the CosineAnnealingLR learning rate scheduler (default: False).
- `--batch-size`: Number of samples per minibatch (default: 128).
- `--wd`: Weight decay of SGD optimizer (default: 1e-4).
- `--moco-dim`: Output dimension of the final linear layer in the ResNet backbone (default: 128).
- `--moco-k`: Dimension of the dictionary (queue) in MoCo (default: 4096).
- `--moco-m`: MoCo momentum of updating key encoder (default: 0.999).
- `--moco-t`: Softmax temperature for calculating the InfoNCE loss (default: 0.07).
- `--knn-k`: Number of neighbors in the KNN monitor (default: 200).
- `--knn-t`: Softmax temperature in the KNN monitor (default: 0.1).
- `--resume`: Path of the checkpoint to resume (default: "").
- `--results-dir`: Directory where results will be saved (default: "./results/").

Pretrained models are saved in the `./trained_models/` directory.

## MoCo Linear Evaluation on MiniImageNet and CIFAR-10

To evaluate the pretrained MoCo model with linear probing on MiniImageNet or CIFAR-10, run:

```bash
python3 eval.py [--lr] [--epochs] [--momentum] [--schedule] [--batch-size] [--moco-dim] [--moco-k] [--moco-m] [--moco-t] \
                [--cifar] [--miniin] [--results-dir] [--path]
```

#### Arguments:
- `--lr`: Learning rate of SGD optimizer (default: 0.5).
- `--epochs`: Number of training epochs (default: 100).
- `--momentum`: Momentum of SGD optimizer (default: 0.9).
- `--schedule`: Milestones for MultiStepLR (default: [60,80]).
- `--cifar`: Evaluate on CIFAR-10 (default: False).
- `--miniin`: Evaluate on MiniImageNet (default: False).
- `--results-dir`: Directory where evaluation results will be saved (default: "./eval_results/").
- `--path`: Path to the pretrained model to be evaluated (default: "./trained_models/").

Parameters not listed here retain the same values as in the pretraining phase.

## Experiment Tracking

All experiment results are tracked using [Comet](https://www.comet.com/jaysenoner99/deep-learning/view/new/panels). Logs, metrics, and visualizations can be accessed there.

## Results

The performance of the pretrained model evaluated via linear probing is documented in the Comet dashboard. Metrics such as accuracy and loss trends are available.

## Acknowledgments

This implementation is inspired by the original MoCo paper:
> He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9729-9738).

## License
This project is licensed under the MIT License.

