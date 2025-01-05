import comet_ml
from torch.utils.data import random_split
import torch.nn as nn
from tqdm import tqdm
import torch
import os
import json
import pandas as pd
import argparse
from model import MoCo
from dataloader import MiniImageNetDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets


def test(model, test_loader, criterion, epoch, eval_args):
    model.eval()
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    total_val_loss = 0.0

    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for images, target in test_bar:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            total_val_loss += loss.item() * images.size(0)
            _, pred = output.topk(5, 1, True, True)  # get top-5 (includes top-1)
            pred = pred.t()  # transpose to shape [k, batch_size]
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_top1 = correct[0].float().sum().item()  # just the first prediction
            correct_top5 = correct[:5].float().sum().item()  # any of top 5 predictions
            total_top1 += correct_top1
            total_top5 += correct_top5
            total_num += images.size(0)
            test_bar.set_description(
                "Test Epoch: [{}/{}] Eval_Loss: {:.4f}, Acc@1:{:.2f}%, Acc@5:{:.2f}%".format(
                    epoch,
                    eval_args.epochs,
                    total_val_loss / total_num,
                    total_top1 / total_num * 100,
                    total_top5 / total_num * 100,
                )
            )

    return (
        (total_top1 / total_num) * 100,
        (total_top5 / total_num) * 100,
        total_val_loss / total_num,
    )


def train(model, train_loader, optimizer, criterion, epoch, eval_args):
    model.eval()
    adjust_learning_rate(optimizer, epoch, eval_args)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)
    for image, target in train_bar:
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_num += train_loader.batch_size
        total_loss += loss.item() * train_loader.batch_size
        train_bar.set_description(
            "Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}".format(
                epoch,
                eval_args.epochs,
                optimizer.param_groups[0]["lr"],
                total_loss / total_num,
            )
        )

    return total_loss / total_num


def adjust_learning_rate(optimizer, epoch, eval_args):
    """Decay the learning rate based on schedule"""
    lr = eval_args.lr
    for milestone in eval_args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def linear_eval(
    model, train_loader, test_loader, eval_args, num_classes, exp, plot_name=""
):
    results = {"lin_train_loss": [], "lin_eval_acc@1": [], "lin_eval_acc@5": []}

    if not os.path.exists(eval_args.results_dir):
        os.mkdir(eval_args.results_dir)
    with open(eval_args.results_dir + "/eval_args.json", "w") as fid:
        json.dump(eval_args.__dict__, fid, indent=2)

    for param in model.parameters():
        param.requires_grad = False

    model.encoder.fc = nn.Linear(model.encoder.fc.in_features, num_classes)
    model.encoder.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.encoder.fc.bias.data.zero_()
    model.encoder.fc.weight.requires_grad = True
    model.encoder.fc.bias.requires_grad = True

    criterion = nn.CrossEntropyLoss().cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2
    optimizer = torch.optim.SGD(
        parameters,
        eval_args.lr,
        momentum=eval_args.momentum,
        weight_decay=eval_args.wd,
    )

    model = model.cuda()
    for epoch in tqdm(range(1, eval_args.epochs + 1)):
        lin_train_loss = train(
            model, train_loader, optimizer, criterion, epoch, eval_args
        )
        results["lin_train_loss"].append(lin_train_loss)
        lin_eval_acc_1, lin_eval_acc_5, lin_eval_loss = test(
            model, test_loader, criterion, epoch, eval_args
        )
        results["lin_eval_acc@1"].append(lin_eval_acc_1)
        results["lin_eval_acc@5"].append(lin_eval_acc_5)
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(eval_args.results_dir + "/eval_log.csv", index_label="epoch")
        exp.log_metric("lin_train_loss " + plot_name, lin_train_loss, step=epoch)
        exp.log_metric("lin_eval_loss " + plot_name, lin_eval_loss, step=epoch)
        exp.log_metric("lin_eval_acc_1 " + plot_name, lin_eval_acc_1, step=epoch)
        exp.log_metric("lin_eval_acc_5 " + plot_name, lin_eval_acc_5, step=epoch)


def init_eval_cifar10(eval_args):
    transform_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),  # Flips the image horizontally with 50% probability
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./cifar", train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root="./cifar", train=False, download=True, transform=transform_test
    )

    trainset, valset = random_split(trainset, [40000, 10000])

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=eval_args.batch_size,
        shuffle=True,
        num_workers=16,
        drop_last=False,
        pin_memory=True,
    )

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=eval_args.batch_size,
        shuffle=False,
        num_workers=16,
        drop_last=False,
        pin_memory=True,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=eval_args.batch_size,
        shuffle=False,
        num_workers=16,
        drop_last=False,
        pin_memory=True,
    )
    return trainset, trainloader, valloader, testloader


def init_eval_miniImageNet(eval_args):
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                std=[0.229, 0.224, 0.225],
            ),  # Normalize with ImageNet std
        ]
    )

    test_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                std=[0.229, 0.224, 0.225],
            ),  # Normalize with ImageNet std
        ]
    )

    testset = MiniImageNetDataset(
        root_dir="./Dataset/SPLITTED/Test/", mode="eval", transform=test_transform
    )

    testloader = DataLoader(
        testset,
        batch_size=eval_args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    trainset = MiniImageNetDataset(
        root_dir="./Dataset/SPLITTED/Train/", mode="eval", transform=train_transform
    )
    dataset_length = len(trainset)
    train_size = int(0.8 * dataset_length)
    val_size = dataset_length - train_size

    # Split the dataset into training and validation sets
    train_subset, val_subset = random_split(trainset, [train_size, val_size])

    trainloader = DataLoader(
        train_subset,
        batch_size=eval_args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    valloader = DataLoader(
        val_subset,
        batch_size=eval_args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    return trainset, trainloader, valloader, testloader


def main():
    exp = comet_ml.Experiment(
        api_key="UShvCEYUvHN87Fc42mwbWhPMq",
        project_name="Deep Learning",
        auto_metric_logging=False,
        auto_param_logging=False,
    )

    parser = argparse.ArgumentParser(
        description="Linear evaluation of pretrained MoCo model"
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=30,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )

    parser.add_argument(
        "--cifar",
        action="store_true",
        help="If True, perform linear evaluation on CIFAR10",
    )

    parser.add_argument(
        "--miniin",
        action="store_true",
        help="if True, perform linear evaluation on MiniImageNet",
    )

    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "--batch-size", default=128, type=int, metavar="N", help="mini-batch size"
    )

    parser.add_argument("--wd", default=0, type=float, metavar="W", help="weight decay")

    parser.add_argument(
        "--schedule",
        default=[60, 80],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )

    parser.add_argument(
        "--momentum", default=0.9, type=float, help="momentum of SGD optimizer"
    )

    parser.add_argument(
        "--results-dir",
        default="./eval_results/",
        type=str,
        metavar="PATH",
        help="where to save results",
    )

    parser.add_argument(
        "--path",
        default="./trained_models/",
        type=str,
        metavar="PATH",
        help="path to the pretrained model",
    )

    parser.add_argument(
        "--moco-k",
        default=8192,
        type=int,
        help="Dictionary size of pretrained moco model",
    )
    parser.add_argument(
        "--moco-m",
        default=0.999,
        type=float,
        help="momentum of key encoder in the pretrained model",
    )

    eval_args = parser.parse_args()

    parameters = {
        "batch_size": eval_args.batch_size,
        "epochs": eval_args.epochs,
        "learning_rate": eval_args.lr,
        "momentum": eval_args.momentum,
        "weight-decay": eval_args.wd,
        "moco-k": eval_args.moco_k,
        "schedule": eval_args.schedule,
        "moco-m": eval_args.moco_m,
    }

    exp.log_parameters(parameters)

    if not os.path.isfile(eval_args.path):
        raise FileNotFoundError(f"Checkpoint file not found: {eval_args.path}")

    print(f"Loading checkpoint from {eval_args.path}")

    checkpoint = torch.load(eval_args.path, map_location="cuda")

    test_results = dict.fromkeys(["model", "dataset", "test_top1_acc", "test_top5_acc"])
    test_results["model"] = eval_args.path
    model = MoCo(
        dim=128,
        K=eval_args.moco_k,
        m=eval_args.moco_m,
        T=0.07,
        symmetric=False,
    ).cuda()

    model.load_state_dict(checkpoint["state_dict"], strict=False)

    if eval_args.cifar:
        _, trainloader, valloader, testloader = init_eval_cifar10(eval_args)

        num_classes = 10
        test_results["dataset"] = "cifar"
        linear_eval(
            model.encoder_query,
            trainloader,
            valloader,
            eval_args,
            num_classes,
            exp,
            "cifar10",
        )

        # After training the linear prediction head on top of the learned representation, evaluate the performances
        # of the whole model on the test set.
        top1_acc, top5_acc, _ = test(
            model.encoder_query,
            testloader,
            nn.CrossEntropyLoss().cuda(),
            eval_args.epochs,
            eval_args,
        )

        print(
            f"Accuracy on the Test Set: Acc@1: {top1_acc:.2f}%, Acc@5: {top5_acc:.2f}%"
        )
        test_results["test_top1_acc"] = top1_acc
        test_results["test_top5_acc"] = top5_acc

        data_frame = pd.DataFrame(data=test_results, index=range(3))
        data_frame.to_csv(eval_args.results_dir + "/log_test_results.csv")

        exp.log_metric("cifar_test_top1_acc", top1_acc)
        exp.log_metric("cifar_test_top5_acc", top5_acc)

    elif eval_args.miniin:
        _, trainloader, valloader, testloader = init_eval_miniImageNet(eval_args)

        num_classes = 100

        test_results["dataset"] = "miniin"
        linear_eval(
            model.encoder_query,
            trainloader,
            valloader,
            eval_args,
            num_classes,
            exp,
        )

        # After training the linear prediction head on top of the learned representation, evaluate the performances
        # of the whole model on the test set.
        top1_acc, top5_acc, _ = test(
            model.encoder_query,
            testloader,
            nn.CrossEntropyLoss().cuda(),
            eval_args.epochs,
            eval_args,
        )

        print(
            f"Accuracy on the Test Set: Acc@1: {top1_acc:.2f}%, Acc@5: {top5_acc:.2f}%"
        )

        test_results["test_top1_acc"] = top1_acc
        test_results["test_top5_acc"] = top5_acc

        exp.log_metric("test_top1_acc", top1_acc)
        exp.log_metric("test_top5_acc", top5_acc)
        data_frame = pd.DataFrame(data=test_results, index=range(1))
        data_frame.to_csv(eval_args.results_dir + "/log_test_results.csv")


if __name__ == "__main__":
    main()
