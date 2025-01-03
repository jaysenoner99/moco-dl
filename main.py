import comet_ml
import argparse
from model import MoCo
from dataloader import MiniImageNetDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch
import os
import json
from train_and_test import train, test
import pandas as pd
from tqdm import tqdm


# MoCo parameters:
#
#
#
#
#
#
#
def main():
    parser = argparse.ArgumentParser(description="Train MoCo on MiniImageNet")
    exp = comet_ml.Experiment(
        api_key="UShvCEYUvHN87Fc42mwbWhPMq",
        project_name="Deep Learning",
        auto_metric_logging=False,
        auto_param_logging=False,
    )

    parser.add_argument("--name", default="", type=str, help="name of the project")

    # lr: 0.06 for batch 512 (or 0.03 for batch 256)
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.03,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--schedule",
        default=[120, 160],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

    parser.add_argument(
        "--batch-size", default=128, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--wd", default=1e-4, type=float, metavar="W", help="weight decay"
    )

    # moco specific configs:
    parser.add_argument("--moco-dim", default=128, type=int, help="feature dimension")
    parser.add_argument(
        "--moco-k", default=4096, type=int, help="queue size; number of negative keys"
    )
    parser.add_argument(
        "--moco-m",
        default=0.999,
        type=float,
        help="moco momentum of updating key encoder",
    )
    parser.add_argument(
        "--moco-t", default=0.07, type=float, help="softmax temperature"
    )

    # Argument to parse for bn splits
    # parser.add_argument(
    #     "--bn-splits",
    #     default=8,
    #     type=int,
    #     help="simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu",
    # )

    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="use a symmetric loss function that backprops to both crops",
    )

    # knn monitor
    parser.add_argument("--knn-k", default=200, type=int, help="k in kNN monitor")
    parser.add_argument(
        "--knn-t",
        default=0.1,
        type=float,
        help="softmax temperature in kNN monitor; could be different with moco-t",
    )

    # utils
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--results-dir",
        default="./results",
        type=str,
        metavar="PATH",
        help="path to cache (default: none)",
    )
    args = parser.parse_args()
    # MoCo train augmentations
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                std=[0.229, 0.224, 0.225],
            ),  # Normalize with ImageNet std
        ]
    )
    # MiniImageNet_mean = [0.4727902,  0.44887177, 0.404713]
    # MiniImageNet_std = [0.28407582, 0.2758255,  0.29091981]

    test_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                std=[0.229, 0.224, 0.225],
            ),  # Normalize with ImageNet std
        ]
    )
    model = MoCo(
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        symmetric=args.symmetric,
    ).cuda()

    experiment_name = args.name
    parameters = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "momentum": 0.9,
        "weight_decay": args.wd,
        "moco_k": args.moco_k,
        "moco_m": args.moco_m,
        "symmetric": args.symmetric,
    }
    exp.log_parameters(parameters)
    train_data = MiniImageNetDataset(
        root_dir="./Dataset/SPLITTED/Train/", mode="train", transform=train_transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    val_data = MiniImageNetDataset(
        root_dir="./Dataset/SPLITTED/Test/", mode="eval", transform=test_transform
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    memory_data = MiniImageNetDataset(
        root_dir="./Dataset/SPLITTED/Train/", mode="eval", transform=test_transform
    )
    memory_loader = DataLoader(
        memory_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9
    )

    # load model if resume
    epoch_start = 1
    if args.resume != "":
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_start = checkpoint["epoch"] + 1
        print("Loaded from: {}".format(args.resume))

    # logging
    results = {"train_loss": [], "test_acc@1": [], "test_acc@5": []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    # dump args
    with open(args.results_dir + "/args" + experiment_name + ".json", "w") as fid:
        json.dump(args.__dict__, fid, indent=2)

    # training loop
    for epoch in tqdm(range(epoch_start, args.epochs + 1)):
        train_loss = train(model, train_loader, optimizer, epoch, args)
        results["train_loss"].append(train_loss)
        test_acc_1, test_acc_5 = test(
            model.encoder_query, memory_loader, val_loader, epoch, args
        )
        results["test_acc@1"].append(test_acc_1)
        results["test_acc@5"].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(
            args.results_dir + "/log" + experiment_name + ".csv", index_label="epoch"
        )
        # save model
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            args.results_dir + "/model_last" + experiment_name + ".pth",
        )

        exp.log_metric("loss", train_loss, step=epoch)
        exp.log_metric("knn_acc_top1", test_acc_1, step=epoch)
        exp.log_metric("knn_acc_top5", test_acc_5, step=epoch)


if __name__ == "__main__":
    main()
