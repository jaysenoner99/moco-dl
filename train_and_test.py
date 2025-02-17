from tqdm import tqdm
import math
import torch
import torch.nn.functional as F


# Train for a single epoch
def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(train_optimizer, epoch, args)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2 in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)

        loss = net(im_1, im_2)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += im_1.size(0)
        total_loss += loss.item() * im_1.size(0)
        train_bar.set_description(
            "Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}".format(
                epoch,
                args.epochs,
                train_optimizer.param_groups[0]["lr"],
                total_loss / total_num,
            )
        )
    return total_loss / total_num


# Adjust the learning rate to follow a cosine decay schedule or a multi step decay schedule
def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # multi step lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Test the model during pretraining using a knn monitor
def test(net, memory_data_loader, test_data_loader, epoch, args):
    net.eval()
    classes = memory_data_loader.dataset.num_classes
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank for the knn monitor
        for data, target in tqdm(memory_data_loader, desc="Feature extracting"):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.tensor(
            memory_data_loader.dataset.targets, device=feature_bank.device
        )
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(
                feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t
            )

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            total_top5 += sum(
                [
                    target[i].item() in pred_labels[i, :5].tolist()
                    for i in range(target.size(0))
                ]
            )
            test_bar.set_description(
                "Test Epoch: [{}/{}] Acc@1:{:.2f}%, Acc@5:{:.2f}%".format(
                    epoch,
                    args.epochs,
                    total_top1 / total_num * 100,
                    total_top5 / total_num * 100,
                )
            )

    return total_top1 / total_num * 100, total_top5 / total_num * 100


# Function to compute the predicted label using a KNN.
#
# -Compute the similarity matrix between the query features and the feature bank using matrix multiplication.
# -Select the top-k nearest neighbors (similarity scores and indices).
# -Retrieve the labels of the top-k neighbors and scale the similarity weights using the temperature parameter.
# -Create one-hot encodings for the labels of the top-k neighbors.
# -Compute the prediction scores by summing the weighted one-hot encodings for each class.
# -Return the predicted labels, sorted by descending prediction score.
#


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, classes, device=sim_labels.device
    )
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
