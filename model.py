import torch.nn as nn
import torchvision.models as models
import torch


class ModelBase(nn.Module):
    def __init__(self, feature_dim=128):
        super(ModelBase, self).__init__()

        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, feature_dim)

        # Debugging print
        # print(self.encoder)

    def forward(self, x):
        x = self.encoder(x)
        return x


class MoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_query = ModelBase(feature_dim=dim)
        self.encoder_key = ModelBase(feature_dim=dim)

        for param_query, param_key in zip(
            self.encoder_query.parameters(), self.encoder_key.parameters()
        ):
            param_key.data.copy_(param_query.data)
            param_key.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def ema_update(self):
        for param_query, param_key in zip(
            self.encoder_query.parameters(), self.encoder_key.parameters()
        ):
            param_key.data = param_key.data * self.m + param_query.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # For simplicity i allowed only dictionary sizes that are multiples of the batch size.
        assert self.K % batch_size == 0

        self.queue[:, ptr : ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def shuffle_batch(self, batch):
        idx_shuffle = torch.randperm(batch.shape[0]).cuda()

        idx_unshuffle = torch.argsort(idx_shuffle)

        return batch[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def unshuffle_batch(self, batch, unshuffle_idx):
        return batch[unshuffle_idx]

    def contrastive_loss(self, im_q, im_k):
        q = self.encoder_query(im_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            im_k_, idx_unshuffle = self.shuffle_batch(im_k)
            k = self.encoder_key(im_k_)
            k = nn.functional.normalize(k, dim=1)
            k = self.unshuffle_batch(k, idx_unshuffle)

        # Compute logits to be passed to the Cross Entropy Loss
        # Positive logits: Nx1
        positive_logits = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits: NxK

        negative_logits = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # Logits: Nx(1+K)

        logits = torch.cat([positive_logits, negative_logits], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, im1, im2):
        with torch.no_grad():
            self.ema_update()

        loss, _, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss
