import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

class CategoriesSampler():

    def __init__(self, label_for_imgs, n_batch, n_cls, n_per, ep_per_batch=1):
        self.n_batch = n_batch # batchs for each epoch
        self.n_cls = n_cls # ways
        self.n_per = n_per # shots
        self.ep_per_batch = ep_per_batch # episodes for each batch, defult set 1

        self.cat =  list(np.unique(label_for_imgs))
        self.catlocs = {}

        for c in self.cat:
            self.catlocs[c] = np.argwhere(label_for_imgs == c).reshape(-1)


    def __len__(self):
        return  self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                selected_classes = np.random.choice(self.cat, self.n_cls, replace=False)

                for c in selected_classes:
                    l = np.random.choice(self.catlocs[c], self.n_per, replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_cls * n_per
            yield batch.view(-1)

class DCategoriesSampler(Sampler):

    def __init__(self, label_for_imgs, n_batch, n_cls, n_per, ep_per_batch=1, num_replicas=None, rank=None):

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        self.n_batch = n_batch # batchs for each epoch
        self.n_cls = n_cls # ways
        self.n_per = n_per # shots
        self.num_samples = self.n_cls * self.n_per

        self.ep_per_batch = ep_per_batch # episodes for each batch, defult set 1

        self.cat =  list(np.unique(label_for_imgs))
        self.catlocs = {}

        for c in self.cat:
            self.catlocs[c] = np.argwhere(label_for_imgs == c).reshape(-1)


    def __len__(self):
        return  self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                selected_classes = np.random.choice(self.cat, self.n_cls * self.num_replicas, replace=False)

                for c in selected_classes:
                    l = np.random.choice(self.catlocs[c], self.n_per, replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_cls * n_per
            batch = batch.view(-1)

            # subsample
            offset = self.num_samples * self.rank
            batch = batch[offset: offset + self.num_samples]
            assert len(batch) == self.num_samples


            yield batch

