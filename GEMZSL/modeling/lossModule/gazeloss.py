import torch 
from scipy.optimize import linear_sum_assignment
from torch import nn

class gazeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, M, G):
        """ Performs matching and calculate loss based on the matching
        Params:
            M: The attribute based hotmap, Tensor of dim [B, K, W, H], range(0, 1)
                

            G: gaze map, Tensor of dim [B, K, W, H], range(0, 1)

        Mid outputs:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected attribute-based hotmap (in order)
                - index_j is the indices of the corresponding selected gaze targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = K
        Returns:
            loss: bce loss to make the overall hotmap to be like gazemap
        """
        bs, k = M.shape[:2]
        M0 = M.flatten(0, 1).flatten(1,2)
        G0 = G.flatten(0, 1).flatten(1,2)
        cost = torch.cdist(M0, G0, p=1)
        cost = cost.view(bs, k ,-1).cpu()

        indices = [linear_sum_assignment(c[i].detach().numpy()) for i, c in enumerate(cost.split(k, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])


        M = M[(batch_idx, src_idx)]
        G = G[(batch_idx, tgt_idx)]
        loss = self.bce(M, G)
        return loss
