import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMixConLoss(nn.Module):
    def __init__(self, temperature=0.06):
        super().__init__()
        self.temperature = temperature

    def forward(self, projection1, projection2, labels_a, labels_b, lam, index, args):
        batch_size = projection1.shape[0]
        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        anchor_dot_contrast = torch.div(torch.matmul(projection2, projection1.T), self.temperature)

        mask_a = torch.eye(batch_size).cuda()
        mask_b = torch.zeros(batch_size, batch_size).cuda()
        mask_b[torch.arange(batch_size).unsqueeze(1), index.view(-1, 1)] = 1

        mask = lam * mask_a + (1 - lam) * mask_b

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability

        exp_logits = torch.exp(logits)
        if args.negative_pair == 'diff_label':
            labels_a = labels_a.contiguous().view(-1, 1)
            logits_mask = torch.ne(labels_a, labels_a.T).cuda() + (mask_a.bool() + mask_b.bool())
            exp_logits *= logits_mask.float()

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos
        loss = loss.view(1, batch_size)

        loss = loss.mean()   
        return loss
