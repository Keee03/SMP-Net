import torch
import torch.nn as nn


class Neg_Pearson(nn.Module):
    def __init__(self):
        super(Neg_Pearson,self).__init__()
    def forward(self, preds, labels):
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])
            sum_y = torch.sum(labels[i])
            sum_xy = torch.sum(preds[i]*labels[i])
            sum_x2 = torch.sum(torch.pow(preds[i],2))
            sum_y2 = torch.sum(torch.pow(labels[i],2))
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += 1 - pearson
        loss = loss/preds.shape[0]
        return loss