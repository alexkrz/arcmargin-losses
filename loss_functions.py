import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, in_features, out_features, loss_type="arcface", eps=1e-7, s=None, m=None):
        """
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        """
        super().__init__()
        loss_type = loss_type.lower()
        assert loss_type in ["arcface", "sphereface", "cosface"]
        if loss_type == "arcface":
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == "sphereface":
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == "cosface":
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        """
        input shape (N, in_features)
        """
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        # fmt: off
        if self.loss_type == "cosface":
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == "arcface":
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == "sphereface":
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))
        # fmt: on

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class ArcMarginHeader(nn.Module):
    """
    ArcMarginHeader class
    Adjusted ArcMarginProduct class from https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 1.0,
        m1: float = 1.0,
        m2: float = 0.0,
        m3: float = 0.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.epsilon = 1e-6

    def forward(self, input, label):
        # Cosine similarity between normalized input and normalized weight
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight), bias=None)
        cos_theta = cos_theta.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)

        # Get the angle (theta) between input and weight vectors
        theta = torch.acos(cos_theta)

        # Apply the angular margin penalty
        cos_theta_m = torch.cos(self.m1 * theta + self.m2) - self.m3

        # One-hot encode labels for the corresponding class weight
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * cos_theta_m + (1 - one_hot) * cos_theta

        # Scale the output
        output *= self.s
        return output


class ArcFaceHeader(ArcMarginHeader):
    """
    ArcFaceHeader class
    Reference: https://ieeexplore.ieee.org/document/8953658 (CVPR, 2019)
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__(in_features=in_features, out_features=out_features, s=s, m2=m)


class CosFaceHeader(ArcMarginHeader):
    """
    CosFaceHeader class
    Reference: https://ieeexplore.ieee.org/document/8578650 (CVPR, 2018)
    """

    def __init__(self, in_features, out_features, s=1.0, m=0.35):
        super().__init__(in_features=in_features, out_features=out_features, s=s, m3=m)


class SphereFaceHeader(ArcMarginHeader):
    """
    SphereFaceHeader class
    Reference: https://ieeexplore.ieee.org/document/8100196 (CVPR, 2017)
    """

    def __init__(self, in_features, out_features, m=4.0):
        super().__init__(in_features=in_features, out_features=out_features, s=1, m1=m)
