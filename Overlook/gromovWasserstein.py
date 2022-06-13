# coding:utf-8
import torch.nn as nn
import torch 
from torch.autograd import Variable
class gromovWasserstein(nn.Module):
    def __init__(self,beta=0.5,affinity_type='cosine',l_type='KL'):
        super(gromovWasserstein, self).__init__()
        self.affinity_type=affinity_type
        self.l_type=l_type
        self.beta = beta
        self.rate = 0.99
        self.iter_num = 50
        print("gw add rate is :"+str(self.beta))
    
    def forward(self,feat_stu,feat_tea,t):
        affinity_stu = self.affinity_matrix(feat_stu)
        affinity_tea = self.affinity_matrix(feat_tea)
        T = torch.eye(feat_stu.size(0)).cuda()

        if type(t)!=int:
            T = self.beta*t + T
        T = T/T.sum()
        cost = self.L(affinity_stu,affinity_tea,T)
        loss = (cost * T).sum()
        return loss
    
    def affinity_matrix_cross(self,feat1,feat2):
        if self.affinity_type=='cosine':
            energy1 = torch.sqrt(torch.sum(feat1 ** 2, dim=1, keepdim=True))  # (batch_size, 1)
            energy2 = torch.sqrt(torch.sum(feat2 ** 2, dim=1, keepdim=True))
            cos_sim = torch.matmul(feat1, torch.t(feat2)) / (torch.matmul(energy1, torch.t(energy2)))
            affinity = cos_sim
        else:
            pass
        return affinity

    def affinity_matrix(self,feat):
        if self.affinity_type=='cosine':
            energy = torch.sqrt(torch.sum(feat ** 2, dim=1, keepdim=True))  # (batch_size, 1)
            cos_sim = torch.matmul(feat, torch.t(feat)) / (torch.matmul(energy, torch.t(energy)) )
            affinity = cos_sim
        else:
            feat = torch.matmul(feat, torch.t(feat))  # (batch_size, batch_size)
            feat_diag = torch.diag(feat).view(-1, 1).repeat(1, feat.size(0))  # (batch_size, batch_size)
            affinity = 1-torch.exp(-(feat_diag + torch.t(feat_diag) - 2 * feat)/feat.size(1))
        return affinity
    
    def L(self,affinity_stu,affinity_tea,T):
        stu_1 = Variable(torch.ones(affinity_stu.size(0),1).cuda())
        tea_1 = Variable(torch.ones(affinity_tea.size(0),1).cuda())
        p=T.mm(tea_1)
        q=T.t().mm(stu_1)
        if self.l_type == 'L2':
            # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
            # cost_st = f1(affinity_stu)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(affinity_tea)^T
            # cost = cost_st - h1(affinity_stu)*T*h2(affinity_tea)^T
            f1_st = (affinity_stu ** 2).mm(p).mm(tea_1.t())  
            f2_st = stu_1.mm(q.t()).mm((affinity_tea ** 2).t())
            cost_st = f1_st + f2_st
            cost = cost_st - 2 * affinity_stu.mm(T).mm(affinity_tea.t())
        elif self.l_type=='KL':
            # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
            # cost_st = f1(affinity_stu)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(affinity_tea)^T
            # cost = cost_st - h1(affinity_stu)*T*h2(affinity_tea)^T
            f1_st = torch.matmul(affinity_stu * torch.log(affinity_stu+ 1e-7) - affinity_stu, p).mm(tea_1.t())
            f2_st = stu_1.mm(torch.matmul(torch.t(q), torch.t(affinity_tea)))
            cost_st = f1_st + f2_st
            cost = cost_st - torch.matmul(torch.matmul(affinity_stu, T), torch.t(torch.log(affinity_tea+1e-7)))
        return cost