import torch
import torch.nn as nn

#定义损失函数
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,self_vectors1,self_vectors2,noise_vectors):
        BATCH_SIZE, embed_size = self_vectors1.shape
        #将输入词向量与目标词向量作维度转化处理
        self_vectors1=self_vectors1.view(BATCH_SIZE,embed_size,1)
        self_vectors2=self_vectors2.view(BATCH_SIZE,1, embed_size)
        #目标词损失
        out_loss = torch.bmm(self_vectors2, self_vectors1).sigmoid().log()
        out_loss = out_loss.squeeze()
        #负样本损失
        noise_loss = torch.bmm(noise_vectors.neg(), self_vectors1).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)
        #综合计算两类损失
        return -(out_loss + noise_loss).mean()
        #return -out_loss.mean()