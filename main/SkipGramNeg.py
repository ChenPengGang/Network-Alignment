import torch
import torch.nn as nn
import copy

#定义模型
class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        #定义词向量层
        self.self_embed=nn.Embedding(n_vocab, n_embed)
        self.in_trainsform=nn.Linear(n_embed,n_embed)
        self.out_trainsform=nn.Linear(n_embed,n_embed)
        #词向量层参数初始化
        self.self_embed.weight.data.uniform_(-1,1)
        self.in_trainsform.weight.data.uniform_(0,0.3)
        self.out_trainsform.weight.data.uniform_(0,0.3)
    #输入词的前向过程
    def forward_in(self,self_words):
        self_vector=self.in_trainsform(self.self_embed(self_words))
        return self_vector
    def forward_out(self,self_words):
        self_vector = self.out_trainsform(self.self_embed(self_words))
        return self_vector
    #负样本词的前向过程
    def forward_noise(self, size, N_SAMPLES,device,noise_dist):
        noise_dist = noise_dist.to(device)
        #从词汇分布中采样负样本
        noise_words = torch.multinomial(noise_dist,
                                        size * N_SAMPLES,
                                        replacement=True)
        noise_vectors = self.out_trainsform(self.self_embed(noise_words)).view(size, N_SAMPLES, self.n_embed)

        return noise_vectors