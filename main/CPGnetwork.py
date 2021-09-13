import random
import torch
import numpy as np
from torch.autograd import Variable
#from numba import jit
from collections import defaultdict,ChainMap
import torch.nn.functional as fun
from copy import copy

def progress1(percent, width=50):
    '''进度打印功能'''
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
    print('\r%s %d%%' % (show_str, percent), end='')

class CPG_Network:
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.nodes=self.G.nodes()
        self.edges=self.G.edges()
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.vocab_prepare()


    def get_edges(self,num):
        return random.sample(self.edges,num)

    def get_nodes(self,num):
        return random.sample(self.nodes,num)

    def get_frepuency4degree(self):
        frequency_list=[]
        for node in self.nodes:
            frequency_list+=[node]*self.G.degree(node)
        return frequency_list

    def vocab_prepare(self):
        # 构建词典
        self.vocab2int = {w: c for c, w in enumerate(self.G.nodes())}
        self.int2vocab = {c: w for c, w in enumerate(self.G.nodes())}




