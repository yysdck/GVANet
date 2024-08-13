from .P_CAM import *
import torch.nn as nn
from .SppCSPC import SppCSPC
from .coordatt import CoordAtt
class MVFAB(nn.Module):
    def __init__(self, indim):
        super().__init__()
        self.preattention = CoordAtt(inp=indim,oup=indim)
        self.multivies_expansion = SppCSPC(ch_in=indim,ch_out=indim)

        self.pamodel = PAM_Module(in_dim=indim)
        self.camodel = CAM_Module(in_dim=indim)
    def forward(self, x):
        res = self.preattention(x)
        res = self.SppCSPC(res)
        res_pa = self.pamodel(res)
        res_ca = self.camodel(res)
        res_copy = res_ca + res_pa
        res = res + res_copy



        return res