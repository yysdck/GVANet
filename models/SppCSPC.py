import torch
import torch.nn as nn
class Bconv(nn.Module):
    def __init__(self,ch_in,ch_out,k,s):
        '''
        :param ch_in: 输入通道数
        :param ch_out: 输出通道数
        :param k: 卷积核尺寸
        :param s: 步长
        :return:
        '''
        super(Bconv, self).__init__()
        self.conv=nn.Conv2d(ch_in,ch_out,k,s,padding=k//2)
        self.bn=nn.BatchNorm2d(ch_out)
        self.act=nn.SiLU()
    def forward(self,x):
        '''
        :param x: 输入
        :return:
        '''
        return self.act(self.bn(self.conv(x)))
class SppCSPC(nn.Module):
    def __init__(self,ch_in,ch_out):
        '''
        :param ch_in: 输入通道
        :param ch_out: 输出通道
        '''
        super(SppCSPC, self).__init__()
        #分支一
        self.conv1=nn.Sequential(
            Bconv(ch_in,ch_out,1,1),
            Bconv(ch_out,ch_out,3,1),
            Bconv(ch_out,ch_out,1,1)
        )
        #分支二（SPP）
        self.mp1=nn.MaxPool2d(5,1,5//2) #卷积核为5的池化
        self.mp2=nn.MaxPool2d(9,1,9//2) #卷积核为9的池化
        self.mp3=nn.MaxPool2d(13,1,13//2) #卷积核为13的池化

        #concat之后的卷积
        self.conv1_2=nn.Sequential(
            Bconv(4*ch_out,ch_out,1,1),
            Bconv(ch_out,ch_out,3,1)
        )


        #分支三
        self.conv3=Bconv(ch_in,ch_out,1,1)

        #此模块最后一层卷积
        self.conv4=Bconv(2*ch_out,ch_out,1,1)
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(ch_in,2*ch_in),
            torch.nn.ReLU(),
            torch.nn.Linear(2*ch_in,ch_in),
            torch.nn.ReLU()
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(ch_in, 2 * ch_in),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * ch_in, ch_in),
            torch.nn.ReLU()
        )
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(ch_in, 2 * ch_in),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * ch_in, ch_in),
            torch.nn.ReLU()
        )
    def forward(self,x):
        #分支一输出
        output1=self.conv1(x)


        mp_output1=self.mp1(output1)
        mp_output1=self.mlp1(mp_output1.permute(0,2,3,1))
        mp_output1=mp_output1.permute(0,3,1,2)
        mp_output2=self.mp2(output1)
        mp_output2 = self.mlp2(mp_output2.permute(0, 2, 3, 1))
        mp_output2 = mp_output2.permute(0, 3, 1, 2)
        mp_output3=self.mp3(output1)
        mp_output3 = self.mlp3(mp_output3.permute(0, 2, 3, 1))
        mp_output3 = mp_output3.permute(0, 3, 1, 2)

        #合并以上并进行卷积
        result1=self.conv1_2(torch.cat((output1,mp_output1,mp_output2,mp_output3),dim=1))

        #分支三
        result2=self.conv3(x)

        return self.conv4(torch.cat((result1,result2),dim=1))
