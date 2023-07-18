import torch
import torch.nn as nn

#Building Block of CNN which will be repeatedly used
class CNN_Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(CNN_Block,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=stride,bias=False,padding_mode="reflect"),#padding=1 can be added
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.conv(x)    

#Input Image, Output Image both are sent to the Discriminator by concatenating them along the channels
class Discriminator(nn.Module):
    def __init__(self,in_channels=3,features=[64,128,256,512]):
        super(Discriminator,self).__init__()
        self.initial=nn.Sequential(
            nn.Conv2d(in_channels*2,features[0],kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )        

        layers=[]
        in_channels=features[0]
        for feature in features[1:]:
            layers.append(CNN_Block(in_channels,feature,stride=1 if feature==features[-1] else 2))
            in_channels=feature

        layers.append(nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode="reflect"))
        
        self.model=nn.Sequential(*layers)


    def forward(self,x,y):
        x=torch.cat([x,y],dim=1)#concatenation across the channels
        x=self.initial(x)
        return self.model(x)


def test():
    x=torch.randn((1,3,256,256))    
    y=torch.randn((1,3,256,256))
    model=Discriminator(in_channels=3)
    pred=model(x,y)
    print(model)
    print(pred.shape)

if __name__=='__main__':
    test()