import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

pretrained_model = 'tf_efficientnet_b2_ns'
latent_size = 256

class Encoder(nn.Module):
    def __init__(self, latent_size =  latent_size , pretrained = True , num_channels = 1408 , dropout_rate = 0.5):
        super().__init__()
        self.backbone = timm.create_model(pretrained_model, pretrained=pretrained, num_classes=0,global_pool='',in_chans=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.prelu = nn.PReLU()
        self.lin = nn.Linear( num_channels, latent_size)
        self.do = nn.Dropout(p= dropout_rate)
        
    def forward(self,image):
        image = self.backbone(image)     
        image = self.pool(image)
        
        image = image.view(image.shape[0], -1)    
        image = self.do(image)
        image = self.prelu(self.lin(image))
        return image

class Classifier(nn.Module):
    def __init__(self,latent_size =  latent_size):
        super().__init__()
        self.lin = nn.Linear(  latent_size,1)
    
    def forward(self,image):        
        image = self.lin(image)
        return image

class Discriminator(nn.Module):
    def __init__(self,latent_size = latent_size):
        super().__init__()
        self.discrim = nn.Sequential(
            nn.Linear(latent_size, latent_size // 2),
            nn.PReLU(),
            nn.Linear(latent_size//2, latent_size//2),
            nn.PReLU(),
            nn.Linear(latent_size/2, 1)
        )
    
    def forward(self, x):
        x = self.discrim(x)
        return x