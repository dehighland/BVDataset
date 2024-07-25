import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import timm
from ncps.torch import LTC
from ncps.wirings import AutoNCP

### Model Classes ###  
class Mobile(nn.Module):

    def __init__(self):
        
        super(Mobile, self).__init__()
        
        # Model
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.fc = nn.Sequential(
                                nn.ReLU(),
                                nn.Linear(1000,1),
                                nn.Sigmoid()
                                )
        
        #Xavier Initialization
        xavier_init(self)
        
    def forward(self, x):
        result = self.model(x)
        result = self.fc(result)
        
        return(result)

class VGG16(nn.Module):

    def __init__(self):
        
        super(VGG16, self).__init__()
        
        # Model
        self.model = torchvision.models.vgg16(weights='DEFAULT')
        self.fc = nn.Sequential(
                                nn.ReLU(),
                                nn.Linear(1000,1),
                                nn.Sigmoid()
                                )
        
        #Xavier Initialization
        xavier_init(self)
        
    def forward(self, x):
        result = self.model(x)
        result = self.fc(result)
        
        return(result)
 
class Xception(nn.Module):

    def __init__(self):
        
        super(Xception, self).__init__()
        
        # Model
        self.model = timm.create_model('xception', pretrained=True, num_classes=1)
        self.end = nn.Sigmoid()
        
        #Xavier Initialization
        xavier_init(self)
        
    def forward(self, x):
        result = self.model(x)
        result = self.end(result)
        
        return(result)


class ResNet18Model(nn.Module):
    '''Image model used to either identify images as diagnostic or non-diagnostic for BV or to identify whether or not
        an image is a clue cell'''
        
    def __init__(self):
        
        super(ResNet18Model, self).__init__()

        # ResNet Backbone
        self.backbone = torchvision.models.resnet18(weights='DEFAULT')
        
        # Output Layer                                  
        self.fc = nn.Sequential(
                                nn.ReLU(),
                                nn.Linear(1000,1),
                                nn.Sigmoid()
                                )
                                
        #Xavier Initialization
        xavier_init(self)
                                    
    def forward(self,x):
        result = self.backbone(x)
        result = self.fc(result)
        
        return(result)  
        
class ResNet34Model(nn.Module):
    '''Image model used to either identify images as diagnostic or non-diagnostic for BV or to identify whether or not
        an image is a clue cell'''
        
    def __init__(self):
        
        super(ResNet34Model, self).__init__()

        # ResNet Backbone
        self.backbone = torchvision.models.resnet34(weights='DEFAULT')
        
        # Output Layer                                  
        self.fc = nn.Sequential(
                                nn.ReLU(),
                                nn.Linear(1000,1),
                                nn.Sigmoid()
                                )
                                
        #Xavier Initialization
        xavier_init(self)
                                    
    def forward(self,x):
        result = self.backbone(x)
        result = self.fc(result)
        
        return(result) 
        
class CCD2MDMLP(nn.Module):
    '''Basic MLP used to get a baseline for the diagnostic ability of non-image data such as pH and binary whiff test results'''

    def __init__(self):
        
        super(CCD2MDMLP, self).__init__()
                                
        self.fc = nn.Sequential(
                                nn.Linear(1, 2),
                                nn.BatchNorm1d(2),
                                nn.ReLU(),
                                
                                nn.Linear(2, 1),
                                nn.Sigmoid()
                                )
                                
        #Xavier Initialization
        xavier_init(self)
                                    
    def forward(self,x):
        result = self.fc(x)
        
        return(result)
        
class NonImageMLP(nn.Module):
    '''Basic MLP used to get a baseline for the diagnostic ability of non-image data such as pH and binary whiff test results'''

    def __init__(self):
        
        super(NonImageMLP, self).__init__()
                                
        self.fc = nn.Sequential(
                                nn.Linear(1, 4),
                                nn.BatchNorm1d(4),
                                nn.ReLU(),
                                
                                nn.Linear(4, 1),
                                nn.Sigmoid()
                                )
                                
        #Xavier Initialization
        xavier_init(self)
                                    
    def forward(self,x):
        result = self.fc(x)
        
        return(result)

class CombinerMLP(nn.Module):
    '''Basic MLP used to mix two inputs (one of which won't be an image model result) into one diagnostic output. Inputs are
        meant to be binary whiff test data, pH values, or the final result from a pretrained image model'''
        
    def __init__(self):
        
        super(CombinerMLP, self).__init__()
       
        # Merge Layer
        self.fc_merge = nn.Sequential(
                                nn.Linear(2,4),
                                nn.BatchNorm1d(4),
                                nn.ReLU(),
                                
                                nn.Linear(4,1),
                                nn.Sigmoid()
                                )
                                
        #Xavier Initialization
        xavier_init(self)
                                    
    def forward(self,x,y):
        z = torch.cat((x,y), 1)
        result = self.fc_merge(z)
        
        return(result)

class DoubleCombinerMLPB(nn.Module):
    def __init__(self):
        '''Slightly complicated MLP where pH and binary whiff test values are input as y and z and mixed together with a small
            MLP before being combined with final results from a pretrained image model input as x'''
        
        super(DoubleCombinerMLPB, self).__init__()
       
        # Merge Layers
        self.fc_pHWhiff = nn.Sequential(
                                nn.Linear(2,2),
                                nn.BatchNorm1d(2),
                                nn.ReLU(),
                                
                                nn.Linear(2,1),
                                nn.ReLU(),
                                )
                                
        self.fc_all = nn.Sequential(
                                nn.Linear(2,4),
                                nn.BatchNorm1d(4),
                                nn.ReLU(),
                                
                                nn.Linear(4,1),
                                nn.Sigmoid()
                                )
                                
        #Xavier Initialization
        xavier_init(self)
                                    
    def forward(self,x,y,z):
        non_img = torch.cat((y,z), 1)
        non_img = self.fc_pHWhiff(non_img)
        
        result = torch.cat((x,non_img), 1)
        result = self.fc_all(result)
        
        return(result) 

class DoubleCombinerMLPA(nn.Module):
    def __init__(self):
        '''Slightly complicated MLP where pH and binary whiff test values are input as y and z and mixed together with a small
            MLP before being combined with final results from a pretrained image model input as x'''
        
        super(DoubleCombinerMLPA, self).__init__()
       
        # Merge Layers
        self.fc_all = nn.Sequential(
                                nn.Linear(3,4),
                                nn.BatchNorm1d(4),
                                nn.ReLU(),
                                
                                nn.Linear(4,1),
                                nn.Sigmoid()
                                )
                                
        #Xavier Initialization
        xavier_init(self)
                                    
    def forward(self,x,y,z):
        result = torch.cat((x,y,z), 1)
        result = self.fc_all(result)
        
        return(result)        

def xavier_init(self):
    for m in self.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
