"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

# TODO: Choose from either model and uncomment that line
# class KeypointModel(nn.Module):
class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        NOTE: You could either choose between pytorch or pytorch lightning, 
            by switching the class name line.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        self.loss_fn = nn.MSELoss()
#         self.dropout_p = 0.5
        modules=[]
        for _ in range(4):
            modules.append(nn.Conv2d(32,32,3,padding=1))#N*32*96*96
            modules.append(nn.BatchNorm2d(32))
            modules.append(nn.LeakyReLU())
            modules.append(nn.MaxPool2d(2,2))#N*32
#             modules.append(nn.Dropout(self.dropout_p))
        self.model = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),#N*32*96*96
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),#N*32*48*48
#             nn.Dropout(self.dropout_p),
            *modules,#48/(2**4)
            nn.Flatten(),
            nn.Linear(32*3*3,500),
            nn.ReLU(),
            nn.Linear(500,200),
            nn.ReLU(),
            nn.Linear(200,30)
            
        )
    
     
#     def general_step(self, batch, batch_idx, mode):
#         images, keypoints = batch['image'], batch['keypoints']
#         pred_keypoints = torch.squeeze(self.forward(images)).view(batch['image'].size(0),15,2) 
#         loss = self.loss_fn(pred_keypoints, keypoints)
#         return loss / batch['image'].size(0)
#     def training_step(self, batch, batch_idx):
#         loss = self.general_step(batch, batch_idx, "train")
#         return {'loss':loss}

#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.hparams['batch_size'])

#     def validation_step(self, batch, batch_idx):
#         loss = self.general_step(batch, batch_idx, "val")
#         return {'val_loss':loss}

#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams['batch_size'])

#     def configure_optimizers(self):
#         LR = 1e-3
#         optim = torch.optim.Adam(self.model.parameters(), lr=LR)
#         return optim
#         pass
#     torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
#     torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
#     torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################
        x =self.model(x)
#         x = x.view(-1,30)

#         pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
