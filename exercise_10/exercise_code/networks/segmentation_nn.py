"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models, transforms

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################
        filters = self.hparams['filters']
        features = models.mobilenet_v3_large(pretrained=True).features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for param in features.parameters():
            param.requires_grad = False
            
            
        self.model = nn.Sequential(
            *list(features.children()),
            #alexnet-> [256,6,6]
            #mobilenet_v2-> [1280, 8, 8]

            # out = (in + 2*pad - kernel)/stride + 1
            
            # 8 -> 15 -> (15+2*1-3)+1=15
            nn.Upsample(scale_factor=15/8),
            nn.Conv2d(filters[0], filters[1], 3, stride=1, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(),
#             nn.Dropout2d(),

            # 15 -> 30 ->(30+2*1-3)+1=30
            nn.Upsample(scale_factor=2),
            nn.Conv2d(filters[1], filters[2], 3, stride=1, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.LeakyReLU(),
#             nn.Dropout2d(),

            # 30 -> 60 ->(60+2*1-3)+1=60
            nn.Upsample(scale_factor=2),
            nn.Conv2d(filters[2], filters[3], 3, stride=1, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.LeakyReLU(),
#             nn.Dropout2d(),

            # out = (in-1)*stride - 2*pad + kernel + output_padding
            # 60 -> (60-1)*2-2+3+1=120
            nn.ConvTranspose2d(filters[3], filters[4], kernel_size=3,stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(filters[4]),
            nn.LeakyReLU(),
#             nn.Dropout2d(),

            # 120 -> (120-1)*2-2+3+1=240
            nn.ConvTranspose2d(filters[4], num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Dropout2d()
#         pass
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(x.device)
        x=x.to(device)
#         print(next(self.model.parameters()).device)
        self.model = self.model.to(device)
        #print(next(self.model.parameters()).device)
        x = self.model(x)
#         pass

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
