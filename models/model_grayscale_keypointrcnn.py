import torch.nn as nn
import torch.nn.functional
from models.efficentnet import EfficientNet

class CustomConv(nn.Module):
    """
    Custom Convolutional layer to mimic Hadamart product:
    1st option is to multiply learnable weights with the image
    2nd option apply convolutional layer with padding (to preserve the size) and
    apply size interpolation for the cases where exact size of the input can not be achieved due to the kernel size.
    arguments:
    product - bool, whether to perform multiplication or not.
                If TRUE - performs multiplication, if FALSE - performs convolution
    """
    def __init__(self, product=True):
        super(CustomConv, self).__init__()
        self.product = product
        self.weight = torch.nn.Parameter(data=torch.Tensor(1, 1, 64, 64), requires_grad=True)
        self.weight.data.uniform_(0.1, 1)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=64, padding=32)
        self.resize = nn.UpsamplingBilinear2d(size=64)

    def forward(self, x):
        if self.product:
            out = self.weight.mul(x)
        else:
            out = self.conv(x)
            out = self.resize(out)
        return out

class Encoder(nn.Module):
    def __init__(self, product=False, reduce=False):
        super(Encoder, self).__init__()
        self.reduce = reduce
        self.conv = CustomConv(product=product)
        """Max pooling can be used to imitate presence of fewer measurement in real experiment"""
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        """2 fully connected layers that will be used depending on input feature (i.e. measurements) count """
        self.fc_reduced = nn.Linear(in_features=1024, out_features=1024)
        self.fc = nn.Linear(in_features=4096, out_features=1024)

    def forward(self, x):
        # Input shape: (batch_size, 1, 64, 64)
        x = self.conv(x)
        # whether to reduce measurements number or no
        if self.reduce:
            x = self.max_pool(x)
            # Output shape: (batch_size, 1, 32, 32)
            x = x.view(x.size(0), -1)  # Flatten the tensor
            # Output shape: (batch_size, 1*32*32)
            x = self.fc_reduced(x)
            # Output shape: (batch_size, 1024)
        else:
            # Output shape: (batch_size, 1, 64, 64)
            x = x.view(x.size(0), -1)  # Flatten the tensor
            # Output shape: (batch_size, 1*64*64)
            x = self.fc(x)
            # Output shape: (batch_size, 1024)
        x = x.view(x.size(0), 1, 32, 32)
        # Output shape: (batch_size, 1, 32, 32)
        return x


class FSRCNN(nn.Module):
    def __init__(self):
        """FSRCNN model as described in the corresponding paper, with the difference in the upsampling layer.
        Original paper uses transposed convolutional layer that in the modern literature is known to be responsible for
        checkboard patterns in the results. Instead more recent appraoch of resize+convolution is adopted."""
        super(FSRCNN, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=56, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(56),
            nn.ReLU()
        )
        self.mapping = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU()
        )
        # Deconvolution layer.
        self.deconv = torch.nn.UpsamplingBilinear2d(size=64)
        self.expansion = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=3, padding=1)


    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.mapping(x)
        x = self.deconv(x)
        x = self.expansion(x)
        return x


class KeypointRCNNLoss(nn.Module):
    """Keypointrcnn specific loss function that include regression of the keypoints with the L1 distance
    or with the smoothed l1 to not consider outliers."""
    def __init__(self, num_keypoints, smoothl1=False):
        super(KeypointRCNNLoss, self).__init__()
        self.num_keypoints = num_keypoints
        self.smoothl1 = smoothl1

    def forward(self, keypoints_regression, targets):
        """
        Compute the keypoint detection loss.

        Args:
            keypoints_regression (Tensor): Predicted keypoint coordinates of shape (N, 3 * num_keypoints).
            targets (Tensor): Ground truth keypoints of shape (N, num_keypoints * 3).

        Returns:
            Tensor: Total keypoint detection loss.
        """
        num_keypoints = self.num_keypoints

        if self.smoothl1:
            # Compute the keypoint regression loss (using Smooth L1 loss)
            loss_reg = nn.functional.smooth_l1_loss(keypoints_regression, targets, reduction='mean')
        else:
            loss_reg = nn.functional.l1_loss(keypoints_regression, targets, reduction='mean')

        total_loss = loss_reg

        return total_loss

class KeypointPredictor(nn.Module):
    """Keypoint prediction head from the RCNN model"""
    def __init__(self, in_channels, num_keypoints, input_size=4):
        super(KeypointPredictor, self).__init__()
        self.keypoint_regression = nn.Conv2d(in_channels, out_channels=8, kernel_size=1)
        self.fc_layers = nn.Linear(8 * input_size * input_size, num_keypoints * 3)


    def forward(self, x):
        keypoints_regression = self.keypoint_regression(x)
        keypoints_regression = keypoints_regression.view(keypoints_regression.size(0), -1)
        keypoints_regression = self.fc_layers(keypoints_regression)
        return keypoints_regression


class KeypointRCNN(nn.Module):
    """Model for keypoint detection from grayscale human images

        Args:
            last_layer_size (int): size of the feature map (height or width, where height == width) to pass to fully-connected layer
                                    in keypointrcnn head.
            num_keypoints (int): number of keypoints.

        """
    def __init__(self, num_keypoints, last_layer_size=4):
        super(KeypointRCNN, self).__init__()
        """Efficientnet is used as pretrained backbone for feature extraction.
         Depending on the memory/speed needs, different efficientnet variations can be used.
         Each variation is returing tensor with the different number of channels. When changing the version of 
         efficientnet, corresponding number in in_channels should be changed. The number of channels for each version
         provided bellow."""
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')

        """
        efficientnet-b0 - 112,
        efficientnet-b1 - 112
        efficientnet-b2 - 120
        efficientnet-b3 - 136
        efficientnet-b4 - 160
        efficientnet-b5 - 176
        """
        in_channels = 112
        self.keypoint_head = KeypointPredictor(in_channels, num_keypoints, input_size=last_layer_size)


    def forward(self, images):
        # Backbone forward pass
        endpoints = self.backbone.extract_endpoints(images)
        features = endpoints['reduction_4']

        # Keypoint predictor head forward pass
        keypoints_regression = self.keypoint_head(features)

        return keypoints_regression


class KeypointDetection(nn.Module):
    """Main model that combines encoder, upsampling and keypoint detector."""
    def __init__(self, num_keypoints):
        super(KeypointDetection, self).__init__()
        self.encoder = Encoder()
        self.frcnn = FSRCNN()
        self.transition_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.decoder = KeypointRCNN(num_keypoints=num_keypoints)


    def forward(self, x):
        x_seg = self.encoder(x)
        x_up = self.frcnn(x_seg)
        x_up = torch.nn.functional.sigmoid(x_up)
        x_transition = self.transition_conv(x_up)

        x = self.decoder(x_transition)
        return x