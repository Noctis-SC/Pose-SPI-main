import torch.nn.functional
import torch.nn as nn
import torch.nn.functional

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
        self.weight = torch.nn.Parameter(data=torch.Tensor(1, 1, 64, 64), requires_grad=True) # (1, 1, 64*64) Should be the same as the input shape
        self.weight.data.uniform_(0.1, 1)
        # for 1dim input Conv2d -> Conv1d
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=64, padding=32) #to remove resize operation add kernel size 65
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
        # Deconvolution layer. #Upsampling can be replaced by pixelshuffle
        self.deconv = torch.nn.UpsamplingBilinear2d(size=64)
        self.expansion = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=3, padding=1)


    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.mapping(x)
        x = self.deconv(x)
        x = self.expansion(x)
        return x

# Define a simple CNN architecture
class SimpleCNN(nn.Module):
    """Simple model for keypoint detection based on 1 channel feature map"""
    def __init__(self, num_keypoints):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_keypoints * 3)  # (x, y, visibility) for each keypoint
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class KeypointDetection(nn.Module):
    """Main model combining encoder, upsample and keypoint detector"""
    def __init__(self, num_keypoints, product=True, reduce=False):
        super(KeypointDetection, self).__init__()
        self.encoder = Encoder(product=product, reduce=reduce)
        self.frcnn = FSRCNN()
        self.decoder = SimpleCNN(num_keypoints=num_keypoints)

    def forward(self, x):
        x_seg = self.encoder(x)
        x_up = self.frcnn(x_seg)
        x = self.decoder(x_up)

        return x