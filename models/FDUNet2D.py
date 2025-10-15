import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        # For the batch normalization, the momentum parameter was set to 0.99, and the epsilon was 0.001
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.001)
        # self.silu = nn.SiLU()
        # self.elu = nn.ELU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.3) if dropout else None

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

# Define the Dense Block
class DenseBlock2D(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock2D, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size=1)
        self.conv2 = ConvBlock(in_channels, in_channels // 4, kernel_size=3, dropout=True)
        self.conv3 = ConvBlock(in_channels * 5 // 4, in_channels, kernel_size=1)
        self.conv4 = ConvBlock(in_channels, in_channels // 4, kernel_size=3, dropout=True)
        self.conv5 = ConvBlock(in_channels * 6 // 4, in_channels, kernel_size=1)
        self.conv6 = ConvBlock(in_channels, in_channels // 4, kernel_size=3, dropout=True)
        self.conv7 = ConvBlock(in_channels * 7 // 4, in_channels, kernel_size=1)
        self.conv8 = ConvBlock(in_channels, in_channels // 4, kernel_size=3, dropout=True)
        # self.conv9 = ConvBlock(in_channels * 8 // 4, in_channels, kernel_size=1)

    def forward(self, x):
        # black line: Apply the first convolution
        x1 = self.conv1(x)
        
        # blue line: Apply the second convolution
        x2 = self.conv2(x1)
        
        # orange line: Concatenate the outputs with the input
        x3 = torch.cat([x, x2], 1)

        # black line: Apply the third convolution 
        x4 = self.conv3(x3) 

        # blue line: Apply the fourth convolution
        x5 = self.conv4(x4)

        # orange line: Concatenate the outputs with the input
        x6 = torch.cat([x3, x5], 1)

        # black line: Apply the fifth convolution
        x7 = self.conv5(x6)

        # blue line: Apply the sixth convolution
        x8 = self.conv6(x7)

        # orange line: Concatenate the outputs with the input
        x9 = torch.cat([x6, x8], 1)

        # black line: Apply the seventh convolution
        x10 = self.conv7(x9)

        # blue line: Apply the eighth convolution
        x11 = self.conv8(x10)

        # orange line: Concatenate the outputs with the input
        out = torch.cat([x9, x11], 1)

        return out
    

class TransitionDown2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionDown2D, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2)
        # self.norm = nn.BatchNorm2d(out_channels)
        # self.silu = nn.SiLU()
        # self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2
        # return self.elu(self.norm(self.conv(x)))

class TransitionUp2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp2D, self).__init__()
        self.conv = ConvBlock(in_channels, in_channels, kernel_size=1, stride=1)
        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.001)
        # self.silu = nn.SiLU()
        # self.elu = nn.ELU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv_trans(x1)
        return self.norm(self.relu(x2))
        # return self.elu(self.norm(self.conv_trans(x)))

class FullyDenseUNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(FullyDenseUNet2D, self).__init__()
        
        # first light purple line: 3x3 Conv + ReLU + BN
        self.init_conv = ConvBlock(in_channels, 16, kernel_size=3)
        
        # dark orange line: First Dense Block
        self.dense_block1 = DenseBlock2D(16)
        # green line: First Down Block, transition down
        self.trans_down1 = TransitionDown2D(32, 32)

        # Second dense block and transition down
        self.dense_block2 = DenseBlock2D(32)
        self.trans_down2 = TransitionDown2D(64, 64)

        # Third dense block and transition down
        self.dense_block3 = DenseBlock2D(64)
        self.trans_down3 = TransitionDown2D(128, 128)

        # Fourth dense block and transition down
        self.dense_block4 = DenseBlock2D(128)
        self.trans_down4 = TransitionDown2D(256, 256)

        # Fifth dense block and transition down
        self.dense_block5 = DenseBlock2D(256)
        self.trans_down5 = TransitionDown2D(512, 512)

        # Sixth dense block
        self.dense_block6 = DenseBlock2D(512)

        # Transition up and seventh dense block
        self.trans_up1 = TransitionUp2D(1024, 512)
        # first black line: 1x1x1 Conv + ReLU + BN
        self.conv1 = ConvBlock(1024, 256, kernel_size=1, stride=1)
        self.dense_block7 = DenseBlock2D(256) # Concatenated with skip connection

        # Transition up and eighth dense block
        self.trans_up2 = TransitionUp2D(512, 256)
        # Second black line: 1x1x1 Conv + ReLU + BN
        self.conv2 = ConvBlock(512, 128, kernel_size=1, stride=1)
        self.dense_block8 = DenseBlock2D(128) # Concatenated with skip connection

        # Transition up and ninth dense block
        self.trans_up3 = TransitionUp2D(256, 128)
        # Third black line: 1x1x1 Conv + ReLU + BN
        self.conv3 = ConvBlock(256, 64, kernel_size=1, stride=1)
        self.dense_block9 = DenseBlock2D(64) # Concatenated with skip connection

        # Transition up and tenth dense block
        self.trans_up4 = TransitionUp2D(128, 64)
        # Fourth black line: 1x1x1 Conv + ReLU + BN
        self.conv4 = ConvBlock(128, 32, kernel_size=1, stride=1)
        self.dense_block10 = DenseBlock2D(32) # Concatenated with skip connection

        # Transition up and eleventh dense block
        self.trans_up5 = TransitionUp2D(64, 32)
        # Fifth black line: 1x1x1 Conv + ReLU + BN
        self.conv5 = ConvBlock(64, 16, kernel_size=1, stride=1)
        self.dense_block11 = DenseBlock2D(16) # Concatenated with skip connection

        # Final output layer, in the final step, a 1 × 1 convolution followed by batch normalization reduces the channel depth from 32 filters to 1 output filter
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.001)
    
    def forward(self, x):
        # Initial convolution
        x_init = self.init_conv(x)

        # Encoder path
        out_dense1 = self.dense_block1(x_init)
        out_trans1 = self.trans_down1(out_dense1)

        out_dense2 = self.dense_block2(out_trans1)
        out_trans2 = self.trans_down2(out_dense2)

        out_dense3 = self.dense_block3(out_trans2)
        out_trans3 = self.trans_down3(out_dense3)

        out_dense4 = self.dense_block4(out_trans3)
        out_trans4 = self.trans_down4(out_dense4)

        out_dense5 = self.dense_block5(out_trans4)
        out_trans5 = self.trans_down5(out_dense5)

        out_dense6 = self.dense_block6(out_trans5)

        # Decoder path
        # import pdb
        # pdb.set_trace()
        out_up1 = self.trans_up1(out_dense6)
        out_up1 = torch.cat((out_up1, out_dense5), dim=1)
        out_conv1 = self.conv1(out_up1)
        out_dense7 = self.dense_block7(out_conv1)

        out_up2 = self.trans_up2(out_dense7)
        out_up2 = torch.cat((out_up2, out_dense4), dim=1)
        out_conv2 = self.conv2(out_up2)
        out_dense8 = self.dense_block8(out_conv2)

        out_up3 = self.trans_up3(out_dense8)
        out_up3 = torch.cat((out_up3, out_dense3), dim=1)
        out_conv3 = self.conv3(out_up3)
        out_dense9 = self.dense_block9(out_conv3)

        out_up4 = self.trans_up4(out_dense9)
        out_up4 = torch.cat((out_up4, out_dense2), dim=1)
        out_conv4 = self.conv4(out_up4)
        out_dense10 = self.dense_block10(out_conv4)

        out_up5 = self.trans_up5(out_dense10)
        out_up5 = torch.cat((out_up5, out_dense1), dim=1)
        out_conv5 = self.conv5(out_up5)
        out_dense11 = self.dense_block11(out_conv5)

        # Final output
        out = self.norm(self.final_conv(out_dense11))
        
        return x + out


# Example of creating a model instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FullyDenseUNet2D().to(device)

# 使用DataParallel来使用多GPU
# model = nn.DataParallel(model)
# model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

model.to("cuda:0")

# input_tensor = torch.randn(32, 1, 128, 128, 128).to("cuda:0") 不是必须
input_tensor = torch.randn(8, 1, 256, 256).to(device)

# Forward pass through the model
output = model(input_tensor)
# print(output.shape)  # 输出: torch.Size([32, 1, 128, 128, 128])
