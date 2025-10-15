import torch
# import lpips
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# MSE Loss for train and val, for image super-resolution
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, inputs, targets):
        return F.mse_loss(inputs, targets)

class PerceptualLoss(nn.Module):
    def __init__(self, net='vgg'):
        """
        初始化Perceptual Loss类。

        参数:
            net (str): 使用的网络，可以是'vgg'或'alex'。
        """
        super(PerceptualLoss, self).__init__()
        self.lpips_loss = lpips.LPIPS(net=net)

    def compute_loss_for_axis(self, input_volume, target_volume, axis):
        total_loss = 0.0
        num_slices = input_volume.size(axis)

        for i in range(num_slices):
            if axis == 2:  # 对应原始输入的深度维度
                input_slice = input_volume[:, :, i, :, :]
                target_slice = target_volume[:, :, i, :, :]
            elif axis == 3:  # 对应原始输入的高度维度
                input_slice = input_volume[:, :, :, i, :]
                target_slice = target_volume[:, :, :, i, :]
            elif axis == 4:  # 对应原始输入的宽度维度
                input_slice = input_volume[:, :, :, :, i]
                target_slice = target_volume[:, :, :, :, i]

            # 复制通道以模拟三通道图像
            input_slice = input_slice.repeat(1, 3, 1, 1, 1)[:, :, 0, :, :]
            target_slice = target_slice.repeat(1, 3, 1, 1, 1)[:, :, 0, :, :]

            # 调整为适合lpips输入的形状
            input_slice = input_slice.view(-1, 3, *input_slice.shape[2:])
            target_slice = target_slice.view(-1, 3, *target_slice.shape[2:])

            # 计算感知损失
            loss = self.lpips_loss(input_slice, target_slice)
            total_loss += loss.mean()

        return total_loss / num_slices

    def forward(self, input_volume, target_volume):
        # 计算三个轴向的感知损失，并计算总和
        loss_x = self.compute_loss_for_axis(input_volume, target_volume, 2)
        # loss_y = self.compute_loss_for_axis(input_volume, target_volume, 3)
        # loss_z = self.compute_loss_for_axis(input_volume, target_volume, 4)
        
        # 返回三个轴向损失的平均值
        # total_loss = (loss_x + loss_y + loss_z) / 3
        # total_loss = (loss_x + loss_y) / 2
        total_loss = loss_x
        return total_loss

'''
# PSNR for val, for image super-resolution
class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, inputs, targets, max_pixel=1.0):
        mse = F.mse_loss(inputs, targets)
        return 20 * torch.log10(max_pixel / torch.sqrt(mse))
'''

# Define a loss function, here is an example of a combination of Dice loss and Cross Entropy
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.dice_loss(inputs, targets) + self.ce_loss(inputs, targets)

# Perceptual Loss (Pseudo code)
# class PerceptualLoss(nn.Module):
#     def __init__(self):
#         super(PerceptualLoss, self).__init__()
#         # Load the VGG model pre-trained on ImageNet
#         self.vgg = torchvision.models.vgg16(pretrained=True).features
#         # Freeze VGG parameters
#         for param in self.vgg.parameters():
#             param.requires_grad = False
# 
#     def forward(self, inputs, targets):
#         # Extract features from both the inputs and the targets
#         features_input = self.vgg(inputs)
#         features_target = self.vgg(targets)
#         # Calculate the loss as the L2 norm between feature representations
#         return F.mse_loss(features_input, features_target)

# You can combine MSE and Perceptual Loss
# class CombinedSuperResolutionLoss(nn.Module):
#     def __init__(self):
#         super(CombinedSuperResolutionLoss, self).__init__()
#         self.mse_loss = MSELoss()
#         self.perceptual_loss = PerceptualLoss()
# 
#     def forward(self, inputs, targets):
#         # You can also add a weighting factor to balance the two losses
#         return self.mse_loss(inputs, targets) + self.perceptual_loss(inputs, targets)

# Example of usage:
# Initialize the loss function
mse_loss_func = MSELoss()
# perceptual_loss_func = PerceptualLoss() # Uncomment this when using PerceptualLoss
# combined_loss_func = CombinedSuperResolutionLoss() # Uncomment this when using Combined Loss

# Assuming the model output is 'outputs' and the true high-resolution images are 'high_res_targets'
# Calculate the loss
# mse_loss = mse_loss_func(outputs, high_res_targets)
# print(mse_loss)

# Assuming the model output is 'outputs' and the true labels are 'labels'
# Initialize the loss function
loss_func = CombinedLoss()

# Calculate the loss
# loss = loss_func(outputs, labels)
# print(loss)

# For perceptual loss, first the VGG model should be loaded and the code should be executed in an environment
# where the pretrained models are available.
# perceptual_loss = perceptual_loss_func(outputs, high_res_targets) # Uncomment this when using PerceptualLoss
# print(perceptual_loss)

# If using combined loss
# combined_loss = combined_loss_func(outputs, high_res_targets) # Uncomment this when using Combined Loss
# print(combined_loss)

