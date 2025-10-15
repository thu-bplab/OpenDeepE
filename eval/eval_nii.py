import torch
import os
import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from math import exp

def normalize_data(data):
   min_val = torch.min(data)
   max_val = torch.max(data)
   normalized_data = (data - min_val) / (max_val - min_val)
   return normalized_data

class RMSE(nn.Module):
   """
   A PyTorch module for computing the Root Mean Square Error (RMSE) between true and predicted values.
   """
    
   def __init__(self):
      """
      Initializes the RMSE module.
      """
      super(RMSE, self).__init__()

   def forward(self, y_true, y_pred):
      """
      Calculate the RMSE between the true and predicted values.
        
      Parameters:
      - y_true: Tensor of shape (n_samples, ...), true values.
      - y_pred: Tensor of shape (n_samples, ...), predicted values.
        
      Returns:
      - rmse_value: Tensor, the RMSE between y_true and y_pred.
      """
      mse = F.mse_loss(y_true, y_pred)
      rmse_value = torch.sqrt(mse)
      return rmse_value

# PSNR for val, for 3D image super-resolution
class PSNR(nn.Module):
   def __init__(self):
      super(PSNR, self).__init__()

   def forward(self, inputs, targets, max_pixel=1.0):
      mse = F.mse_loss(inputs, targets)
      return 20 * torch.log10(max_pixel / torch.sqrt(mse))

# SSIM2D for val, for 3D image super-resolution
class SSIM2D(nn.Module):
   def __init__(self):
      super(SSIM2D, self).__init__()

   def forward(self, inputs, targets, data_range):
      # 确保输入为numpy数组，并且数据类型为float
      inputs_np = inputs.squeeze().detach().cpu().numpy().astype(np.float32)
      targets_np = targets.squeeze().detach().cpu().numpy().astype(np.float32)
      # 计算SSIM
      ssim_value = ssim(inputs_np, targets_np, data_range=data_range)
      return ssim_value

class Evaluator:
   def __init__(self, model, criterion_mse, criterion_psnr, criterion_ssim, save_dir='saved_images'):
      self.model = model
      self.criterion_mse = criterion_mse
      self.criterion_psnr = criterion_psnr
      self.criterion_ssim = criterion_ssim
      self.save_dir = os.path.join(save_dir, 'sample_images')
      os.makedirs(self.save_dir, exist_ok=True)

   def evaluate(self, dataloader, batch_size, milestone):
      self.model.eval()
      images_saved = 0
      total_loss = 0
      final_total_ssim = 0
      final_total_psnr = 0
      images_to_save = 10  # Number of images to save during evaluation

      with torch.no_grad():
         for batch in dataloader:
            sparse_inputs = batch['input'].cuda().float()
            gt = batch['target'].cuda().float()

            # 使用实际的批次大小调用模型采样函数
            actual_batch_size = sparse_inputs.size(0)  # 获取当前批次的实际大小
            outputs = self.model.sample(batch_size=actual_batch_size, condition_tensors=sparse_inputs) 
            # 数据归一化到 [0, 1]
            outputs_norm = normalize_data(outputs)
            gt_norm = normalize_data(gt)
            mse_loss = self.criterion_mse(outputs_norm, gt_norm)
            total_loss += mse_loss.item()

            total_ssim = 0
            total_psnr = 0
            for i in range(outputs_norm.shape[0]):
               total_psnr += self.criterion_psnr(outputs_norm[i], gt_norm[i]).item()
               total_ssim += self.criterion_ssim(outputs_norm[i], gt_norm[i], data_range=1.0).item()

               # Save images conditionally
               if images_saved < images_to_save:
                     img_filename = os.path.join(self.save_dir, f'sample_image_{milestone}_{i}.png')
                     save_image(outputs[i], img_filename)
                     images_saved += 1

            final_total_ssim += total_ssim / outputs_norm.shape[0]
            final_total_psnr += total_psnr / outputs_norm.shape[0] 

      avg_loss = total_loss / len(dataloader)
      avg_psnr = final_total_psnr / len(dataloader)
      avg_ssim = final_total_ssim / len(dataloader)

      self.model.train()
      return avg_loss, avg_psnr, avg_ssim 


class EvaluatorNifti:
    def __init__(self, model, criterion_mse, criterion_psnr, criterion_ssim, save_dir='saved_images'):
        self.model = model
        self.criterion_mse = criterion_mse
        self.criterion_psnr = criterion_psnr
        self.criterion_ssim = criterion_ssim
        self.save_dir = os.path.join(save_dir, 'sample_images')
        os.makedirs(self.save_dir, exist_ok=True)

    def evaluate(self, dataloader, milestone):
        self.model.eval()
        images_saved = 0
        total_loss = 0
        final_total_ssim = 0
        final_total_psnr = 0
        images_to_save = 2000  # Number of images to save during evaluation

        with torch.no_grad():
            for batch in dataloader:
                sparse_inputs = batch['input'].cuda().float()
                gt = batch['target'].cuda().float()

                # 使用实际的批次大小调用模型采样函数
                actual_batch_size = sparse_inputs.size(0)  # 获取当前批次的实际大小
                outputs = self.model(sparse_inputs)
                
                # 数据归一化到 [0, 1]
                outputs_norm = normalize_data(outputs)
                gt_norm = normalize_data(gt)
                mse_loss = self.criterion_mse(outputs_norm, gt_norm)
                total_loss += mse_loss.item()

                total_ssim = 0
                total_psnr = 0
                for i in range(outputs_norm.shape[0]):
                    total_psnr += self.criterion_psnr(outputs_norm[i], gt_norm[i]).item()
                    total_ssim += self.criterion_ssim(outputs_norm[i], gt_norm[i], data_range=1.0).item()

                    # Save images conditionally
                    if images_saved < images_to_save:
                        img_filename = os.path.join(self.save_dir, f'sample_image_{milestone}_{i}.nii')
                        self.save_nifti_image(outputs[i], img_filename)
                        images_saved += 1

                final_total_ssim += total_ssim / outputs_norm.shape[0]
                final_total_psnr += total_psnr / outputs_norm.shape[0] 

        avg_loss = total_loss / len(dataloader)
        avg_psnr = final_total_psnr / len(dataloader)
        avg_ssim = final_total_ssim / len(dataloader)

        self.model.train()
        return avg_loss, avg_psnr, avg_ssim

    def save_nifti_image(self, image_tensor, filename):
        # 将图像张量转换为 numpy 数组并转换为 float32
        image_np = image_tensor.cpu().numpy().astype(np.float32)
        
        # 创建单位矩阵，元素类型为 float32
        affine = np.eye(4, dtype=np.float32)
        
        # 存储每个切片的数据到 NIfTI 图像对象中
        nii_image = nib.Nifti1Image(image_np, affine)
        
        # 保存 NIfTI 图像
        nib.save(nii_image, filename)


# SSIM3D for val, for 3D image super-resolution
def gaussian(window_size, sigma):
   gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
   return gauss/gauss.sum()

def create_window_3D(window_size, channel):
   _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
   _2D_window = _1D_window.mm(_1D_window.t())
   _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
   window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
   return window

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
   mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
   mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

   mu1_sq = mu1.pow(2)
   mu2_sq = mu2.pow(2)

   mu1_mu2 = mu1*mu2

   sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
   sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
   sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

   C1 = 0.01**2
   C2 = 0.03**2

   ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

   if size_average:
      return ssim_map.mean()
   else:
      return ssim_map.mean(1).mean(1).mean(1)

class SSIM3D(nn.Module):
   def __init__(self, window_size = 11, size_average = True):
      super(SSIM3D, self).__init__()
      self.window_size = window_size
      self.size_average = size_average
      self.channel = 1
      self.window = create_window_3D(window_size, self.channel)

   def forward(self, img1, img2):
      (_, channel, _, _, _) = img1.size()

      if channel == self.channel and self.window.data.type() == img1.data.type():
         window = self.window
      else:
         window = create_window_3D(self.window_size, channel)
            
         if img1.is_cuda:
            window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

      return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
   (_, channel, _, _, _) = img1.size()
   window = create_window_3D(window_size, channel)
    
   if img1.is_cuda:
      window = window.cuda(img1.get_device())
   window = window.type_as(img1)
    
   return _ssim_3D(img1, img2, window, window_size, channel, size_average)
