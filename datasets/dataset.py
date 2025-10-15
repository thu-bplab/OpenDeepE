from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
import torchvision.transforms.functional as TF
from scipy.ndimage import zoom
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import re
import os


class NiftiVesselPairGenerator(Dataset):
    def __init__(self, input_folder: str, dataset_type: str, apply_transform=None):
        self.input_folder = input_folder
        self.sparse_image = os.path.join(input_folder, dataset_type, 'sparse_image_two_angle')
        # self.sparse_image = os.path.join(input_folder, dataset_type, 'sparse_image_four_angle')
        self.gt_image = os.path.join(input_folder, dataset_type, 'gt_image')
        self.pair_files = self.pair_file()
        self.scaler = MinMaxScaler()
        self.transform = apply_transform

    def pair_file(self):
        def sort_key(filename):
            # 提取前缀和数字部分
            match = re.search(r'(\d+)_(\d+).nii', filename)
            prefix = int(match.group(1))  # 前缀部分转换为整数
            number = int(match.group(2))  # 数字部分转换为整数
            return (prefix, number)  # 返回一个元组，用于多条件排序
        # Sort for Testing:
        input_files = sorted(glob(os.path.join(self.sparse_image, '*.nii')), key=sort_key)
        target_files = sorted(glob(os.path.join(self.gt_image, '*.nii')), key=sort_key)
        # Sort for Training:
        # input_files = sorted(glob(os.path.join(self.sparse_image, '*.nii')))
        # target_files = sorted(glob(os.path.join(self.gt_image, '*.nii')))
        pairs = []
        for input_file, target_file in zip(input_files, target_files):
            
            input_digits = int("".join(re.findall("\d", input_file)))
            target_digits = int("".join(re.findall("\d", target_file)))
            # 检查数字是否匹配
            if input_digits != target_digits:
                print(f"Mismatch detected:")
                print(f"  Input file: {input_file}, Digits: {input_digits}")
                print(f"  Target file: {target_file}, Digits: {target_digits}")
                raise AssertionError("File numbers do not match!")

            assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file)))
            input_number = int(re.findall("\d+", os.path.basename(input_file))[0])
            target_number = int(re.findall("\d+", os.path.basename(target_file))[0])
            # 检查文件名中的数字是否匹配
            if input_number != target_number:
                print(f"Mismatch detected:")
                print(f"  Input file: {input_file}, Number: {input_number}")
                print(f"  Target file: {target_file}, Number: {target_number}")
                raise AssertionError("File numbers do not match!")

            assert input_number == target_number, "File Number Does Not Match!!"
            pairs.append((input_file, target_file))
        return pairs

    def read_image(self, file_path):
        # 读取 NIfTI 文件并转换为 float32
        img = nib.load(file_path).get_fdata(dtype=np.float32)
        img = img / 1000.0
        # 截断为 float16
        img = img.astype(np.float16)
        # 转换回 float32
        img = img.astype(np.float32)
        
        return img

    def normalize(self, img, img_min, img_max):
        img = (img - img_min) / (img_max - img_min)  # 缩放到 [0, 1]
        img = 2 * img - 1  # 缩放到 [-1, 1]
        return img

    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice], cmap='gray')
        plt.show()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        
        input_img = self.read_image(input_file)
        target_img = self.read_image(target_file)

        if input_img.shape[0] == 240 and input_img.shape[1] == 240:
            zoom_factor = (256 / 240, 256 / 240)
            input_img = zoom(input_img, zoom_factor)
            target_img = zoom(target_img, zoom_factor)

        if self.transform is not None:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        
        return {'input': input_img, 'target': target_img}




def main():
    transform = Compose([
    Lambda(lambda img: TF.to_tensor(img)),
    ])
    dataset = ImagePairGenerator(input_folder='/data/bml/KD/Dataset/2D/brain', dataset_type='train', apply_transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(len(dataset)) 
    
    for batch in dataloader:
            sparse = batch['input']
            gt = batch['target']
            print(sparse.shape)
            print(gt.shape)

if __name__ == '__main__':
    main()

