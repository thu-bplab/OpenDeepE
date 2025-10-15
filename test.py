import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import h5py
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from models import FullyDenseUNet2D
from datasets.dataset import NiftiVesselPairGenerator
from eval.test_image import NiftiInferenceEvaluator, RMSE, PSNR, SSIM2D


def main(args):
    # Configs
    dataroot = args.dataroot
    cate = args.cate
    batch_size = args.batch_size
    model_path = args.model_path
    cuda = args.cuda
    ddp = args.ddp
    results_dir = args.results_dir
    model = args.model
    channel_mults = [int(x) for x in args.channel_mults.split(",")]

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    # Load model
    if model == 'deepe':
        print("Here comes the DeepE model...")
        model = FullyDenseUNet2D(in_channels=1, out_channels=1).to(device)
    else:
        raise ValueError(f"Model {model} not found")

    # DataParallel: Multi GPU
    if ddp:
        model = nn.DataParallel(model)
    
    # state_dict = torch.load(model_path, map_location=device)
    # new_state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items() if k.startswith('module.')}
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Dataloader
    print("Initializing Test Datasets and DataLoaders...")
    inputfolder = os.path.join(dataroot, cate)
    transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.unsqueeze(0)),
    ]) 
    # train
    test_dataset = NiftiVesselPairGenerator(
        inputfolder,
        dataset_type='test',
        apply_transform=transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=40)

    # Evaluation
    evaluator = NiftiInferenceEvaluator(model, RMSE(), PSNR(), SSIM2D(), results_dir)
    test_rmse, test_psnr, test_ssim2d = evaluator.infer(test_dataloader)
    print(f"Test Results: RMSE: {test_rmse}, PSNR: {test_psnr}, SSIM2D: {test_ssim2d}")

    # Save quantitative test results
    if results_dir != '':
        test_results = {
            'test_rmse': test_rmse, 
            'test_psnr': test_psnr,
            'test_ssim2d': test_ssim2d
        }
        df = pd.DataFrame([test_results])
        df.to_csv(f'{results_dir}/test_results.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Medical Image Testing Script')
    parser.add_argument('--dataroot', type=str, default='/home/kongdi24/Dataset/nifti', help='root directory of the dataset')
    parser.add_argument('--cate', type=str, default='vessel', help='category of the input dataset')
    
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for testing')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model file')
    parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')
    parser.add_argument('--ddp',action='store_true',default=True)
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save test results (default: none)')

    parser.add_argument("--model", type=str, default="deepe", choices=["deepe"])
    parser.add_argument("--channel-mults", default="1,2,2,4,4", help="Defines the U-net architecture's depth and width.")
    parser.add_argument("--dropout", default=0.0, type=float)

    args = parser.parse_args()
    main(args)
