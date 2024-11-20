import os
import numpy as np
import pyexr

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

import dataset
import net

import time
import cv2

def ssim(img1, img2):
    """
    SSIM is a measure of image quality that quantifies the structural similarity between two images.
    img1: torch.Tensor or np.ndarray, shape (N, C, H, W)
    img2: torch.Tensor or np.ndarray, shape (N, C, H, W)
    """
    assert img1.shape == img2.shape

    if isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor):
        img1, img2 = img1.cpu(), img2.cpu()
        kernel = dict(kernel_size=11, stride=1, padding=5)
        mu1, mu2 = F.avg_pool2d(img1, **kernel), F.avg_pool2d(img2, **kernel)
        sigma1_sq = F.avg_pool2d(img1**2, **kernel) - mu1**2
        sigma2_sq = F.avg_pool2d(img2**2, **kernel) - mu2**2
        sigma12 = F.avg_pool2d(img1*img2, **kernel) - mu1*mu2
    elif isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):
        blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
        mu1, mu2 = blur(img1), blur(img2)
        sigma1_sq = blur(img1**2) - mu1**2
        sigma2_sq = blur(img2**2) - mu2**2
        sigma12 = blur(img1*img2) - mu1*mu2
    else:
        raise TypeError("Input images must be both torch.Tensor or both np.ndarray")

    C1, C2 = 0.01**2, 0.03**2
    l12 = (2*mu1*mu2 + C1) / (mu1**2 + mu2**2 + C1)
    c12 = (2*torch.sqrt(sigma1_sq)*torch.sqrt(sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
    s12 = (sigma12 + C2/2) / (torch.sqrt(sigma1_sq)*torch.sqrt(sigma2_sq) + C2/2)
    
    return l12 * c12 * s12

def BMFRGammaCorrection(img):
    if isinstance(img, np.ndarray):
        return np.clip(np.power(np.maximum(img, 0.0), 0.454545), 0.0, 1.0)
    elif isinstance(img, torch.Tensor):
        return torch.pow(torch.clamp(img, min=0.0, max=1.0), 0.454545)

# def ComputeMetrics(truth_img, test_img):    
#     truth_img = BMFRGammaCorrection(truth_img)
#     test_img  = BMFRGammaCorrection(test_img)
    
#     SSIM = ssim(truth_img, test_img)
#     PSNR = peak_signal_noise_ratio(truth_img, test_img)
#     return SSIM, PSNR

def ComputeMetrics(truth_img, test_img):    
    truth_img = BMFRGammaCorrection(truth_img)
    test_img  = BMFRGammaCorrection(test_img)
    
    min_dim = min(truth_img.shape[:2])
    win_size = min(7, min_dim // 2 * 2 + 1)  # 确保 win_size 是奇数且不超过图像尺寸
    
    # SSIM = structural_similarity(truth_img, test_img, multichannel=True)
    SSIM = structural_similarity(truth_img, test_img, win_size=win_size, channel_axis=-1, data_range=1.0)
    PSNR = peak_signal_noise_ratio(truth_img, test_img)
    return SSIM, PSNR

def Inference(model, device, dataloader, saving_root=""):
    model.eval()
    SSIMs = []
    PSNRs = []
    TIMEs = []
    with torch.no_grad():
        for img_idx, (inputs_crops, targets_crops) in enumerate(dataloader):
            inputs = inputs_crops.to(device, non_blocking=True)
            targets = targets_crops.to(device, non_blocking=True)
            
            start_time = time.time()
            outputs = model(inputs).detach()
            end_time = time.time()
            
            output = outputs.cpu().numpy()[0].transpose((1, 2, 0)) # BMFR
            target = targets.cpu().numpy()[0].transpose((1, 2, 0))
            SSIM, PSNR = ComputeMetrics(target, output)
            SSIMs.append(SSIM)
            PSNRs.append(PSNR)
            TIMEs.append(end_time - start_time)
                
            pyexr.write(os.path.join(saving_root, str(img_idx)+".exr"), output)
            
    print("Test:")
    SSIM_mean = np.mean(SSIMs)
    PSNR_mean = np.mean(PSNRs)
    TIME_mean = np.mean(TIMEs)
    print("mean SSIM:", SSIM_mean)
    print("mean PSNR:", PSNR_mean)
    print("mean time", TIME_mean)
    SSIMs.append("mean: "+str(SSIM_mean))
    PSNRs.append("mean: "+str(PSNR_mean))
    TIMEs.append("mean: "+str(TIME_mean))
    np.savetxt(os.path.join(saving_root, "ssim.txt"), SSIMs, fmt="%s")
    np.savetxt(os.path.join(saving_root, "psnr.txt"), PSNRs, fmt="%s")
    np.savetxt(os.path.join(saving_root, "time.txt"), TIMEs, fmt="%s")
            
    return SSIM_mean, PSNR_mean



if __name__ == "__main__":
    torch.cuda.set_device(5)
    torch.backends.cudnn.deterministic = True  # same result for cpu and gpu
    torch.backends.cudnn.benchmark = False # key in here: Should be False. Ture will make the training process unstable
    device = torch.device("cuda")
    print(f"test on device: {device}")

    database = dataset.DataBase()
    dataset_test = dataset.BMFRFullResAlDataset(database, use_test=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    timestamp = "ver_1120"
    episode_name = "nppd"
    test_saving_root = os.path.join("results", timestamp, episode_name)
    os.makedirs(test_saving_root, exist_ok=True)

    model_pretrain = net.repWeightSharingKPNet(device).to(device)
    checkpoint = torch.load("./checkpoints/sponza/model.pt") # NOTE
    model_pretrain.load_state_dict(checkpoint['model_state_dict'])
    model_deployment = net.repWeightSharingKPNet(device, is_deployment=True, model_pretrain=model_pretrain).to(device)
    Inference(model_deployment, device, dataloader_test, test_saving_root)

