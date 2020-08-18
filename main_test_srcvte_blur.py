import os.path
import cv2
import logging
import time
import os

import numpy as np
from datetime import datetime
from collections import OrderedDict
from scipy.io import loadmat
#import hdf5storage
from scipy import ndimage
from scipy.signal import convolve2d

import torch

from utils import utils_deblur
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from models.network_usrnet import USRNet as net


'''
Spyder (Python 3.6)
PyTorch 1.4.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/USRNet
        https://github.com/cszn/KAIR

If you have any question, please feel free to contact with me.
Kai Zhang (e-mail: cskaizhang@gmail.com)

by Kai Zhang (12/March/2020)
'''

"""
# --------------------------------------------
testing code of USRNet for the Table 1 in the paper
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={0--0},
  year={2020}
}
# --------------------------------------------
|--model_zoo                # model_zoo
   |--usrgan                # model_name, optimized for perceptual quality
   |--usrnet                # model_name, optimized for PSNR
   |--usrgan_tiny           # model_name, tiny model optimized for perceptual quality
   |--usrnet_tiny           # model_name, tiny model optimized for PSNR
|--testsets                 # testsets
   |--set5                  # testset_name
   |--set14
   |--urban100
   |--bsd100
   |--srbsd68               # already cropped
|--results                  # results
   |--srbsd68_usrnet        # result_name = testset_name + '_' + model_name
   |--srbsd68_usrgan
   |--srbsd68_usrnet_tiny
   |--srbsd68_usrgan_tiny
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    model_name = 'usrnet_tiny'      # 'usrgan' | 'usrnet' | 'usrgan_tiny' | 'usrnet_tiny'
    testset_name = 'srcvte'      # test set,  'set5' | 'srbsd68' | 'srcvte'
    test_sf = [4] # if 'gan' in model_name else [2, 3, 4]  # scale factor, from {1,2,3,4}

    load_kernels = False
    show_img = False           # default: False
    save_L = False              # save LR image
    save_E = True              # save estimated image
    save_LEH = False           # save zoomed LR, E and H images

    # ----------------------------------------
    # load testing kernels
    # ----------------------------------------
    # kernels = hdf5storage.loadmat(os.path.join('kernels', 'kernels.mat'))['kernels']
    kernels = loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels'] if load_kernels else None

    n_channels = 1 if 'gray' in  model_name else 3  # 3 for color image, 1 for grayscale image
    model_pool = '/home/dengzeshuai/pretrained_models/USRnet/'  # fixed
    testsets = '/home/datasets/sr/'     # fixed
    results = 'results'       # fixed
    noise_level_img = 0       # fixed: 0, noise level for LR image
    noise_level_model = noise_level_img  # fixed, noise level of model, default 0
    result_name = testset_name + '_' + model_name + '_blur'
    model_path = os.path.join(model_pool, model_name+'.pth')

    # ----------------------------------------
    # L_path = H_path, E_path, logger
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name)  # L_path and H_path, fixed, for Low-quality images
    if testset_name == 'srcvte':
        L_path = os.path.join(testsets, testset_name, 'LR_val')
        H_path = os.path.join(testsets, testset_name, 'HR_val')
        video_names = os.listdir(H_path)
    E_path = os.path.join(results, result_name)    # E_path, fixed, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    if 'tiny' in model_name:
        model = net(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                    nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    else:
        model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                    nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for key, v in model.named_parameters():
        v.requires_grad = False
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    model = model.to(device)

    logger.info('Model path: {:s}'.format(model_path))
    logger.info('Params number: {}'.format(number_parameters))
    logger.info('Model_name:{}, image sigma:{}'.format(model_name, noise_level_img))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    need_H = True if H_path is not None else False
    H_paths = util.get_image_paths(H_path) if need_H else None

    # --------------------------------
    # read images
    # --------------------------------
    test_results_ave = OrderedDict()
    test_results_ave['psnr_sf_k'] = []
    test_results_ave['ssim_sf_k'] = []
    test_results_ave['psnr_y_sf_k'] = []
    test_results_ave['ssim_y_sf_k'] = []

    for sf in test_sf:
        loop = kernels.shape[1] if load_kernels else 1
        for k_index in range(loop):

            test_results = OrderedDict()
            test_results['psnr'] = []
            test_results['ssim'] = []
            test_results['psnr_y'] = []
            test_results['ssim_y'] = []

            if load_kernels:
                kernel = kernels[0, k_index].astype(np.float64)
            else:
                ## other kernels
                # kernel = utils_deblur.blurkernel_synthesis(h=25)  # motion kernel
                kernel = utils_deblur.fspecial('gaussian', 25, 1.6) # Gaussian kernel
                kernel = sr.shift_pixel(kernel, sf)  # pixel shift; optional
                kernel /= np.sum(kernel)

            util.surf(kernel) if show_img else None
            # idx = 0

            for idx, img in enumerate(L_paths):

                # --------------------------------
                # (1) classical degradation, img_L
                # --------------------------------
                
                img_name, ext = os.path.splitext(os.path.basename(img))    
                if testset_name == 'srcvte':
                    video_name = os.path.basename(os.path.dirname(img))
                img_L = util.imread_uint(img, n_channels=n_channels)
                img_L = util.uint2single(img_L)
                
                # generate degraded LR image
                # img_L = ndimage.filters.convolve(img_H, kernel[..., np.newaxis], mode='wrap')  # blur
                # img_L = sr.downsample_np(img_L, sf, center=False)  # downsample, standard s-fold downsampler
                # img_L = util.uint2single(img_L)  # uint2single

                # np.random.seed(seed=0)  # for reproducibility
                # img_L += np.random.normal(0, noise_level_img, img_L.shape) # add AWGN

                util.imshow(util.single2uint(img_L)) if show_img else None

                x = util.single2tensor4(img_L)
                k = util.single2tensor4(kernel[..., np.newaxis])
                sigma = torch.tensor(noise_level_model).float().view([1, 1, 1, 1]) 
                [x, k, sigma] = [el.to(device) for el in [x, k, sigma]]

                # --------------------------------
                # (2) inference
                # --------------------------------
                x = model(x, k, sf, sigma)

                # --------------------------------
                # (3) img_E
                # --------------------------------
                img_E = util.tensor2uint(x)
                
                if save_E:
                    if testset_name == 'srcvte':
                        save_path = os.path.join(E_path, video_name)
                        util.mkdir(save_path)
                        # util.imsave(img_E, os.path.join(save_path, img_name+'_k'+str(k_index+1)+'.png'))
                        util.imsave(img_E, os.path.join(save_path, img_name+'.png'))
                    else:
                        util.imsave(img_E, os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index+1)+'_'+model_name+'.png'))


                # --------------------------------
                # (4) img_H
                # --------------------------------
                if need_H:
                    img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
                    img_H = img_H.squeeze()
                    img_H = util.modcrop(img_H, sf)
                    
                    psnr = util.calculate_psnr(img_E, img_H, border=sf)  # change with your own border
                    ssim = util.calculate_ssim(img_E, img_H, border=sf)
                    test_results['psnr'].append(psnr)
                    test_results['ssim'].append(ssim)

                    if np.ndim(img_H) == 3:  # RGB image
                        img_E_y = util.rgb2ycbcr(img_E, only_y=True)
                        img_H_y = util.rgb2ycbcr(img_H, only_y=True)
                        psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=sf)
                        ssim_y = util.calculate_ssim(img_E_y, img_H_y, border=sf)
                        test_results['psnr_y'].append(psnr_y)
                        test_results['ssim_y'].append(ssim_y)
                        logger.info('{:->4d} --> {:>4s}--> {:>10s} -- x{:>2d} --k{:>2d} PSNR: {:.2f}dB SSIM: {:.4f}'.format(idx, video_name, img_name+ext, sf, k_index, psnr_y, ssim_y))
                    else:
                        logger.info('{:->4d} --> {:>4s}--> {:>10s} -- x{:>2d} --k{:>2d} PSNR: {:.2f}dB SSIM: {:.4f}'.format(idx, video_name, img_name+ext, sf, k_index, psnr, ssim))
                
            if need_H:
                ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
                ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
                logger.info('Average PSNR/SSIM(RGB) - {} - x{} --PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, sf, ave_psnr, ave_ssim))
                logger.info('------> Average PSNR(RGB) - {} - x{}, kernel:{} sigma:{} --PSNR: {:.2f} dB; SSIM: {:.4f}'.format(testset_name, sf, k_index+1, noise_level_model, ave_psnr, ave_ssim))
                if np.ndim(img_H) == 3:
                    ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                    ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                    logger.info('------> Average PSNR(Y) - {} - x{}, kernel:{} sigma:{} --PSNR: {:.2f} dB; SSIM: {:.4f}'.format(testset_name, sf, k_index+1, noise_level_model, ave_psnr_y, ave_ssim_y))

                test_results_ave['psnr_sf_k'].append(ave_psnr)
                test_results_ave['ssim_sf_k'].append(ave_ssim)
                if np.ndim(img_H) == 3:
                    test_results_ave['psnr_y_sf_k'].append(ave_psnr_y)
                    test_results_ave['ssim_y_sf_k'].append(ave_ssim_y)
    
    logger.info(test_results_ave['psnr_sf_k'])
    logger.info(test_results_ave['ssim_sf_k'])
    if np.ndim(img_H) == 3:
        logger.info(test_results_ave['psnr_y_sf_k'])
        logger.info(test_results_ave['ssim_y_sf_k'])


if __name__ == '__main__':

    main()
