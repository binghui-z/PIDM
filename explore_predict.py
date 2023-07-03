# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import warnings

warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config as DiffConfig
import numpy as np
from config.diffconfig import DiffusionConfig, get_model_conf
import torch.distributed as dist
import os, glob, cv2, time, shutil, pickle, random
from models.unet_autoenc import BeatGANsAutoencConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
import torchvision.transforms as transforms
import argparse
from config.dataconfig import Config as DataConfig
from data.dataloader import Interhand26m

class Predictor(Interhand26m):
    def __init__(self, opt, is_inference, labels_required = False):
        """Load the model into memory to make running multiple predictions efficient"""
        super(Predictor, self).__init__(opt, is_inference, labels_required) # 要继承Interhand26m初始化中的内容，则需要用super,如果只是继承函数则不需要

        conf = DiffConfig(DiffusionConfig, opt.cong_file, show=False)

        self.model = get_model_conf().make_model()
        ckpt = torch.load(opt.model_file)
        self.model.load_state_dict(ckpt["ema"])
        self.model = self.model.cuda()
        self.model.eval()

        self.betas = conf.diffusion.beta_schedule.make()
        self.diffusion = create_gaussian_diffusion(self.betas, predict_xstart = False)#.to(device)
        
        # self.pose_list = glob.glob('data/deepfashion_256x256/target_pose/*.npy')
        rand_id = np.random.randint(0, len(self.data_Lst))
        tgt_pose = self.data_Lst[rand_id]
        pose_Dic = pickle.load(open(tgt_pose, "rb"), encoding='latin1')
        self.pose_list = []
        self.img_list = []
        for key in pose_Dic.keys():
            input = cv2.resize(pose_Dic[key]["image"], dsize=(self.resolution, self.resolution))
            img, param = self.get_image_tensor(input)
            target_label_tensor = self.get_label_tensor(self.proj_joints(pose_Dic[key]["joint_c"], pose_Dic[key]["cam_param"]["cameraIn"]), img, param) 
            self.img_list.append(img)
            self.pose_list.append(target_label_tensor)

    def predict_pose(
        self,
        num_poses=1,
        sample_algorithm='ddim',
        nsteps=100,

        ):
        """Run a single prediction on the model"""
        src = self.img_list[0].cuda()
        tgt_pose = torch.stack([ps.cuda() for ps in np.random.choice(self.pose_list, num_poses)], 0)

        src = src.repeat(num_poses,1,1,1)

        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, tgt_pose], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose])
            samples = xs[-1].cuda()

        samples_grid = torch.cat([src[0],torch.cat([samps for samps in samples], -1)], -1)
        samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0)/2.0
        pose_grid = torch.cat([torch.zeros_like(src[0]),torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)

        output = torch.cat([1-pose_grid, samples_grid], -2)

        numpy_imgs = output.unsqueeze(0).permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save('output_pose.png')


    def predict_appearance(
        self,
        sample_algorithm='ddim',
        nsteps=100,
        ):
        """Run a single prediction on the model"""
        src_id, ref_id = random.sample(range(len(self.img_list)), 2) #随机采样image和ref_img
        src = Image.fromarray(cv2.cvtColor(self.img_list[src_id],cv2.COLOR_BGR2RGB))
        src = self.transforms(src).unsqueeze(0).cuda()
        
        ref = Image.fromarray(cv2.cvtColor(self.img_list[ref_id],cv2.COLOR_BGR2RGB))
        ref = self.transforms(ref).unsqueeze(0).cuda()

        mask = BackgroundRemoval()(self.img_list[ref_id])
        mask = transforms.ToTensor()(Image.fromarray(cv2.cvtColor(mask[:,:,:3],cv2.COLOR_BGR2RGB))).unsqueeze(0).cuda()

        ref_pose = self.pose_list[ref_id]
        pose =  ref_pose.unsqueeze(0).cuda()


        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond = [src, pose, ref, mask], progress = True, cond_scale = 2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000//nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, pose, ref, mask], diffusion=self.diffusion)
            samples = xs[-1].cuda()


        samples = torch.clamp(samples, -1., 1.)

        output = (torch.cat([src, ref, mask*2-1, samples], -1) + 1.0)/2.0

        numpy_imgs = output.permute(0,2,3,1).detach().cpu().numpy()
        fake_imgs = (255*numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save('results/output_app.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--cong_file', type=str, default=r'config\diffusion.conf')
    parser.add_argument('--model_file', type=str, default=r'checkpoints\64x64-2023-06-30_20-09-48\last.pt')

    # class interhand:
    parser.add_argument('--path', type=str, default=r'\\Seuvcl-7016\h\2306_neuhand_data\vcl4k_mask')
    parser.add_argument('--input_size', type=int, default=64)
    parser.add_argument('--scale_param', type=int, default=0)
    parser.add_argument('--resolution', type=int, default=64)
    args = parser.parse_args()

    obj = Predictor(args, is_inference=True, labels_required=True) # is_inference=True-> val; is_inference=False-> train; 

    # 姿态迁移
    obj.predict_pose(num_poses=8, sample_algorithm = 'ddim',  nsteps = 50)
    
    # 风格迁移
    # obj.predict_appearance(sample_algorithm = 'ddim',  nsteps = 50)
