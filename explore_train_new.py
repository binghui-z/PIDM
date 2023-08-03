import os
import warnings

warnings.filterwarnings("ignore")

import time, cv2, torch
import torch.distributed as dist
from config.diffconfig import DiffusionConfig, get_model_conf
from config.dataconfig import Config as DataConfig
from tensorfn import load_config as DiffConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
from tensorfn.optim import lr_scheduler
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import data as deepfashion_data
import data.dataloader as Interhand26m
import torchvision
from torchvision.utils import make_grid, save_image
from model import UNet
from torch.utils.tensorboard import SummaryWriter

def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)




def train(conf, loader, val_loader, model, ema, diffusion, betas, optimizer, scheduler, guidance_prob, cond_scale, device):

    logs_folder = conf.training.ckpt_path + "/log";  os.makedirs(logs_folder, exist_ok=True)
    samples_folder = conf.training.ckpt_path + "/samples";  os.makedirs(samples_folder, exist_ok=True)
    checkpoints_folder = conf.training.ckpt_path + "/checkpoints";  os.makedirs(checkpoints_folder, exist_ok=True)
    writer = SummaryWriter(logs_folder)
    i = 0

    loss_list = []
    loss_mean_list = []
    loss_vb_list = []
 
    for epoch in range(0, 300):


        start_time = time.time()

        for batch in tqdm(loader):

            i = i + 1

            img = torch.cat([batch['source_image'], batch['target_image']], 0)
            target_img = torch.cat([batch['target_image'], batch['source_image']], 0)
            target_pose = torch.cat([batch['target_skeleton'], batch['source_skeleton']], 0)

            img = img.to(device)
            target_img = target_img.to(device)
            target_pose = target_pose.to(device)
            time_t = torch.randint(
                0,
                conf.diffusion.beta_schedule["n_timestep"],
                (img.shape[0],),
                device=device,
            )

            loss_dict = diffusion.training_losses(model, x_start = target_img, t = time_t, cond_input = [img, target_pose], prob = 1 - guidance_prob)
            
            loss = loss_dict['loss'].mean()
            loss_mse = loss_dict['mse'].mean()
            loss_vb = loss_dict['vb'].mean()
        

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            scheduler.step()
            optimizer.step()
            loss = loss_dict['loss'].mean()

            loss_list.append(loss.detach().item())
            loss_mean_list.append(loss_mse.detach().item())
            loss_vb_list.append(loss_vb.detach().item())

            accumulate(
                # ema, model.module, 0 if i < conf.training.scheduler.warmup else 0.9999
                ema, model, 0 if i < conf.training.scheduler.warmup else 0.9999
            )


            if i%args.save_logs_every_iters == 0 and args.save_logs_every_iters>0:

                writer.add_scalar('loss', (sum(loss_list)/len(loss_list)), i) 
                writer.add_scalar('loss_vb', (sum(loss_vb_list)/len(loss_vb_list)), i) 
                writer.add_scalar('loss_mean', (sum(loss_mean_list)/len(loss_mean_list)), i) 
                loss_list = []
                loss_mean_list = []
                loss_vb_list = []

            if i%args.save_checkpoints_every_iters == 0 and args.save_checkpoints_every_iters>0:
                path = checkpoints_folder + f"/model_iter-{str(i).zfill(6)}.pt"
                torch.save(
                    {
                        "model": model_module.state_dict(),
                        "ema": ema.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter": i,
                    },
                    path + f"/model_{str(i).zfill(6)}.pt"
                )

        if (epoch)%args.save_checkpoints_every_epochs==0 and args.save_checkpoints_every_epochs>0:

            print ('Epoch Time '+str(int(time.time()-start_time))+' secs')
            print ('Model Saved Successfully for #epoch '+str(epoch)+' #steps '+str(i))

            model_module = model
            path = conf.training.ckpt_path + '/last.pt'
            torch.save(
                {
                    "model": model_module.state_dict(),
                    "ema": ema.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": i,
                    "epoch": epoch,
                },
                path
            )

        if (epoch)%args.save_images_every_epochs==0 and args.save_images_every_epochs>0:

            print ('Generating samples at epoch number ' + str(epoch))
            val_batch = next(val_loader)
            val_img = val_batch['source_image'].cuda()
            val_pose = val_batch['target_skeleton'].cuda()

            with torch.no_grad():

                if args.sample_algorithm == 'ddpm':
                    print ('Sampling algorithm used: DDPM')
                    samples = diffusion.p_sample_loop(ema, x_cond = [val_img, val_pose], progress = True, cond_scale = cond_scale)
                elif args.sample_algorithm == 'ddim':
                    print ('Sampling algorithm used: DDIM')
                    nsteps = 50
                    noise = torch.randn(val_img.shape).cuda()
                    seq = range(0, 1000, 1000//nsteps)
                    xs, x0_preds = ddim_steps(noise, seq, ema, betas.cuda(), [val_img, val_pose])
                    samples = xs[-1].cuda()


            grid = torch.cat([(val_img+1)/2, val_pose[:,:3], (samples+1)/2], 0)
            grid = torchvision.utils.make_grid(tensor = grid, nrow=len(val_batch))
            save_image(grid, str(samples_folder + f'/sample_epoch-{epoch}.png'))


def main(settings, EXP_NAME):

    [args, DiffConf, DataConf] = settings

    if DiffConf.ckpt is not None: 
        DiffConf.training.scheduler.warmup = 0

    DiffConf.distributed = False
    # local_rank = int(os.environ['LOCAL_RANK'])
    
    DataConf.data.train.batch_size = args.batch_size//2  #src -> tgt , tgt -> src
    
    # val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required = True, distributed = True)
    val_dataset, train_dataset = Interhand26m.get_train_val_dataloader(DataConf.data, labels_required = True)

    
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    val_dataset = iter(cycle(val_dataset))

    model = get_model_conf().make_model()
    model = model.to(args.device)
    ema = get_model_conf().make_model()
    ema = ema.to(args.device)

    # if DiffConf.distributed:
    #     model = nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[local_rank],
    #         find_unused_parameters=True
    #     )

    optimizer = DiffConf.training.optimizer.make(model.parameters())
    scheduler = DiffConf.training.scheduler.make(optimizer)

    if DiffConf.ckpt is not None:
        ckpt = torch.load(DiffConf.ckpt, map_location=lambda storage, loc: storage)

        if DiffConf.distributed:
            model.module.load_state_dict(ckpt["model"])

        else:
            model.load_state_dict(ckpt["model"])

        ema.load_state_dict(ckpt["ema"])
        scheduler.load_state_dict(ckpt["scheduler"])

    betas = DiffConf.diffusion.beta_schedule.make()
    diffusion = create_gaussian_diffusion(betas, predict_xstart = False)

    train(
        DiffConf, train_dataset, val_dataset, model, ema, diffusion, betas, optimizer, scheduler, args.guidance_prob, args.cond_scale, args.device
    )

if __name__ == "__main__":

    # init_distributed()

    import argparse

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='pidm_deepfashion')
    parser.add_argument('--DiffConfigPath', type=str, default='./config/diffusion.conf')
    parser.add_argument('--DataConfigPath', type=str, default='./config/data.yaml')
    parser.add_argument('--dataset_path', type=str, default='./dataset/deepfashion')
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--cond_scale', type=int, default=2)
    parser.add_argument('--guidance_prob', type=int, default=0.1)
    parser.add_argument('--sample_algorithm', type=str, default='ddim') # ddpm, ddim
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--save_wandb_logs_every_iters', type=int, default=50)
    parser.add_argument('--save_wandb_images_every_epochs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_gpu', type=int, default=8)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    parser.add_argument('--save_logs_every_iters', type=int, default=50)
    parser.add_argument('--save_images_every_iters', type=int, default=-1)
    parser.add_argument('--save_checkpoints_every_iters', type=int, default=-1)
    parser.add_argument('--save_images_every_epochs', type=int, default=1)
    parser.add_argument('--save_checkpoints_every_epochs', type=int, default=1)

    args = parser.parse_args()

    print ('Experiment: '+ args.exp_name)
    args.exp_name = '64x64-'+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    DiffConf = DiffConfig(DiffusionConfig,  args.DiffConfigPath, args.opts, False)
    DataConf = DataConfig(args.DataConfigPath)
    DiffConf.training.ckpt_path = os.path.join(args.save_path, args.exp_name)
    # DiffConf.ckpt = r"checkpoints\64x64-2023-06-30_20-09-48\last.pt"
    

    if not os.path.isdir(args.save_path): os.mkdir(args.save_path)
    if not os.path.isdir(DiffConf.training.ckpt_path): os.mkdir(DiffConf.training.ckpt_path)
    main(settings = [args, DiffConf, DataConf], EXP_NAME = args.exp_name)
