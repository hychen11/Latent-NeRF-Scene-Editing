import pyrallis
import os
import torch
import imageio
from math import sqrt
from torchvision.utils import save_image, make_grid
from src.latent_nerf.configs.train_config import TrainConfig
from src.latent_nerf.training.trainer import Trainer

from bg_generator import bg_generator
from bg_nerf.provider import bg_NeRFDataset
from bg_nerf.network import bg_NeRFNetwork
from bg_nerf.utils import bg_Trainer

from fg_generator import fg_generator
from fg_nerf.provider import fg_NeRFDataset
from fg_nerf.network import fg_NeRFNetwork
# from fg_nerf.utils import fg_Trainer

from src.latent_nerf.training.trainer import Trainer
from src.latent_nerf.models.network_grid import NeRFNetwork
from src.latent_nerf.training.nerf_dataset import NeRFDataset
from src.latent_nerf.configs.render_config import RenderConfig
from src.latent_nerf.configs.train_config import TrainConfig

from config import bg_config,fg_config
import matplotlib.pyplot as plt
import numpy as np
from util.camera import get_camera_mat, get_random_pose, get_camera_pose

import util.bounding_box_generator as box
import util.decoder_ref as dec
import util.generator_ref as gen
import util.camera as cam
from sd import StableDiffusion
# from src.stable_diffusion import StableDiffusion
from src.latent_nerf.training.losses.sparsity_loss import sparsity_loss
from src.latent_nerf.training.losses.shape_loss import ShapeLoss
from src.optimizer import Adan

from PIL import Image
from src.utils import make_path
from pathlib import Path
import sys
from fg_nerf.utils import *
from encoding import get_encoder
from loguru import logger
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#ordination is based on background

@pyrallis.wrap()
def main(cfg:TrainConfig):
    # print("bg_generate begin")
    # bg_generator()
    # print("bg_generate finished")

    # print("clear cache begin")
    # torch.cuda.empty_cache()
    # print("clear cache finish")

    # print("fg_generate begin")
    # fg_generator()
    # print("fg_generate finished")
    
    # print("fg_generate begin")
    # trainer = Trainer(cfg)
    # trainer.train()
    # print("fg_generate finished")

    print("render begin")
    render_add()
    print("render finished")
    torch.cuda.empty_cache()
    print("Global_Training begin")
    Global_Training()
    print("Global_Training finished")

    

def render_add():
    bbox_generator=box.BoundingBoxGenerator()
    gen_=gen.Generator(bounding_box_generator=bbox_generator)
    
    batch_size=1
    device='cuda'

    # don't rotate bg
    R_bg = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float().to(device)

    #load model
    opt=bg_config()
    test_loader = bg_NeRFDataset(opt, device=device, type='test').dataloader()
    # test_loader = bg_NeRFDataset(opt, device=device, type='test',downscale=5).dataloader()

    mat=get_camera_mat_from_intrinsics(test_loader.intrinsics).repeat(200,1,1)

    #each pic has 800 * 800 * 3
    camera_matrices=mat,test_loader.poses.to(device)

    #get transformations
    s_val=[[0.5,0.5,0.5]]
    #x,y,z translation
    # t_val=[[0.5,0.6,0.5]]
    t_val=[[0.6,0.6,0.6]]
    r_val=[0.0]
    transformations=gen_.get_transformations(s_val,t_val,r_val,batch_size)
    # print(f"transformations{transformations}")
    
    with torch.no_grad():
        #get output scene picture
        out=gen_(batch_size=batch_size,camera_matrices=camera_matrices,transformations=transformations,bg_rotation=R_bg,mode='val',loader=test_loader)
        #global Training Score Distillation (use latent nerf)

def Global_Training():
    # load two models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bg_opt=bg_config()
    seed_everything(bg_opt.seed)
    model_bg = bg_NeRFNetwork(
        encoding="hashgrid",
        bound=bg_opt.bound,
        cuda_ray=bg_opt.cuda_ray,
        density_scale=1,
        min_near=bg_opt.min_near,
        density_thresh=bg_opt.density_thresh,
        bg_radius=bg_opt.bg_radius,
    )
    bg_checkpoint_path='/data/hychen/final/step1_chair/checkpoints/ngp_ep0300.pth'
    bg_trainer = bg_Trainer(name='ngp' , opt=bg_opt, workspace=bg_opt.workspace,model=model_bg, device=device,use_checkpoint="chy")
    bg_trainer.model.load_state_dict(torch.load(bg_checkpoint_path)['model'],strict=False)
    test_loader = bg_NeRFDataset(bg_opt, device=device, type='test',downscale=5).dataloader()
    train_loader = bg_NeRFDataset(bg_opt, device=device, type='train').dataloader()
    
    for p in bg_trainer.model.parameters():
        p.requires_grad = False
    
    cfg=TrainConfig()
    fg_checkpoint_path='/data/hychen/final/step2.2/lego/checkpoints/step_005000.pth'
    # fg_checkpoint_path='/data/hychen/final/step2.1/dog/checkpoints/df_ep0100.pth'
    # fg_checkpoint_path='/data/hychen/ln_data/experiments/lego_man/checkpoints/step_005000.pth'
    fg_trainer = Trainer(cfg)
    guidance = StableDiffusion(device)
    for p in guidance.parameters():
        p.requires_grad = False
    fg_trainer.diffusion=guidance
    # fg_trainer.model.load_state_dict(torch.load(fg_checkpoint_path)['model'],strict=False)
    fg_trainer.load_checkpoint(fg_checkpoint_path)

    train_dataloader = NeRFDataset(cfg.render, device='cuda', type='train', H=cfg.render.train_h,
                                    W=cfg.render.train_w, size=100).dataloader()
    val_loader = NeRFDataset(cfg.render, device='cuda', type='val', H=cfg.render.eval_h,
                                W=cfg.render.eval_w,
                                size=cfg.log.eval_size).dataloader()
    # Will be used for creating the final video
    val_large_loader = NeRFDataset(cfg.render, device='cuda', type='val', H=cfg.render.eval_h,
                                    W=cfg.render.eval_w,
                                    size=cfg.log.full_eval_size).dataloader()
    downscale=5

    # opt=fg_config()
    # print(opt)

    # seed_everything(opt.seed)
    # model = fg_NeRFNetwork(opt)
    # print(model)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_loader = fg_NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()
    # downscale=5
    # valid_loader = fg_NeRFDataset(opt, device=device, type='val', H=opt.H//downscale, W=opt.W//downscale, size=200).dataloader()
    
    # # Adan usually requires a larger LR
    # optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
    # # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    # # guidance = StableDiffusion(device,model_name='CompVis/stable-diffusion-v1-4',latent_mode=False)
    # # guidance=StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)
    # guidance = StableDiffusion(device)
    # fg_trainer = fg_Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)

    # max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    # fg_trainer.train(train_loader, valid_loader, max_epoch)
    # # fg_checkpoint_path='/data/hychen/stable-dreamfusion-main/workspace_noshape/checkpoints/df_ep0100.pth'
    # # trainer.load_checkpoint(fg_checkpoint_path)

    # fg_trainer.test(test_loader)




    # fg_opt=fg_config()
    
    # model_fg = fg_NeRFNetwork(fg_opt).to(device)
    # fg_checkpoint_path='/data/hychen/ln_data/test/Dreamfusion_sd/checkpoints/df_ep0100.pth'
    # # fg_checkpoint_path='/data/hychen/ln_data/dog_0418/checkpoints/df_ep0003.pth'
    # # fg_checkpoint_path='/data/hychen/ln_data/dog_obj/checkpoints/df_ep0100.pth'
    # # fg_checkpoint_path='/home/hychen/stable-dreamfusion/workspace3/checkpoints/df_ep0100.pth'
    # optimizer = lambda model: torch.optim.Adam(model.get_params(fg_opt.lr), betas=(0.9, 0.99), eps=1e-15)
    # scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
    # guidance=StableDiffusion(device)
    # for p in guidance.parameters():
    #     p.requires_grad = False
    # fg_trainer = fg_Trainer(' '.join(sys.argv), 'df', fg_opt, model_fg, guidance, device=device, workspace=fg_opt.workspace, optimizer=optimizer, ema_decay=None, fp16=fg_opt.fp16, lr_scheduler=scheduler, use_checkpoint=fg_opt.ckpt, eval_interval=fg_opt.eval_interval, scheduler_update_every_step=True)
    # fg_trainer.load_checkpoint(fg_checkpoint_path)

    # # load training/value data: training data->ramdom pose
    # train_loader = fg_NeRFDataset(fg_opt, device=device, type='train', H=fg_opt.h, W=fg_opt.w, size=100).dataloader()
    # downscale=5
    # valid_loader = fg_NeRFDataset(fg_opt, device=device, type='val', H=fg_opt.H//downscale, W=fg_opt.W//downscale, size=200).dataloader()
    
    # train_loader = bg_NeRFDataset(bg_opt, device=device, type='train',downscale=5).dataloader()
    # valid_loader = bg_NeRFDataset(bg_opt, device=device, type='test',downscale=5).dataloader()
    # for data in train_loader:
    #     print(f"H W{data['H'],data['W']}")
    model_dict=[]

    # set training parameters
    lr=1e-3
    lr=5*lr
    fg_params = list(fg_trainer.nerf.get_params(lr=lr))
    bg_params = list(bg_trainer.model.get_params(lr=lr))
    params = fg_params + bg_params

    # params=[fg_trainer.model.get_params(lr=lr),transformations.get_params(lr=lr)]

    optimizer = torch.optim.Adam(params, betas=(0.9, 0.99), eps=1e-15)
    # optimizer=Adan(params,eps=1e-8,weight_decay=2e-5,max_grad_norm=5.0,foreach=False)
    scaler=torch.cuda.amp.GradScaler(enabled=True)

    #get transformations
    batch_size=1
    bbox_generator=box.BoundingBoxGenerator()
    gen_=gen.Generator(bounding_box_generator=bbox_generator)
    s_val=[[0.5,0.5,0.5]]
    t_val=[[0.5,0.6,0.5]]
    r_val=[0.0]
    transformations=gen_.get_transformations(s_val,t_val,r_val,batch_size)
    # transformations=torch.tensor(transformations,requires_grad=True)

    losses=init_losses()

    logger.info('Starting Global Training ^_^')
    train_step = 0
    iters = 500
    save_interval=100
    save_path='/data/hychen/final/step4.2_lego_bg'
    path=Path(save_path)
    train_renders_path=make_path(path)
    bg_trainer.model.train()
    fg_trainer.nerf.train()
    past_checkpoints=[]
    ckpt_path=make_path(path/'checkpoints')
    loss = 0

    # update_extra_interval = 16
    from tqdm import tqdm
    pbar = tqdm(total=iters, initial=train_step,bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    # H,W=160,160
    H,W=64,64

    depth_range=[0.5, 6.0]
    n_steps=512
    n_points=H*W
    di = depth_range[0] + torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (depth_range[1] - depth_range[0])
    di = di.repeat(batch_size, n_points, 1).to(device)
    box_bound=1
    padd = 0.1
    all_preds1 = []
    all_preds2 = []
    #train all 64 64 , eval all 800 800
    while train_step < iters:
        #add a model change function
        for i,data in enumerate(train_loader):
            train_step+=1
            pbar.update(1)
            optimizer.zero_grad()

            color, sigma = [], []
            # render_pic [1, 3, 160, 160]
            with torch.cuda.amp.autocast(enabled=True):
                rays_o,rays_d=data['rays_o'],data['rays_d']
                B, N = rays_o.shape[:2]
                # H, W = data['H'], data['W']
                p_bg, r_bg = get_evaluation_points_bg_uniform(rays_o,rays_d, di)

                rays_o_box, rays_d_box = transform_rays_to_box(rays_o, rays_d, transformations)
                p_fg, r_fg = get_evaluation_points_bg_uniform(rays_o_box,rays_d_box, di)
                
                sigmas1, rgbs1 = bg_trainer.model.forward(p_bg, r_bg)
                mask_box = torch.all(p_bg <= box_bound + padd, dim=-1) & torch.all(p_bg >= -box_bound - padd, dim=-1)
                # sigmas1[mask_box == 0] = 0.

                sigmas2, rgbs2,_ = fg_trainer.nerf.forward(p_fg, r_fg)
                mask_box = torch.all(p_fg <= box_bound + padd, dim=-1) & torch.all(p_fg >= -box_bound - padd, dim=-1)
                sigmas2[mask_box == 0] = 0.

                sigmas1 = sigmas1.reshape(batch_size, n_points, n_steps)
                rgbs1 = rgbs1.reshape(batch_size, n_points, n_steps, -1)
                sigmas2 = sigmas2.reshape(batch_size, n_points, n_steps)
                rgbs2 = rgbs2.reshape(batch_size, n_points, n_steps, -1)

                v1_4_latent_rgb_factors = torch.tensor([
                    #   R        G        B
                    [0.298, 0.207, 0.208],  # L1
                    [0.187, 0.286, 0.173],  # L2
                    [-0.158, 0.189, 0.264],  # L3
                    [-0.184, -0.271, -0.473],  # L4
                ], dtype=rgbs2.dtype, device=rgbs2.device)

                latent_image = rgbs2[0] @ v1_4_latent_rgb_factors
                # latents_ubyte = ((latent_image + 1) / 2).clamp(0, 1)
                latents_ubyte = ((latent_image + 1) / 2).clamp(0, 1)  # change scale from -1..1 to 0..1
                latents_ubyte=latents_ubyte.unsqueeze(0)  


                color.append(rgbs1)
                # color.append(rgbs2)
                color.append(latents_ubyte)
                sigma.append(sigmas1)
                sigmas2[sigmas2 < 1e-1] = 0.0
                sigma.append(sigmas2)

                sigma = F.relu(torch.stack(sigma, dim=0))
                color = torch.stack(color, dim=0)

                sigma_sum,color_sum=composite_function(sigma,color)
                weights=calc_volume_weights(di,rays_d,sigma_sum)
                color_map=torch.sum(weights.unsqueeze(-1)*color_sum,dim=-2)
                # print(weights.shape)
                sigma_sum=weights.reshape(1,H,W,512).permute(0,3,1,2).contiguous()
                # sigma_sum=weights.reshape(1,H,W,512).permute(0,3,1,2).contiguous()

                pred_rgb=color_map.reshape(-1, H, W, 3).permute(0, 3, 1, 2).contiguous()
                # pred1 = pred_rgb[0].detach().cpu().numpy()
                # pred1 = (pred1 * 255).astype(np.uint8)
                # all_preds1.append(pred1)
                
                # #input the pose of the camera
                # # bg_color = torch.rand((B * N, 3), device=rays_o.device)
                # bg_color = 1
                # outputs=fg_trainer.model.render(rays_o=rays_o,rays_d=rays_d,bg_color=bg_color)
                # # print(f"outputs.shape {outputs['image'].shape}")

                # pred_rgb = outputs['image'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                # pred_ws = outputs['weights_sum'].reshape(B, 1, H, W)

                # # pred_rgb=outputs['image'].reshape(1,160,160,3)
                # # pred_rgb=pred_rgb.permute(0,3,1,2).contiguous()
                # # alphas = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)

                ref_text="a detailed lego man wearing a kimono standing on a chair"
                # ref_text2="a detailed lego man"
                # ref_text="a blue German Sheperd standing on a chair"
                text_z = guidance.get_text_embeds([ref_text],[''])
                # text_z = guidance.get_text_embeds([ref_text])
                loss_guidance = guidance.train_step(text_z, pred_rgb)
                loss = loss_guidance
                loss += 5e-4 * losses['sparsity_loss'](sigma_sum)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if train_step % save_interval == 0:
                past_checkpoints=save_checkpoint(ckpt_path,train_step,past_checkpoints,bg_trainer,optimizer,scaler)
                evalue(bg_trainer,fg_trainer,data,train_renders_path,train_step,i,di,transformations,box_bound,padd)
                bg_trainer.model.train()
                fg_trainer.nerf.train()

    logger.info('Start Testing ^_^')
    bg_trainer.model.eval()
    fg_trainer.nerf.eval()
    with torch.no_grad():
        render(bg_trainer,fg_trainer,test_loader,transformations,downscale)

    # all_preds1=[]
    # with torch.no_grad():
    #     for i,data in enumerate(valid_loader):
    #         with torch.cuda.amp.autocast(enabled=True):
    #             rays_o,rays_d=data['rays_o'],data['rays_d']
    #             H, W = data['H'], data['W']
    #             # bg_color=torch.rand(25600,3).to(device)
    #             outputs=bg_trainer.model.render(rays_o,rays_d, staged=True)
    #             pred_rgb = outputs['image'].reshape(-1, H, W, 3)
    #             pred = pred_rgb[0].detach().cpu().numpy()
    #             pred = (pred * 255).astype(np.uint8)
    #             pred=tensor2numpy(pred_rgb[0])
    #             all_preds1.append(pred)
    # all_preds1 = np.stack(all_preds1, axis=0)
    # name1='video'
    # imageio.mimwrite(os.path.join(save_path, f'{name1}_rgb.mp4'), all_preds1, fps=25, quality=8, macro_block_size=1)
    logger.info('Finish Global Training ^_^')   
     
def render(bg_trainer,fg_trainer,dataloader,transformations,downscale):
    res=800//downscale
    device='cuda'
    n_steps=256
    n_points=res*res
    batch_size=1
    depth_range=[0.5, 6.]

    # Arange Pixels
    pixels = arange_pixels((res, res), batch_size, invert_y_axis=False)[1].to(device)
    pixels[..., -1] *= -1.

    di = depth_range[0] + torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (depth_range[1] - depth_range[0])
    di = di.repeat(batch_size, n_points, 1).to(device)
    box_bound=1
    padd = 0.1
    pre2=[]
    all_preds1 = []
    with torch.no_grad():
        for i,data in enumerate(dataloader):
            color, sigma = [], []

            rays_o,rays_d=data['rays_o'],data['rays_d']
            print(f"rays_o.shape{rays_o.shape}")
            p_bg, r_bg = get_evaluation_points_bg_uniform(rays_o,rays_d, di)
            
            rays_o_box, rays_d_box = transform_rays_to_box(rays_o, rays_d, transformations)
            p_fg, r_fg = get_evaluation_points_bg_uniform(rays_o_box,rays_d_box, di)
            
            sigmas1, rgbs1 = bg_trainer.model.forward(p_bg, r_bg)
            mask_box = torch.all(p_bg <= box_bound + padd, dim=-1) & torch.all(p_bg >= -box_bound - padd, dim=-1)
            # sigmas1[mask_box == 0] = 0.

            sigmas2, rgbs2,_ = fg_trainer.nerf.forward(p_fg, r_fg)
            mask_box = torch.all(p_fg <= box_bound + padd, dim=-1) & torch.all(p_fg >= -box_bound - padd, dim=-1)
            sigmas2[mask_box == 0] = 0.

            sigmas1 = sigmas1.reshape(batch_size, n_points, n_steps)
            rgbs1 = rgbs1.reshape(batch_size, n_points, n_steps, -1)
            sigmas2 = sigmas2.reshape(batch_size, n_points, n_steps)
            rgbs2 = rgbs2.reshape(batch_size, n_points, n_steps, -1)
            v1_4_latent_rgb_factors = torch.tensor([
                    #   R        G        B
                    [0.298, 0.207, 0.208],  # L1
                    [0.187, 0.286, 0.173],  # L2
                    [-0.158, 0.189, 0.264],  # L3
                    [-0.184, -0.271, -0.473],  # L4
                ], dtype=rgbs2.dtype, device=rgbs2.device)

            latent_image = rgbs2[0] @ v1_4_latent_rgb_factors
            # latents_ubyte = ((latent_image + 1) / 2).clamp(0, 1)
            latents_ubyte = ((latent_image + 1) / 2).clamp(0, 1)  # change scale from -1..1 to 0..1
            latents_ubyte=latents_ubyte.unsqueeze(0)  

            color.append(rgbs1)
            # color.append(rgbs2)
            color.append(latents_ubyte)

            sigma.append(sigmas1)
            sigma.append(sigmas2)
            sigmas2[sigmas2 < 1e-1] = 0.0

            sigma = F.relu(torch.stack(sigma, dim=0))
            color = torch.stack(color, dim=0)

            sigma_sum,color_sum=composite_function(sigma,color)
            weights=calc_volume_weights(di,rays_d,sigma_sum)

            color_map=torch.sum(weights.unsqueeze(-1)*color_sum,dim=-2)

            pred_rgb=color_map.reshape(-1, res, res, 3)
            pre2.append(pred_rgb[0])
            pred1 = pred_rgb[0].detach().cpu().numpy()
            pred1 = (pred1 * 255).astype(np.uint8)
            all_preds1.append(pred1)
        
    save_path='/data/hychen/final/step4.2/dog_bg'
    os.makedirs(save_path, exist_ok=True)

    all_preds1 = np.stack(all_preds1, axis=0)
    name1='0502'
    print(f"all_preds1.shape{all_preds1.shape}")
    imageio.mimwrite(os.path.join(save_path, f'{name1}_rgb.mp4'), all_preds1, fps=25, quality=8, macro_block_size=1)
    torch.save(pre2, '/data/hychen/final/step4.2/all_preds0424_bg_nograd.pt')
    for i, frame in enumerate(all_preds1):
        imageio.imwrite(os.path.join(save_path, f'{name1}_rgb_{i:04d}.png'), frame)

def init_losses():
    losses={}
    cur_nerf_loss={}
    cfg=TrainConfig()
    for i in range(1):
        cur_nerf_loss['shape_loss'] = ShapeLoss(cfg.guide)
        cur_nerf_loss['sparsity_loss']=sparsity_loss   
        losses[i]=cur_nerf_loss
    losses['sparsity_loss']=sparsity_loss
    return losses

def get_evaluation_points_bg_uniform(rays_o, rays_d, di):
    # rays_o 1 640000 3
    batch_size = rays_o.shape[0]
    # 1
    n_steps = di.shape[-1]
    # 64
    # print(f"rays_d{rays_d.shape}")
    # print(f"rays_o{rays_o.shape}")

    p = rays_o.unsqueeze(-2).contiguous() + \
        di.unsqueeze(-1).contiguous() * \
        rays_d.unsqueeze(-2).contiguous()
    r = rays_d.unsqueeze(-2).repeat(1, 1, n_steps, 1)
    assert(p.shape == r.shape)
    p = p.reshape(batch_size, -1, 3)
    r = r.reshape(batch_size, -1, 3)
    p = p.view(-1,3)
    r = r.view(-1,3)

    return p, r

def transform_rays_to_box(rays_o, rays_d, transformations, box_idx=0,
                        scale_factor=1.):
    bb_s, bb_t, bb_R = transformations
    rays_o_box = (bb_R[:, box_idx] @ (rays_o - bb_t[:, box_idx].unsqueeze(1)
                                    ).permute(0, 2, 1)).permute(
        0, 2, 1) / bb_s[:, box_idx].unsqueeze(1) * scale_factor
    rays_d_box = (bb_R[:, box_idx] @ rays_d.permute(
        0, 2, 1)).permute(0, 2, 1) / bb_s[:, box_idx].unsqueeze(1)

    return rays_o_box, rays_d_box

def composite_function(sigma, feat):
    n_boxes = sigma.shape[0]
    if n_boxes > 1: 
        denom_sigma = torch.sum(sigma, dim=0, keepdim=True)
        denom_sigma[denom_sigma == 0] = 1e-4
        w_sigma = sigma / denom_sigma
        sigma_sum = torch.sum(sigma, dim=0)
        feat_weighted = (feat * w_sigma.unsqueeze(-1)).sum(0)
    else:
        sigma_sum = sigma.squeeze(0)
        feat_weighted = feat.squeeze(0)
    return sigma_sum, feat_weighted

def calc_volume_weights(z_vals, ray_vector, sigma, last_dist=1e10):
    # print(f"z_vals.device{z_vals.device},ray_vector{ray_vector.device},sigma{sigma.device}")
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    #distance z_vals is the depth
    dists = torch.cat([dists, torch.ones_like(z_vals[..., :1]) * last_dist], dim=-1)
    dists = dists * torch.norm(ray_vector, dim=-1, keepdim=True)
    alpha = 1.-torch.exp(-F.relu(sigma)*dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :, :1]),(1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
    return weights
    
def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                  subsample_to=None, invert_y_axis=False):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = scale / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and
            subsample_to < n_points):
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                               replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    if invert_y_axis:
        assert(image_range == (-1, 1))
        pixel_scaled[..., -1] *= -1.
        pixel_locations[..., -1] = (h - 1) - pixel_locations[..., -1]

    return pixel_locations, pixel_scaled

def evalue(bg_trainer,fg_trainer,data,train_renders_path,train_step,i,di,transformations,box_bound,padd,**kwargs):
    bg_trainer.model.eval()
    fg_trainer.nerf.eval()
    

    n_steps=512
    batch_size=1
    rays_o,rays_d=data['rays_o'],data['rays_d']
    B, N = rays_o.shape[:2]
    # H, W = data['H'], data['W']
    H,W=64,64
    n_points=H*W 

    #input the pose of the camera
    color, sigma = [], []
    p_bg, r_bg = get_evaluation_points_bg_uniform(rays_o,rays_d, di)

    rays_o_box, rays_d_box = transform_rays_to_box(rays_o, rays_d, transformations)
    p_fg, r_fg = get_evaluation_points_bg_uniform(rays_o_box,rays_d_box, di)

    sigmas1, rgbs1 = bg_trainer.model.forward(p_bg, r_bg)
    mask_box = torch.all(p_bg <= box_bound + padd, dim=-1) & torch.all(p_bg >= -box_bound - padd, dim=-1)
    sigmas1[mask_box == 0] = 0.

    sigmas2, rgbs2,_ = fg_trainer.nerf.forward(p_fg, r_fg)
    mask_box = torch.all(p_fg <= box_bound + padd, dim=-1) & torch.all(p_fg >= -box_bound - padd, dim=-1)
    sigmas2[mask_box == 0] = 0.

    sigmas1 = sigmas1.reshape(batch_size, n_points, n_steps)
    rgbs1 = rgbs1.reshape(batch_size, n_points, n_steps, -1)
    sigmas2 = sigmas2.reshape(batch_size, n_points, n_steps)
    rgbs2 = rgbs2.reshape(batch_size, n_points, n_steps, -1)
    v1_4_latent_rgb_factors = torch.tensor([
                    #   R        G        B
                    [0.298, 0.207, 0.208],  # L1
                    [0.187, 0.286, 0.173],  # L2
                    [-0.158, 0.189, 0.264],  # L3
                    [-0.184, -0.271, -0.473],  # L4
                ], dtype=rgbs2.dtype, device=rgbs2.device)

    latent_image = rgbs2[0] @ v1_4_latent_rgb_factors
    # latents_ubyte = ((latent_image + 1) / 2).clamp(0, 1)
    latents_ubyte = ((latent_image + 1) / 2).clamp(0, 1)  # change scale from -1..1 to 0..1
    latents_ubyte=latents_ubyte.unsqueeze(0)  

    color.append(rgbs1)
    # color.append(rgbs2)
    color.append(latents_ubyte)

    sigma.append(sigmas1)
    sigma.append(sigmas2)
    sigmas2[sigmas2 < 1e-1] = 0.0

    sigma = F.relu(torch.stack(sigma, dim=0))
    color = torch.stack(color, dim=0)

    sigma_sum,color_sum=composite_function(sigma,color)
    weights=calc_volume_weights(di,rays_d,sigma_sum)

    color_map=torch.sum(weights.unsqueeze(-1)*color_sum,dim=-2)

    pred_rgb=color_map.reshape(-1, H, W, 3)
    # pre2.append(pred_rgb[0])
    # pred1 = pred_rgb[0].detach().cpu().numpy()
    # pred1 = (pred1 * 255).astype(np.uint8)

    # outputs=bg_trainer.model.render(rays_o,rays_d)
    # pred_rgb = outputs['image'].reshape(B, H, W, -1).contiguous()
    # pred_rgb_vis = pred_rgb.permute(0, 2, 3, 1).contiguous().clamp(0, 1)  #
    save_path = train_renders_path
    save_path.mkdir(exist_ok=True)
    pred = tensor2numpy(pred_rgb[0])
    Image.fromarray(pred).save(save_path / f"{train_step}_{i:04d}_evalue_rgb.png")
            
def save_checkpoint(ckpt_path,train_step,past_checkpoints,bg_trainer,optimizer,scaler):
    # ckpt_path=Path('/data/hychen/ln_data/SDS/checkpoints')
    name = f'step_{train_step:06d}'
    state = {
        'train_step': train_step,
        'checkpoints': past_checkpoints,
    }
    state['mean_count'] = bg_trainer.model.mean_count
    state['mean_density'] = bg_trainer.model.mean_density
    state['optimizer'] = optimizer.state_dict()
    state['scaler'] = scaler.state_dict()
    state['model'] = bg_trainer.model.state_dict()

    file_path = f"{name}.pth"
    past_checkpoints.append(file_path)
    if len(past_checkpoints) > 2:
        old_ckpt = ckpt_path / past_checkpoints.pop(0)
        old_ckpt.unlink(missing_ok=True)
    torch.save(state, ckpt_path / file_path)
    return past_checkpoints

def log_train_renders(pred_rgbs: torch.Tensor,train_step,train_renders_path):
    pred_rgb_vis = pred_rgbs.permute(0, 2, 3,
                                            1).contiguous().clamp(0, 1)  #
    save_path = train_renders_path / f'step_{train_step:05d}.jpg'
    save_path.parent.mkdir(exist_ok=True)
    pred = tensor2numpy(pred_rgb_vis[0])
    Image.fromarray(pred).save(save_path)

def tensor2numpy(tensor:torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

#projecting 3D coordinates onto a 2D image plane
def get_camera_mat_from_intrinsics(intrinsics,invert=True):
    fx, fy, cx, cy = intrinsics
    # cx,cy=100,100
    mat = torch.tensor([
        [fx, 0., cx, 0.],
        [0., fy, cy, 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ],dtype=torch.float32).reshape(1, 4, 4)
    if invert:
        mat = torch.inverse(mat)
    return mat

def write_video( out_file, img_list, n_row=5, add_reverse=False,
                write_small_vis=True):
    n_steps, batch_size = img_list.shape[:2]
    nrow = n_row if (n_row is not None) else int(sqrt(batch_size))
    img = [(255*make_grid(img, nrow=nrow, pad_value=1.).permute(
        1, 2, 0)).cpu().numpy().astype(np.uint8) for img in img_list]
    if add_reverse:
        img += list(reversed(img))
    imageio.mimwrite(out_file, img, fps=30, quality=8)
    if write_small_vis:
        img = [(255*make_grid(img, nrow=batch_size, pad_value=1.).permute(
            1, 2, 0)).cpu().numpy().astype(
                np.uint8) for img in img_list[:, :9]]
        if add_reverse:
            img += list(reversed(img))
        imageio.mimwrite(
            (out_file[:-4] + '_sm.mp4'), img, fps=30, quality=4)

def save_video_and_images(imgs, out_folder, name='rotation_object',
                            is_full_rotation=False, img_n_steps=6,
                            add_reverse=False):

    # Save video
    out_file_video = os.path.join(out_folder, '%s.mp4' % name)
    write_video(out_file_video, imgs, add_reverse=add_reverse)

    # Save images
    n_steps, batch_size = imgs.shape[:2]
    if is_full_rotation:
        idx_paper = np.linspace(
            0, n_steps - n_steps // img_n_steps, img_n_steps
        ).astype(np.int)
    else:
        idx_paper = np.linspace(0, n_steps - 1, img_n_steps).astype(np.int)
    for idx in range(batch_size):
        img_grid = imgs[idx_paper, idx]
        save_image(make_grid(
            img_grid, nrow=img_n_steps, pad_value=1.), os.path.join(
                out_folder, '%04d_%s.jpg' % (idx, name)))
    
def check_coordinate_in_object(p,object_bounds):
    for i in range(3):
        if p[i] < object_bounds[0][i] or p[i] > object_bounds[1][i]:
            return False
    return True

if __name__ == '__main__':
    main()