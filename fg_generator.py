
import sys
import torch
import argparse
from fg_nerf.provider import fg_NeRFDataset
from fg_nerf.utils import *
# from fg_nerf.network_grid import fg_NeRFNetwork
from fg_nerf.network import fg_NeRFNetwork

# from src.stable_diffusion import StableDiffusion
from sd import StableDiffusion

# from fg_nerf.gui import NeRFGUI
import os
from config import fg_config

def fg_generator():
    opt=fg_config()
    print(opt)

    seed_everything(opt.seed)
    model = fg_NeRFNetwork(opt)
    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = fg_NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()
    valid_loader = fg_NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()
    test_loader = fg_NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=200).dataloader()
    
    # Adan usually requires a larger LR
    optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
    # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    # guidance = StableDiffusion(device,model_name='CompVis/stable-diffusion-v1-4',latent_mode=False)
    # guidance=StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)
    guidance = StableDiffusion(device)
    trainer = fg_Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)

    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    trainer.train(train_loader, valid_loader, max_epoch)
    # fg_checkpoint_path='/data/hychen/stable-dreamfusion-main/workspace_noshape/checkpoints/df_ep0100.pth'
    # trainer.load_checkpoint(fg_checkpoint_path)

    trainer.test(test_loader)


