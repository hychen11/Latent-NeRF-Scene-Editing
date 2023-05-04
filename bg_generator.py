import torch
from bg_nerf.provider import bg_NeRFDataset
from bg_nerf.utils import *
from bg_nerf.network import bg_NeRFNetwork
from config import bg_config
from torch.utils.tensorboard import SummaryWriter

def bg_generator():
    opt=bg_config()
    # print(opt)
    seed_everything(opt.seed)

    model = bg_NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )
    print(model)

    # input_shape1 = (1, 3)  # (batch_size, input_dim) for the GridEncoder
    # input_shape2 = (1, 3)  # (batch_size, input_dim) for the SHEncoder

    # dummy_input1 = torch.zeros(input_shape1)
    # dummy_input2 = torch.zeros(input_shape2)

    # # Move the dummy inputs to the GPU
    # device = torch.device('cuda')
    # dummy_input1 = dummy_input1.to(device)
    # dummy_input2 = dummy_input2.to(device)
    # print(dummy_input1.type())
    # print(dummy_input2.type())

    # writer = SummaryWriter('runs/network_visualization')
    # writer.add_graph(model.render, (dummy_input1, dummy_input2))
    # writer.close()

    criterion = torch.nn.MSELoss(reduction='none')
    device = torch.device('cuda')

    # visible_DEVICE
    optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    #load dataset
    train_loader = bg_NeRFDataset(opt, device=device, type='train').dataloader()
    
    valid_loader = bg_NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()
    test_loader = bg_NeRFDataset(opt, device=device, type='test').dataloader()


    metrics = [PSNRMeter(), LPIPSMeter(device=device)]
    bg_trainer = bg_Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50)
    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    print(f"max_epoch {max_epoch}")
    # return
    # bg_trainer.train(train_loader, valid_loader, max_epoch)
    bg_trainer.train(train_loader=train_loader,valid_loader=valid_loader, max_epochs=max_epoch)

    # print("finished training")
    
# also test
    #load camera intrinsics
    # intrinsics = test_loader.intrinsics

    # if test_loader.has_gt:
    #     bg_trainer.evaluate(test_loader) # blender has gt, so evaluate it.

    # #render process
    bg_trainer.test(test_loader, write_video=True) # test and save video

# trainer.save_mesh(resolution=256, threshold=10)