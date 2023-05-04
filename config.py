from dataclasses import dataclass, field
from typing import List

@dataclass
class bg_config:
    # path: str = "/data/hychen/counter"
    # path: str="/data/hychen/garden"
    path: str="/data/hychen/nerf_synthetic/chair"

    # path: str="/home/hychen/instant-ngp/nerf"
    # path:str="/data/hychen/images1"
    # path:str='/data/hychen/scene0004_00/output1'
    # path:str='/data/hychen/scene0004_00/output2'
    # path: str = "/home/hychen/instant-ngp/nerf2"
    # path: str = '/data/hychen/scene0004_00/output1'
    test: bool = False
    # workspace: str = "/data/hychen/ln_data/chair_scene2"
    workspace: str = "/data/hychen/final/step1_counter"

    seed: int = 0
    iters: int = 30000
    lr: float = 1e-2
    ckpt: str = "latest"
    num_rays: int = 4096
    cuda_ray: bool = True
    max_steps: int = 1024
    #sample number
    num_steps: int = 512
    upsample_steps: int = 0
    update_extra_interval: int = 16
    max_ray_batch: int = 4096
    patch_size: int = 1
    # fp16: bool = True
    fp16: bool = False
    # ff: bool = False
    # tcnn: bool = False
    color_space: str = "srgb"
    preload: bool = True
    bound: float = 2
    scale: float = 0.33
    offset: List[float] = field(default_factory=lambda: [0, 0, 0])
    dt_gamma: float = 1/128
    min_near: float = 0.2
    density_thresh: float = 10
    bg_radius: float = -1
    error_map: bool = False
    clip_text: str = ""
    rand_pose: int = -1

@dataclass
class fg_config:
    # text: str = 'a detailed dog wearing a purple hoodie'
    text: str = "A German Sheperd"
    negative: str = ''
    guidance: str = 'stable-diffusion'
    
    # optimization options
    O: bool = False
    O2: bool = False
    test: bool = False
    eval_interval: int = 1
    # workspace: str = '/data/hychen/ln_data/lego_0416'
    workspace: str='/data/hychen/final/step2.1/dog'
    seed: int = 4
    
    # mesh export options
    save_mesh: bool = False
    mcubes_resolution: int = 256
    decimate_target: int = 100000
    
    # training options
    iters: int = 10000
    lr: float = 1e-3
    warm_iters: int = 500
    min_lr: float = 1e-4
    ckpt: str = 'latest'
    cuda_ray: bool = True
    taichi_ray: bool = False
    max_steps: int = 1024
    # num_steps: int = 64
    num_steps: int = 256
    upsample_steps: int = 32
    update_extra_interval: int = 16
    max_ray_batch: int = 4096
    albedo: bool = True
    albedo_iters: int = 1000
    jitter_pose: bool = False
    uniform_sphere_rate: float = 0.5
    
    # model options
    # bg_radius: float = -1
    bg_radius: float = 1.4

    density_activation: str = 'softplus'
    density_thresh: float = 0.1
    blob_density: float = 10
    blob_radius: float = 0.5
    backbone: str = 'grid'
    optim: str = 'adam'
    sd_version: str = '2.1'
    hf_key: str = None
    fp16: bool = True
    vram_O: bool = True
    w: int = 64
    h: int = 64
    
    # dataset options
    bound: float = 1
    dt_gamma: float = 0
    min_near: float = 0.1
    radius_range: List[float] = field(default_factory=lambda: [1.0, 1.5])
    fovy_range: List[float] = field(default_factory=lambda: [40, 70])
    dir_text: bool = True
    suppress_face: bool = False
    angle_overhead: float = 30
    angle_front: float = 60
    # lambda_entropy: float = 1e-4
    lambda_entropy: float = 5e-4
    lambda_opacity: float = 0
    lambda_orient: float = 1e-2
    lambda_tv: float = 0
    gui: bool = False
    W: int = 800
    H: int = 800
    radius: float = 3
    fovy: float = 60
    light_theta: float = 60
    light_phi: float = 0
    max_spp: int = 1

    lambda_sparsity: float=5e-4
    lambda_shape: float = 5e-6