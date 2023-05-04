import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
from util.common import (
    arange_pixels, image_points_to_world, origin_to_world
)
import numpy as np
import imageio
import os
from scipy.spatial.transform import Rotation as Rot
from util.camera import get_camera_mat, get_random_pose, get_camera_pose
from train_latent_nerf import check_coordinate_in_object
from bg_nerf.provider import bg_NeRFDataset
from bg_nerf.network import NeRFNetwork1
from bg_nerf.utils import Trainer,get_rays
from config import bg_config
import trimesh
import matplotlib.pyplot as plt
import src.latent_nerf.raymarching.raymarchingrgb as raymarching
from src.latent_nerf.training.trainer import Trainer
from src.latent_nerf.models.network_grid import NeRFNetwork
from src.latent_nerf.configs.render_config import RenderConfig
from src.latent_nerf.configs.train_config import TrainConfig
from src.latent_nerf.models.render_utils import safe_normalize

def visualize_ray_directions1(rays_o, rays_d):
    torch.save({"o":rays_o,"d":rays_d},"./1111")
    # dirs = rays_d[:,0:-1:50,:].cpu().numpy()
    # origins = rays_o[:,0:-1:50,:].cpu().numpy()
    dirs = rays_d[:,0:-1:50,:].cpu().numpy()
    origins = rays_o[:,0:-1:50,:].cpu().numpy()
    # [1, 640000, 3]
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(
    origins[..., 0].flatten(),
    origins[..., 1].flatten(),
    origins[..., 2].flatten(),
    dirs[..., 0].flatten(),
    dirs[..., 1].flatten(),
    dirs[..., 2].flatten(), length=0.5, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')
    plt.savefig('t1.png')

def visualize_ray_directions2(rays_o, rays_d):
    torch.save({"o":rays_o,"d":rays_d},"./2222")
    # dirs = rays_d[:,0:-1:50,:].cpu().numpy()
    # origins = rays_o[:,0:-1:50,:].cpu().numpy()
    dirs = rays_d[:,0:-1:50,:].cpu().numpy()
    origins = rays_o[:,0:-1:50,:].cpu().numpy()
    # [1, 640000, 3]
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(
    origins[..., 0].flatten(),
    origins[..., 1].flatten(),
    origins[..., 2].flatten(),
    dirs[..., 0].flatten(),
    dirs[..., 1].flatten(),
    dirs[..., 2].flatten(), length=0.5, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')
    plt.savefig('t2.png')

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),accumulated_transmittance[:, :-1]), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=2, hf=6, nb_bins=192):
    device = ray_origins.device
    #expand将1维tensor replicate to nb_bins 维
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    # δi = ti+1 − ti is the distance between adjacent samples.
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
    print(x.shape)
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])
    print(f"colors shape:{colors.shape}")
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  # Pixel values
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)

class Generator(nn.Module):
    ''' GIRAFFE Generator Class.

    Args:
        device (pytorch device): pytorch device
        z_dim (int): dimension of latent code z
        z_dim_bg (int): dimension of background latent code z_bg
        decoder (nn.Module): decoder network
        range_u (tuple): rotation range (0 - 1)
        range_v (tuple): elevation range (0 - 1)
        n_ray_samples (int): number of samples per ray
        range_radius(tuple): radius range
        depth_range (tuple): near and far depth plane
        background_generator (nn.Module): background generator
        bounding_box_generator (nn.Module): bounding box generator
        resolution_vol (int): resolution of volume-rendered image
        neural_renderer (nn.Module): neural renderer
        fov (float): field of view
        background_rotation_range (tuple): background rotation range
         (0 - 1)
        sample_object-existance (bool): whether to sample the existance
            of objects; only used for clevr2345
        use_max_composition (bool): whether to use the max
            composition operator instead
    '''

    def __init__(self, device='cuda', z_dim=256, z_dim_bg=128, decoder=None,
                 range_u=(0, 0), range_v=(0.25, 0.25), n_ray_samples=64,
                 range_radius=(2.732, 2.732), depth_range=[0.5, 6.],
                 background_generator=None,
                 bounding_box_generator=None, resolution_vol=16,
                 neural_renderer=None,
                 fov=49.13,
                 backround_rotation_range=[0., 0.],
                 sample_object_existance=False,
                 loader=None,
                 use_max_composition=False, **kwargs):
        super().__init__()
        self.device = device
        self.n_ray_samples = n_ray_samples
        self.range_u = range_u
        self.range_v = range_v
        self.resolution_vol = resolution_vol
        self.range_radius = range_radius
        self.depth_range = depth_range
        self.bounding_box_generator = bounding_box_generator
        self.fov = fov
        self.backround_rotation_range = backround_rotation_range
        self.sample_object_existance = sample_object_existance
        self.z_dim = z_dim
        self.z_dim_bg = z_dim_bg
        self.use_max_composition = use_max_composition
        # self.loader=loader

        #get unique camera_matrix
        self.camera_matrix = get_camera_mat(fov=fov).to(device)

        if decoder is not None:
            self.decoder = decoder.to(device)
        else:
            self.decoder = None

        if background_generator is not None:
            self.background_generator = background_generator.to(device)
        else:
            self.background_generator = None
        if bounding_box_generator is not None:
            self.bounding_box_generator = bounding_box_generator.to(device)
        else:
            self.bounding_box_generator = bounding_box_generator
        if neural_renderer is not None:
            self.neural_renderer = neural_renderer.to(device)
        else:
            self.neural_renderer = None

    def forward(self, batch_size=32, latent_codes=None, camera_matrices=None,
                transformations=None, bg_rotation=None, mode="training", it=0,
                return_alpha_map=False,
                not_render_background=False,
                loader=None,
                only_render_background=False):
    
        if latent_codes is None:
            latent_codes = self.get_latent_codes(batch_size)

        if camera_matrices is None:
            camera_matrices = self.get_random_camera(batch_size)

        if transformations is None:
            transformations = self.get_random_transformations(batch_size)

        if bg_rotation is None:
            bg_rotation = self.get_random_bg_rotation(batch_size)

        if return_alpha_map:
            rgb_v, alpha_map = self.volume_render_image(
                latent_codes, camera_matrices, transformations, bg_rotation,
                mode=mode, it=it, return_alpha_map=True,
                not_render_background=not_render_background)
            return alpha_map
        else:
            print("use volume_render_image")
            rgb_v = self.volume_render_image(
                latent_codes, camera_matrices, transformations, bg_rotation,
                mode=mode, it=it, not_render_background=not_render_background,
                only_render_background=only_render_background,loader=loader)
            if self.neural_renderer is not None:
                rgb = self.neural_renderer(rgb_v)
            else:
                rgb = rgb_v
            return rgb

        # print("use volume_render_image")
        # rgb_v = self.volume_render_image(
        #     latent_codes, camera_matrices, transformations, bg_rotation,
        #     mode=mode, it=it, not_render_background=not_render_background,
        #     only_render_background=only_render_background)
        # return rgb_v
    
    def get_n_boxes(self):
        if self.bounding_box_generator is not None:
            n_boxes = self.bounding_box_generator.n_boxes
        else:
            n_boxes = 1
        return n_boxes

    def get_latent_codes(self, batch_size=32, tmp=1.):
        z_dim, z_dim_bg = self.z_dim, self.z_dim_bg

        n_boxes = self.get_n_boxes()

        def sample_z(x): return self.sample_z(x, tmp=tmp)
        z_shape_obj = sample_z((batch_size, n_boxes, z_dim))
        z_app_obj = sample_z((batch_size, n_boxes, z_dim))
        z_shape_bg = sample_z((batch_size, z_dim_bg))
        z_app_bg = sample_z((batch_size, z_dim_bg))

        return z_shape_obj, z_app_obj, z_shape_bg, z_app_bg

    def sample_z(self, size, to_device=True, tmp=1.):
        z = torch.randn(*size) * tmp
        if to_device:
            z = z.to(self.device)
        return z

    def get_vis_dict(self, batch_size=32):
        vis_dict = {
            'batch_size': batch_size,
            'latent_codes': self.get_latent_codes(batch_size),
            'camera_matrices': self.get_random_camera(batch_size),
            'transformations': self.get_random_transformations(batch_size),
            'bg_rotation': self.get_random_bg_rotation(batch_size)
        }
        return vis_dict

    def get_random_camera(self, batch_size=32, to_device=True):
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat = get_random_pose(
            self.range_u, self.range_v, self.range_radius, batch_size)
        if to_device:
            world_mat = world_mat.to(self.device)
        return camera_mat, world_mat

    def get_camera(self, val_u=0.5, val_v=0.5, val_r=0.5, batch_size=32,
                   to_device=True):
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat = get_camera_pose(
            self.range_u, self.range_v, self.range_radius, val_u, val_v,
            val_r, batch_size=batch_size)
        if to_device:
            world_mat = world_mat.to(self.device)
        return camera_mat, world_mat

    def get_random_bg_rotation(self, batch_size, to_device=True):
        if self.backround_rotation_range != [0., 0.]:
            bg_r = self.backround_rotation_range
            r_random = bg_r[0] + np.random.rand() * (bg_r[1] - bg_r[0])
            R_bg = [
                torch.from_numpy(Rot.from_euler(
                    'z', r_random * 2 * np.pi).as_matrix()
                ) for i in range(batch_size)]
            R_bg = torch.stack(R_bg, dim=0).reshape(
                batch_size, 3, 3).float()
        else:
            R_bg = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        if to_device:
            R_bg = R_bg.to(self.device)
        return R_bg

    def get_bg_rotation(self, val, batch_size=32, to_device=True):
        if self.backround_rotation_range != [0., 0.]:
            bg_r = self.backround_rotation_range
            r_val = bg_r[0] + val * (bg_r[1] - bg_r[0])
            r = torch.from_numpy(
                Rot.from_euler('z', r_val * 2 * np.pi).as_matrix()
            ).reshape(1, 3, 3).repeat(batch_size, 1, 1).float()
        else:
            r = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        if to_device:
            r = r.to(self.device)
        return r

    def get_random_transformations(self, batch_size=32, to_device=True):
        device = self.device
        s, t, R = self.bounding_box_generator(batch_size)
        if to_device:
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def get_transformations(self, val_s=[[0.5, 0.5, 0.5]],
                            val_t=[[0.5, 0.5, 0.5]], val_r=[0.5],
                            batch_size=32, to_device=True):
        device = self.device
        s = self.bounding_box_generator.get_scale(
            batch_size=batch_size, val=val_s)
        t = self.bounding_box_generator.get_translation(
            batch_size=batch_size, val=val_t)
        R = self.bounding_box_generator.get_rotation(
            batch_size=batch_size, val=val_r)

        if to_device:
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def get_transformations_in_range(self, range_s=[0., 1.], range_t=[0., 1.],
                                     range_r=[0., 1.], n_boxes=1,
                                     batch_size=32, to_device=True):
        s, t, R = [], [], []

        def rand_s(): return range_s[0] + \
            np.random.rand() * (range_s[1] - range_s[0])

        def rand_t(): return range_t[0] + \
            np.random.rand() * (range_t[1] - range_t[0])
        def rand_r(): return range_r[0] + \
            np.random.rand() * (range_r[1] - range_r[0])

        for i in range(batch_size):
            val_s = [[rand_s(), rand_s(), rand_s()] for j in range(n_boxes)]
            val_t = [[rand_t(), rand_t(), rand_t()] for j in range(n_boxes)]
            val_r = [rand_r() for j in range(n_boxes)]
            si, ti, Ri = self.get_transformations(
                val_s, val_t, val_r, batch_size=1, to_device=to_device)
            s.append(si)
            t.append(ti)
            R.append(Ri)
        s, t, R = torch.cat(s), torch.cat(t), torch.cat(R)
        if to_device:
            device = self.device
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def get_rotation(self, val_r, batch_size=32, to_device=True):
        device = self.device
        R = self.bounding_box_generator.get_rotation(
            batch_size=batch_size, val=val_r)

        if to_device:
            R = R.to(device)
        return R

    def add_noise_to_interval(self, di):
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti

    def transform_points_to_box(self, p, transformations, box_idx=0,
                                scale_factor=1.):
        bb_s, bb_t, bb_R = transformations
        p_box = (bb_R[:, box_idx] @ (p - bb_t[:, box_idx].unsqueeze(1)
                                     ).permute(0, 2, 1)).permute(
            0, 2, 1) / bb_s[:, box_idx].unsqueeze(1) * scale_factor
        return p_box
    
    def transform_rays_to_box(self, rays_o, rays_d, transformations, box_idx=0,
                            scale_factor=1.):
        bb_s, bb_t, bb_R = transformations
        rays_o_box = (bb_R[:, box_idx] @ (rays_o - bb_t[:, box_idx].unsqueeze(1)
                                        ).permute(0, 2, 1)).permute(
            0, 2, 1) / bb_s[:, box_idx].unsqueeze(1) * scale_factor
        rays_d_box = (bb_R[:, box_idx] @ rays_d.permute(
            0, 2, 1)).permute(0, 2, 1) / bb_s[:, box_idx].unsqueeze(1)
        # # print(f"rays_d_box.shape,rays_o_box.shape{rays_d_box.shape,rays_o_box.shape}")
        # rays_o_box = rays_o_box[:, :, [1, 0, 2]]  # 交换第0和第1维
        # rays_o_box[:, :, 2] *= -1  # 反转第2维
        # rays_d_box = rays_d_box[:, :, [1, 0, 2]]  # 交换第0和第1维
        # rays_d_box[:, :, 2] *= -1  # 反转第2维
        return rays_o_box, rays_d_box
    


    def get_evaluation_points_bg(self, pixels_world, camera_world, di,
                                 rotation_matrix):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]
        # rotation_matrix=rotation_matrix.to(self.device)
        # print(f"deivce{rotation_matrix.device},{camera_world.device},{pixels_world.device}")
        camera_world = (rotation_matrix @
                        camera_world.permute(0, 2, 1)).permute(0, 2, 1)
        pixels_world = (rotation_matrix @
                        pixels_world.permute(0, 2, 1)).permute(0, 2, 1)
        ray_world = pixels_world - camera_world
        
        p = camera_world.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * \
            ray_world.unsqueeze(-2).contiguous()
        r = ray_world.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert(p.shape == r.shape)
        p = p.reshape(batch_size, -1, 3)
        r = r.reshape(batch_size, -1, 3)
        return p, r

    def get_evaluation_points_bg_uniform(self, rays_o, rays_d, di):
        # rays_o 1 640000 3
        batch_size = rays_o.shape[0]
        # 1
        n_steps = di.shape[-1]
        # 64

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
        
    def get_evaluation_points(self, pixels_world, camera_world, di,
                              transformations, i):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        #both in box coordinate
        pixels_world_i = self.transform_points_to_box(
            pixels_world, transformations, i)
        camera_world_i = self.transform_points_to_box(
            camera_world, transformations, i)
        ray_i = pixels_world_i - camera_world_i

        p_i = camera_world_i.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * ray_i.unsqueeze(-2).contiguous()
        ray_i = ray_i.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert(p_i.shape == ray_i.shape)
        # print()
        p_i = p_i.reshape(batch_size, -1, 3)
        ray_i = ray_i.reshape(batch_size, -1, 3)

        return p_i, ray_i

    def composite_function(self, sigma, feat):
        n_boxes = sigma.shape[0]
        if n_boxes > 1:
            if self.use_max_composition:
                bs, rs, ns = sigma.shape[1:]
                sigma_sum, ind = torch.max(sigma, dim=0)
                feat_weighted = feat[ind, torch.arange(bs).reshape(-1, 1, 1),
                                     torch.arange(rs).reshape(
                                         1, -1, 1), torch.arange(ns).reshape(
                                             1, 1, -1)]
            else:
                denom_sigma = torch.sum(sigma, dim=0, keepdim=True)
                denom_sigma[denom_sigma == 0] = 1e-4
                w_sigma = sigma / denom_sigma
                sigma_sum = torch.sum(sigma, dim=0)
                feat_weighted = (feat * w_sigma.unsqueeze(-1)).sum(0)
        else:
            sigma_sum = sigma.squeeze(0)
            feat_weighted = feat.squeeze(0)
        return sigma_sum, feat_weighted

    def calc_volume_weights(self, z_vals, ray_vector, sigma, last_dist=1e10):
        # print(f"z_vals.device{z_vals.device},ray_vector{ray_vector.device},sigma{sigma.device}")
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        #distance z_vals is the depth
        dists = torch.cat([dists, torch.ones_like(z_vals[..., :1]) * last_dist], dim=-1)
        dists = dists * torch.norm(ray_vector, dim=-1, keepdim=True)
        alpha = 1.-torch.exp(-F.relu(sigma)*dists)
        weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :, :1]),(1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
        return weights

    def get_object_existance(self, n_boxes, batch_size=32):
        '''
        Note: We only use this setting for Clevr2345, so that we can hard-code
        the probabilties here. If you want to apply it to a different scenario,
        you would need to change these.
        '''
        probs = [
            .19456788355146545395,
            .24355003312266127155,
            .25269546846185522711,
            .30918661486401804737,
        ]

        n_objects_prob = np.random.rand(batch_size)
        n_objects = np.zeros_like(n_objects_prob).astype(np.int)
        p_cum = 0
        obj_n = [i for i in range(2, n_boxes + 1)]
        for idx_p in range(len(probs)):
            n_objects[
                (n_objects_prob >= p_cum) &
                (n_objects_prob < p_cum + probs[idx_p])
            ] = obj_n[idx_p]
            p_cum = p_cum + probs[idx_p]
            assert(p_cum <= 1.)

        object_existance = np.zeros((batch_size, n_boxes))
        for b_idx in range(batch_size):
            n_obj = n_objects[b_idx]
            if n_obj > 0:
                idx_true = np.random.choice(
                    n_boxes, size=(n_obj,), replace=False)
                object_existance[b_idx, idx_true] = True
        object_existance = object_existance.astype(np.bool)
        return object_existance

    def volume_render_image(self, latent_codes, camera_matrices,
                            transformations, bg_rotation, mode='training',
                            it=0, return_alpha_map=False,
                            not_render_background=False,
                            only_render_background=False,
                            loader=None):
        res = self.resolution_vol
        device = self.device
        n_steps = self.n_ray_samples
        n_points = res * res
        depth_range = self.depth_range
        batch_size = latent_codes[0].shape[0]
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = latent_codes
        assert(not (not_render_background and only_render_background))

        #modify some parameters
        downscale=5
        res=800//downscale
        device='cuda'
        n_steps=512
        n_points=res*res
        batch_size=1
        # depth_range=[2,6]

        # Arange Pixels
        pixels = arange_pixels((res, res), batch_size, invert_y_axis=False)[1].to(device)
        pixels[..., -1] *= -1.

        # # batch_size x n_points x n_steps
        # get uniform sample points (depth)
        di = depth_range[0] + torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (depth_range[1] - depth_range[0])
        di = di.repeat(batch_size, n_points, 1).to(device)
        
        box_bound=1
        padd = 0.1

        # #Render Object
        # #transfer scene coordinate to obj coordinate
        # p_i, r_i = self.get_evaluation_points(
        #     pixels_world, camera_world, di, transformations, i=0)
        # # print(f"p_i,r_i:{p_i,r_i} p_i.shape {p_i.shape} r_i.shape{r_i.shape}")
        # # p_i.shape torch.Size([200, 262144, 3]) r_i.shapetorch.Size([200, 262144, 3])
        # #1*4096*64

        # #predict the rgb and density from xyz
        # #assume the box range is [-0.5,-0.5,-0.5] to [0.5,0.5,0.5]


        # feat_fg, sigma_fg = self.judge(p_i, r_i,bound)

        # Mask out values outside
        # color_shape = [batch_size, n_points*n_steps, 3]
        # sigma_shape= [batch_size, n_points*n_steps, 1]
        # color_fg = torch.zeros(color_shape)
        # sigma_fg = torch.zeros(sigma_shape)

        # #################################################
        # # latent nerf render ! first we assume it is in mask box == color red and density 1
        # mask_box = torch.all(p_i <= bound, dim=-1) & torch.all(p_i >= -bound, dim=-1)
        # mask_box = torch.all(
        #     p_i <= box_bound + padd, dim=-1) & torch.all(
        #         p_i >= -box_bound - padd, dim=-1)
        # # sigma_i[mask_box == 0] = 0.
        # # print(f"mask_box {mask_box.sum()}")
        # sigma_i[mask_box == 0] = 0.
        # # print(f"1:{mask_box},{mask_box.shape}")
        
        
        # color_fg[mask_box] = torch.tensor([1.0, 0.0, 0.0])
        # sigma_fg[mask_box] = 1.0
        # #################################################
        # # print(f"color_fg.shape{color_fg.shape}")
        # # print(f"sigma_fg.shape{sigma_fg.shape}")

        # # Reshape
        # sigma_fg = sigma_fg.reshape(batch_size, n_points, n_steps)
        # color_fg = color_fg.reshape(batch_size, n_points, n_steps, -1)
        # # print(f"color_fg.shape{color_fg.shape}")
        # # print(f"sigma_fg.shape{sigma_fg.shape}")

        #bg render
        # p_bg, r_bg = self.get_evaluation_points_bg_uniform(
        #             pixels_world, camera_world, di)

        # result = get_rays( loader.poses, loader.intrinsics, res, res)
        # p_bg=result['rays_o']
        # r_bg=result['rays_d']
        
        # print(f"p_bg,r_bg:{p_bg,r_bg} p_bg.shape {p_bg.shape} r_bg.shape{r_bg.shape}")

        # print(f"p_bg.shape {p_bg.shape} r_bg.shape{r_bg.shape}")
        # # p_bg.shape torch.Size([200, 262144, 3]) r_bg.shapetorch.Size([200, 262144, 3])


        # color_fg = torch.zeros(color_shape)
        # sigma_fg = torch.zeros(sigma_shape)
        #predict the rgb and density from xyz
        #################################################
        #ngp render!!
        bg_opt=bg_config()
        fg_opt_render=RenderConfig()
        fg_opt_Train=TrainConfig()
        # test_loader = bg_NeRFDataset(opt, device=device, type='test').dataloader()

        all_preds = []
        all_preds_depth = []

        all_preds1 = []
        all_preds_depth1 = []

        all_preds2 = []


        #load background model
        model_bg = NeRFNetwork1(
            encoding="hashgrid",
            bound=bg_opt.bound,
            cuda_ray=bg_opt.cuda_ray,
            density_scale=1,
            min_near=bg_opt.min_near,
            density_thresh=bg_opt.density_thresh,
            bg_radius=bg_opt.bg_radius,
        )
        bg_checkpoint_path='/data/hychen/ln_data/test/checkpoints/ngp_ep0300.pth'
        bg_trainer = Trainer(name='ngp' , opt=bg_opt, model=model_bg, device='cuda',use_checkpoint="chy")
        bg_trainer.model.load_state_dict(torch.load(bg_checkpoint_path)['model'],strict=False)
        # bg_trainer.test(loader,save_path='/data/hychen/ln_data/test/results',write_video=True)
        
        #load foreground model
        model_fg=  NeRFNetwork(fg_opt_render).to('cuda')
        fg_checkpoint_path='/data/hychen/ln_data/experiments/lego_man/checkpoints/step_005000.pth'
        fg_trainer = Trainer(cfg=fg_opt_Train)
        fg_trainer.nerf=model_fg
        # fg_trainer.model.load_state_dict(torch.load(fg_checkpoint_path)['model'],strict=False)
        fg_trainer.load_checkpoint(fg_checkpoint_path)
        fg_trainer.full_eval()

        print("finish load bg_model and bg_model")

        with torch.no_grad():
            for i,data in enumerate(loader):
                print(i)
                # if i>50: continue
                color, sigma = [], []

                rays_o,rays_d=data['rays_o'],data['rays_d']
                p_bg, r_bg = self.get_evaluation_points_bg_uniform(rays_o,rays_d, di)
                
                # rays_o_box, rays_d_box = rays_o,rays_d
                rays_o_box, rays_d_box = self.transform_rays_to_box(rays_o, rays_d, transformations)
                p_fg, r_fg = self.get_evaluation_points_bg_uniform(rays_o_box,rays_d_box, di)
                
                # color_bg,sigma_bg=self.decoder(p_bg,r_bg)
                # color_bg=color_bg.reshape(batch_size,n_points,n_steps,-1)
                # sigma_bg=sigma_bg.reshape(batch_size,n_points,n_steps)

                # color_fg,sigma_fg=self.decoder(p_fg,r_fg)
                # color_fg=color_fg.reshape(batch_size,n_points,n_steps,-1)
                # sigma_fg=sigma_fg.reshape(batch_size,n_points,n_steps)

                # results2= model.run_cuda1(rays_o=rays_o_box, rays_d=rays_d_box,max_steps=n_steps)
                # results1= model.run_cuda1(rays_o=rays_o, rays_d=rays_d,max_steps=n_steps)

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

                # rgbs is 4 feature
                
                # print(f"rgbs2.shape{rgbs2.shape}")
                # rgbs2.shapetorch.Size([1, 25600, 512, 4])
                
                # pred_rgb = fg_trainer.diffusion.decode_latents(rgbs2).permute(0, 2, 3, 1).contiguous()
                # print(f"pred_rgb.shape{pred_rgb.shape}")
                # rgbs2=pred_rgb

                # print(f"rgbs2_before_decode.shape{rgbs2.shape,rgbs2.device}")
                # results2= model.run_cuda1(rays_o=rays_o_box, rays_d=rays_d_box,max_steps=n_steps)
                # Define the matrix to decode albedo values
                decoder_matrix = torch.tensor([
                    #   R       G       B
                    [0.298, 0.207, 0.208, -0.184],  # L1
                    [0.187, 0.286, 0.173, -0.271],  # L2
                    [-0.158, 0.189, 0.264, -0.473],  # L3
                ], dtype=torch.float32).to(device)
                # # Transpose the matrix
                decoder_matrix = decoder_matrix.T.contiguous()
                # Multiply albedo with the decoder matrix
                rgbs2 = rgbs2@decoder_matrix
                # rgbs2 = (((rgbs2 + 1) / 2)
                #     .clamp(0, 1)  # change scale from -1..1 to 0..1
                #     .mul(0xFF)  # to 0..255
                #     .byte()).to(device)
                rgbs2 = (((rgbs2 + 1) / 2)
                    .clamp(0, 1).byte()).to(device)
                print(f"rgbs2.shape{rgbs2.shape}")
                print(f"rgbs1.shape{rgbs1.shape}")

                # print(f"rgbs2_after_decode.shape{rgbs2.shape,rgbs2.device}")

                color.append(rgbs1)
                color.append(rgbs2)
                sigma.append(sigmas1)
                sigma.append(sigmas2)


                sigma = F.relu(torch.stack(sigma, dim=0))
                color = torch.stack(color, dim=0)
                # # print(f"sigma_shape{sigma.shape}")
                # print(f"color_shape{color.shape}")
                # import pdb; pdb.set_trace()
                sigma_sum,color_sum=self.composite_function(sigma,color)
                # (Pdb) p sigma_sum.shape
                # torch.Size([1, 25600, 512])
                # torch.Size([1, 25600, 512, 4])

                # get alpha
                weights=self.calc_volume_weights(di,rays_d,sigma_sum)
                # (Pdb) p weights.shape
                # torch.Size([1, 25600, 512]) 

                color_map=torch.sum(weights.unsqueeze(-1)*color_sum,dim=-2)
                # torch.Size([1, 25600, 4])
                # color_map=color_map@decoder_matrix
                # color_map = (((color_map + 1) / 2)
                #     .clamp(0, 1)  # change scale from -1..1 to 0..1
                #     .mul(0xFF)  # to 0..255
                #     .byte()).cpu()

                # image = Image.fromarray(latents_ubyte.numpy())
                # print(f"color_map.shape{color_map.shape}")
                # import pdb; pdb.set_trace()
                # color_map=image
                # print(f"color_map.shape{image.shape}")

                # # import pdb; pdb.set_trace()
                # feature_map=color_map.reshape(-1,res,res,4).permute(0, 3, 1, 2).contiguous()
                # # torch.Size([1, 4, 160, 160])
                # # 升维了
                # color_map=fg_trainer.diffusion.decode_latents(feature_map).permute(0, 2, 3, 1).contiguous()

                # print(f"color.shape{color.shape}")
                # #composite ?ngp composite不行？
                # denom_sigma = torch.sum(sigma, dim=0, keepdim=True)
                # denom_sigma[denom_sigma == 0] = 1e-4
                # w_sigma = sigma / denom_sigma
                # sigma_sum = torch.sum(sigma, dim=0)
                # color_weighted = (color * w_sigma.unsqueeze(-1)).sum(0)
                # # print(f"sigma_bg.shape {sigma_bg.shape}")
                # # print(f"color_bg.shape {color_bg.shape}")
                # # sigma_bg.shape torch.Size([128])
                # # color_bg.shape torch.Size([128, 3])

                # pred_rgb=results1['image'].reshape(-1, res, res, 3)
                # pred_depth = results1['depth'].reshape(-1, res, res)
                # # print(f"pred_rgb.shape{pred_rgb.shape}")
                # pred_depth = pred_depth[0].detach().cpu().numpy()
                # pred_depth = (pred_depth * 255).astype(np.uint8)
                # pred = pred_rgb[0].detach().cpu().numpy()
                # pred = (pred * 255).astype(np.uint8)

                # pred_rgb1=results2['image'].reshape(-1, res, res, 3)
                # pred_depth1 = results2['depth'].reshape(-1, res, res)
                # # print(f"pred_rgb.shape{pred_rgb.shape}")
                # pred_depth1 = pred_depth1[0].detach().cpu().numpy()
                # pred_depth1 = (pred_depth1 * 255).astype(np.uint8)
                # pred1 = pred_rgb1[0].detach().cpu().numpy()
                # pred1 = (pred1 * 255).astype(np.uint8)

                # pred_rgb2=color_map.reshape(-1, res, res, 3)
                pred_rgb2=color_map.reshape(-1, res, res, 3)
                pred2 = pred_rgb2[0].detach().cpu().numpy()
                pred2 = (pred2 * 255).astype(np.uint8)
        
                # all_preds.append(pred)
                # all_preds_depth.append(pred_depth)

                # all_preds1.append(pred1)
                # all_preds_depth1.append(pred_depth1)

                all_preds2.append(pred2)
                # print(f"color_np.shape{color_np.shape}")
                # all_preds_depth2.append(pred_depth2)
                # break

        # all_preds = np.stack(all_preds, axis=0)
        # all_preds_depth = np.stack(all_preds_depth, axis=0)
        # save_path='/data/hychen/ln_data/test'
        # name='scene0'
        # imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
        # imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        # all_preds1 = np.stack(all_preds1, axis=0)
        # all_preds_depth1 = np.stack(all_preds_depth1, axis=0)
        # name1='scene1'
        # imageio.mimwrite(os.path.join(save_path, f'{name1}_rgb.mp4'), all_preds1, fps=25, quality=8, macro_block_size=1)
        # imageio.mimwrite(os.path.join(save_path, f'{name1}_depth.mp4'), all_preds_depth1, fps=25, quality=8, macro_block_size=1)


        all_preds2 = np.stack(all_preds2, axis=0)
        save_path='/data/hychen/ln_data/test'
        name2='MLP2RGB'
        print(f"all_preds2.shape{all_preds2.shape}")
        imageio.mimwrite(os.path.join(save_path, f'{name2}_rgb.mp4'), all_preds2, fps=25, quality=8, macro_block_size=1)
        print("finish_video")
        # #################################################

    
        # sigma_bg = sigma_bg.reshape(batch_size, n_points, n_steps)
        # color_bg = color_bg.reshape(batch_size, n_points, n_steps, -1)

        # # color.append(color_fg)
        # # sigma.append(sigma_fg)
        # color.append(color_bg)
        # sigma.append(sigma_bg)

        # sigma = F.relu(torch.stack(sigma, dim=0))
        # color = torch.stack(color, dim=0)

        # Composite
        # sigma_sum, color_weighted = self.composite_function(sigma, color)

        # # Get Volume Weights
        # sigma_sum=sigma_sum.to(self.device)
        # color_weighted=color_weighted.to(self.device)
        # weights = self.calc_volume_weights(di, ray_vector, sigma_sum)
        # color_map = torch.sum(weights.unsqueeze(-1) * color_weighted, dim=-2)

        # # Reformat output
        # color_map = color_map.permute(0, 2, 1).reshape(
        #     batch_size, -1, res, res)  # B x feat x h x w
        # print(f"1color_map.shape {color_map.shape}")

        # color_map = color_map.permute(0, 1, 3, 2)  # new to flip x/y
        # print(f"2color_map.shape {color_map.shape}")

        # return color_map

