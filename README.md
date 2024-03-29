# Latent-NeRF in Scene Editing
<!-- <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a> -->

## Description :scroll:	

Official Implementation for "Latent-NeRF in Scene Editing"

> We combine the Latent-NeRF and Dreamfusion with NeRF 3D reconstruction. And it can achieve the text guide scene editing with Diffusion model.

## Recent Updates :newspaper:

* `4.5.2023` - Code release

## Getting Started

### Installation :floppy_disk:	

Our environment is strictly following [Latent-NeRF](https://github.com/eladrich/latent-nerf), [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion),  [torch-ngp](https://github.com/ashawkey/torch-ngp) environment

For `Latent-NeRF` with shape-guidance, additionally install `igl`

```bash
conda install -c conda-forge igl
```

Note that you also need a :hugs: token for StableDiffusion. First accept conditions for the model you want to use, default one is [`CompVis/stable-diffusion-v1-4`]( https://huggingface.co/CompVis/stable-diffusion-v1-4). Then, add a TOKEN file [access token](https://huggingface.co/settings/tokens) to the root folder of this project, or use the `huggingface-cli login` command

### Config:

Before training, please carefully check the `config.py` which contains the **Dreamfusion** (object generation) config file and **Instant NGP** (scene reconstruction) config file, and **Latent-NeRF** config file lies in `/latent_nerf/configs`

### Training :	

Scripts for training are available in the `train.py` file.

Meshes for shape-guidance are available under `shapes/`

If you want to edit objects on your own scene, please use scripts in [Instant NGP](https://github.com/NVlabs/instant-ngp) `scripts/colmap2nerf.py`to convert your Videos or Images into **llff Dataset**, then use scripts in [torch-ngp](https://github.com/ashawkey/torch-ngp) `scripts/llff2nerf.py` to convert scene into the type that torch-ngp can use. And you will also need to finetune the size of image and remember to check the 'jpg' or 'png' of your picture` `.

## Repository structure

```shell
├── LICENSE
├── README.md
├── activation.py
├── bg_generator.py
├── **bg_nerf**	# train for background(scene 3D reconstrcution)
│  ├── network.py
│  ├── provider.py
│  ├── renderer.py
│  └── utils.py
├── config.py # config for dreamfusion and instant-ngp
├── encoding.py
├── fg_generator.py
├── **fg_nerf** # train for foreground through dreamfusion(object 3D generation)
│  ├── network.py
│  ├── network_grid.py
│  ├── provider.py
│  ├── renderer.py
│  └── utils.py
├── global.py
├── meshutils.py
├── profile_data
├── **raymarching** # The CUDA ray marching module
├── requirements.txt
├── sd.py # SDS implementation
├── **shapes** # Various shapes to use for shape-guidance
├── **src** # train for foreground through Latent-NeRF(object 3D generation)	
│  ├── __init__.py
│  ├── **latent_nerf**
│  │  ├── __init__.py
│  │  ├── **configs** # config for Latent-NeRF
│  │  │  ├── __init__.py
│  │  │  ├── render_config.py
│  │  │  └── train_config.py
│  │  ├── **models**
│  │  │  ├── __init__.py
│  │  │  ├── **encoders**
│  │  │  │  ├── __init__.py
│  │  │  │  ├── **freqencoder**
│  │  │  │  ├── **gridencoder**
│  │  │  │  └── **shencoder**
│  │  │  │    └── **src**
│  │  │  │      ├── bindings.cpp
│  │  │  │      ├── shencoder.cu
│  │  │  │      └── shencoder.h
│  │  │  ├── encoding.py
│  │  │  ├── mesh_utils.py
│  │  │  ├── nerf_utils.py
│  │  │  ├── network_grid.py
│  │  │  ├── render_utils.py
│  │  │  └── renderer.py
│  │  ├── **raymarching**
│  │  │  ├── **raymarchinglatent**
│  │  │  └── **raymarchingrgb**
│  │  └── **training**
│  │    ├── __init__.py
│  │    ├── **losses**
│  │    │  ├── shape_loss.py
│  │    │  └── sparsity_loss.py
│  │    ├── nerf_dataset.py
│  │    └── trainer.py
│  ├── optimizer.py
│  ├── stable_diffusion.py
│  └── utils.py
├── train.py # to combine foreground and background or Global SDS the whole scene using prompt
└── **util** # The volume combination module
  ├── bounding_box_generator.py
  ├── camera.py
  ├── common.py
  ├── decoder.py
  ├── decoder_ref.py
  ├── generator.py
  └── generator_ref.py 
```

## Acknowledgments

The `Latent-NeRF in Scene Editing` code is heavily based on the [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion), [Latent-NeRF](https://github.com/eladrich/latent-nerf) and  [torch-ngp](https://github.com/ashawkey/torch-ngp) project.

