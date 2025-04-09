import torch
# import pandas as pd
import os
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget,decode_latent_mesh

from grounding import isolate_object
from scaling import prop_scaling
import trimesh
import point_cloud_utils as pcu
from placing import *
from config import *
from utils import rotate,dist_metric
# import time
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionXLPipeline

def insert_obj(task,primary_object,grounding_object,scene,voxels,obj_path):
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    repo_id = "stabilityai/stable-diffusion-2-base"
    pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    try:
        if not obj_path:
            latents = sample_latents(
                batch_size=1,
                model=model,
                diffusion=diffusion,
                guidance_scale=15,
                model_kwargs=dict(texts=[primary_object]),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=64,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )

            t = decode_latent_mesh(xm, latents).tri_mesh()
            with open(f'primary_object.ply', 'wb') as f:
                t.write_ply(f)                                              # saving primary obejct mesh as ply file
        else:
            trimesh.load_mesh(obj_path).export("primary_object.ply")
                                                           
        print("Generated primary object")
        g_mesh = isolate_object(grounding_object,scene)                     # object grounding from input scene
        print("Grounded object from the scene")
        prop_scaling(primary_object,grounding_object,pipe,g_mesh) # scaling of primary object
        print("done scaling")
        p_mesh = place_obj(voxels,g_mesh)                                          # location finder algorithm
        g_mesh = trimesh.load_mesh(os.path.join(resource_path,scene,scene+".ply"))
        p_mesh,_,sdf1 = rotate(p_mesh,os.path.join(resource_path,scene,scene+".ply"))
        result = trimesh.util.concatenate([g_mesh,p_mesh])
        result.export("result.ply")
        if metrics:
            # Metric to find the distance between primary object and grounding objects suface. Positive value tells the primary object intersects with scene mesh and vice versa. 
            dist_metric(p_mesh,os.path.join(resource_path,scene,scene+".ply"))   
            print("Penetration percent: ",sdf1)
    

    except Exception as e: print(e)