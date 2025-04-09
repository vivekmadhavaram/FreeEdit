import torch
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import numpy as np
# from torchvision.ops import box_convert
# import cv2
import math
# from statistics import mean 
import trimesh
from config import *
from utils import *


# get the list of scaling factors with respect to each dimension
def adjust_scale(g_obj,p_obj,val):
    min_gx,min_gy,min_gz = np.min(g_obj.vertices,axis=0)
    max_gx,max_gy,max_gz = np.max(g_obj.vertices,axis=0)
    min_px,min_py,min_pz = np.min(p_obj.vertices,axis=0)
    max_px,max_py,max_pz = np.max(p_obj.vertices,axis=0)
    scales =[]
    scales.append(((max_gx - min_gx) )/(max_px-min_px))
    scales.append(((max_gy - min_gy) )/(max_py-min_py))
    scale = min(scales)
    
    return [val*scale]*3




def prop_scaling(pl_obj,gr_obj,pipe,g_mesh):
  if scale_b:
    widths = []
    prompt = gr_obj +".  "+pl_obj+".  "
    count = 0 
    while (len(widths) < num_samples and count < max_samples):
      count += 1
      image = pipe(prompt, num_inference_steps=inf_steps).images[0] #generating images containing both objects
      TEXT_PROMPT = gr_obj+", "+pl_obj
      BOX_TRESHOLD = 0.35
      TEXT_TRESHOLD = 0.25
      image = np.array(image)
      img_torch = torch.Tensor(np.transpose(image, (2, 0, 1)))
      model = load_model(CONFIG_PATH, WEIGHTS_PATH)
      boxes, logits, phrases = predict(                              # GroundingDINO to predict dimensions of each object
          model=model, 
          image=img_torch, 
          caption=TEXT_PROMPT, 
          box_threshold=BOX_TRESHOLD, 
          text_threshold=TEXT_TRESHOLD
      )
      if (gr_obj in phrases and pl_obj in phrases):
        box1 = boxes[phrases.index(gr_obj)]
        ow1 = box1[2].item()
        box2 = boxes[phrases.index(pl_obj)]
        ow2 = box2[2].item()
        val1= math.floor(ow2/ow1 *100)/100
        if (val1 <= max_scale):
          widths.append(val1)
    width = def_width
    if(len(widths)>0): 
      width = min(list(filter(lambda x: x != 0 , widths)))
      
    print("Scale calculated ",width)
    p_mesh = trimesh.load("primary_object.ply")
    scale = adjust_scale(g_mesh,p_mesh,width)
    p_mesh.vertices = np.multiply(p_mesh.vertices,scale)
    p_mesh.export("primary_object.ply")
  else:
    print("scale: ",scale_v)
    p_mesh = trimesh.load("primary_object.ply")
    scale = adjust_scale(g_mesh,p_mesh,float(scale_v))
    p_mesh.vertices = np.multiply(p_mesh.vertices,scale)
    p_mesh.export("primary_object.ply")
    
