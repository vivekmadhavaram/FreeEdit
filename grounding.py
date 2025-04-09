import numpy as np
import torch
import os
import trimesh
from config import *
import point_cloud_utils as pcu
from utils import *


def isolate_object(query,filename):
    masks, mask_index = get_isolated_obj(query,os.path.join(output_path,filename),filename) 
    mask = masks[:, mask_index]
    verts,faces,colors = pcu.load_mesh_vfc(os.path.join(resource_path,filename,filename+".ply"))
    if faces.shape[1]>3:                 # if the faces are not triangle, this function triangulates them.
        faces = triangulate(faces)
    coords1 = verts[mask==1]
    colors1 = colors[mask==1]
    indices = np.where(np.array(mask)==1)[0].tolist()      # vertex indices of grounded object
    f_mask = np.all(np.isin(faces,indices), axis=1)        # faces conatining the grounded object vertices from scene
    new_faces = faces[f_mask]                              # filtered faces
    index_map = {value: index for index, value in enumerate(indices)}
    new_faces = list(map(lambda row: list(map(lambda x: index_map[x], row)), new_faces)) # updating old vertex ids according new ordering 
    mesh = trimesh.Trimesh(coords1,new_faces,vertex_colors=colors1)
    return mesh

if __name__=='__main__':
    obj = input("Enter grounding object name")
    scene = input("Provide scene name")
    mesh = isolate_object(obj,scene)
    mesh.export("grounded_obj.ply")




