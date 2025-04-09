import numpy as np
import torch
import trimesh
import point_cloud_utils as pcu
from trimesh import transformations
from config import *
from utils import *
from replace import *


def del_object(old_obj,scene_verts,scene_faces,scene_col,scene):
    masks = get_masks(os.path.join(output_path,scene),scene)
    features = get_features(os.path.join(output_path,scene,scene+"_openmask3d_features.npy"))
    features = normalize_features(features)
    query_embedding = get_query_embedding(old_obj, device, clip_model_type)
    mask_index = get_query_mask(torch.Tensor(features), query_embedding)
    mask = masks[:, mask_index]
    del_coords = scene_verts[mask==1]
    if scene_faces.shape[1]>3:
        scene_faces = triangulate(scene_faces)
    indices = np.where(np.array(mask)==1)[0].tolist()
    del_indices = del_other(del_coords,indices,scene_verts,scene_faces)
    mesh2 = inpainting(del_indices,scene_faces,scene_verts,scene_col)
    old_v = list(set(range(scene_verts.shape[0]))-set(del_indices))
    old_coords = scene_verts[old_v]
    old_colors = scene_col[old_v]
    f_mask = np.all(np.isin(scene_faces,old_v), axis=1)
    new_faces = scene_faces[f_mask]
    
    index_map = {value: index for index, value in enumerate(old_v)}
    new_faces = list(map(lambda row: list(map(lambda x: index_map[x], row)), new_faces))
    mesh = trimesh.Trimesh(old_coords,new_faces,vertex_colors = old_colors)
    result = trimesh.util.concatenate([mesh2,mesh])
    result.export("result.ply")

def deletion(scene,old_obj):
    scene_verts,scene_fac,scene_col = process(scene)
    del_object(old_object,scene_verts,scene_fac,scene_col,scene)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--del_obj',required=True)
    parser.add_argument('--scene',required=True)
    args = parser.parse_args()
    scene = args.scene
    old_object = args.del_obj
    scene_verts,scene_fac,scene_col = process(scene)
    del_object(old_object,scene_verts,scene_fac,scene_col,scene)