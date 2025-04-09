import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import distance
from trimesh import transformations
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget,decode_latent_mesh
import argparse
from config import *
from utils import *





def replace(query,g_object,obj_path):                          # function to scale primary object according to grounding object
    if not obj_path:
        xm = load_model('transmitter', device='cuda')
        model = load_model('text300M', device='cuda')
        diffusion = diffusion_from_config(load_config('diffusion'))
        latents = sample_latents(
                batch_size=1,
                model=model,
                diffusion=diffusion,
                guidance_scale=15,
                model_kwargs=dict(texts=[query]),
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
            t.write_ply(f)
        p_object = trimesh.load_mesh('primary_object.ply')
    else:
        p_object = trimesh.load_mesh(obj_path)
    x1_min,y1_min,z1_min = np.min(p_object.vertices,axis=0)
    x1_max,y1_max,z1_max = np.max(p_object.vertices,axis=0)
    x2_min,y2_min,z2_min = np.min(g_object.vertices,axis=0)
    x2_max,y2_max,z2_max = np.max(g_object.vertices,axis=0)
    if(z2_min <0):
        points = g_object.vertices[g_object.vertices[:,2] < (z2_min*0.9)]
    else:
        points = g_object.vertices[g_object.vertices[:,2] < (z2_min*1.1)]
    if(z1_min <0):
        points1 = p_object.vertices[p_object.vertices[:,2] < (z1_min*0.9)]
    else:
        points1 = p_object.vertices[p_object.vertices[:,2] < (z1_min*1.1)]
    x11_min,y11_min = np.min(points1,axis=0)[:2]
    x11_max,y11_max= np.max(points1,axis=0)[:2]
    x21_min,y21_min = np.min(points,axis=0)[:2]
    x21_max,y21_max= np.max(points,axis=0)[:2]
    scale = np.array([(x21_max-x21_min)/(x11_max - x11_min),(y21_max-y21_min)/(y11_max - y11_min),(z2_max-z2_min)/(z1_max - z1_min)])
    p_object.vertices = p_object.vertices *scale
    g_obj_cent = np.array([(x2_max+x2_min)/2,(y2_max+y2_min)/2,(z2_max+z2_min)/2])
    obj_cen = np.array([(x1_max+x1_min)/2,(y1_max+y1_min)/2,(z1_max+z1_min)/2])
    p_object.vertices = p_object.vertices + ( g_obj_cent-obj_cen)
    return p_object

    




def del_other(del_coords,indices,v,f):                         # delete other objects present on the grounding objects
    min_x,min_y = np.min(del_coords,axis=0)[:2]
    max_x,max_y = np.max(del_coords,axis=0)[:2]
    while True:
        c_indices = indices
        z_max = np.mean(del_coords,axis=0)[2]
        f_mask = np.any(np.isin(f,indices),axis=1)
        new_ind = list(set(f[f_mask].flatten())-set(indices))
        filt_ind = np.where((v[:,2]>=z_max) & (v[:,0]>min_x) &(v[:,0]<max_x)&(v[:,1]>min_y) &(v[:,1]<max_y))[0].tolist()
        new_inds = list(set(filt_ind).intersection(new_ind))
        indices = list(set(indices+new_inds))
        indices.sort()
        if(set(indices) == set(c_indices)):
            return indices


def inpainting(del_indices,f,v,colors):                        # function to create a new surface for inpainting cavities

    mesh_ind = f[np.any(np.isin(f,del_indices), axis=1)]
    mesh_ind = list(set(mesh_ind.flatten().tolist())-set(del_indices))
    mesh1 = trimesh.Trimesh(v[mesh_ind],vertex_colors=colors[mesh_ind])
    min_x,min_y = np.min(mesh1.vertices,axis=0)[:2]
    max_x,max_y = np.max(mesh1.vertices,axis=0)[:2]
    mean = np.mean(distance.cdist(mesh1.vertices,mesh1.vertices))
    dis = np.sqrt((max_x-min_x)**2+(max_y-min_y)**2) * (mean**2)
    mesh1.export("verts.ply")
    pcd = o3d.io.read_point_cloud("verts.ply")
    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = dis 
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius , radius ]))
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                          vertex_normals=np.asarray(mesh.vertex_normals),vertex_colors=np.array(mesh.vertex_colors))
    os.remove("verts.ply")
    return tri_mesh



def repl_object(old_obj,new_obj,scene_verts,scene_faces,scene_col,scene,obj_path):    # function to replace 
    masks, mask_index = get_isolated_obj(old_obj,os.path.join(output_path,scene),scene) 
    mask = masks[:, mask_index]
    mask = masks[:, mask_index]
    del_coords = scene_verts[mask==1]
    rep_mesh = process_obj(del_coords,new_obj,obj_path)
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
    result = trimesh.util.concatenate([rep_mesh,mesh2,mesh])
    result.export("result.ply")
    
    

def process_obj(verts,query,obj_path):    #extracting grounding object actual dimension and scaling replacing object accordingly
    verts_mesh = trimesh.Trimesh(verts)
    x_min,y_min,z_min = np.min(verts,axis=0)
    obj_origin = verts[verts[:,1]==y_min][0]
    if y_min<0:
        points = verts[verts[:,1]<(y_min * 0.8)]
    else:
        points = verts[verts[:,1]<(y_min * 1.2)]
    x_max,y_max,z_max = np.max(points,axis=0)
    point = points[points[:,0]==x_max][0]
    v_1 = [point[0]-obj_origin[0],point[1]-obj_origin[1],point[2]-obj_origin[2]]
    v_2 = [1, 0,0]
    unit_v_1 = v_1 / np.linalg.norm(v_1)
    unit_v_2 = v_2 / np.linalg.norm(v_2)
    dot_product = np.dot(unit_v_1, unit_v_2)
    angle = np.arccos(dot_product)
    verts_mesh = rotation(verts_mesh,- angle,obj_origin)
    rep_mesh = replace(query,verts_mesh,obj_path)
    rep_mesh = rotation(rep_mesh,angle,obj_origin)
    return rep_mesh
    

def replace_obj(old_object,new_object,scene,obj_path):
    scene_verts,scene_fac,scene_col = process(scene)          # preprocessing the scene to align it with axes
    repl_object(old_object,new_object,scene_verts,scene_fac,scene_col,scene,obj_path)
   

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--old_obj',required=True)
    parser.add_argument('--scene',required=True)
    parser.add_argument('--new_obj',required=True)
    parser.add_argument('--primary_object',default=None)
    args = parser.parse_args()
    scene = args.scene
    old_object = args.old_obj
    new_object = args.new_obj
    obj_path = args.primary_object
    scene_verts,scene_fac,scene_col = process(scene)
    repl_object(old_object,new_object,scene_verts,scene_fac,scene_col,scene,obj_path)





