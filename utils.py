import numpy as np
import point_cloud_utils as pcu
import trimesh
from trimesh import transformations
import os
import torch
import clip
from config import *


def triangulate(faces):
    m = faces.shape[0] * 2 
    triangular_faces = np.zeros((m, 3), dtype=faces.dtype)
    for i in range(faces.shape[0]):
        triangular_faces[2*i] = faces[i, [0, 1, 2]]
        triangular_faces[2*i + 1] = faces[i, [0, 2, 3]]
    return triangular_faces

def sdf_metric(mesh,surface):
    query_pts = mesh.vertices
    v, f = pcu.load_mesh_vf(surface)
    if f.shape[1] >3 :
        f= triangulate(f)
    sdf,_,_ = pcu.signed_distance_to_mesh(query_pts, v.astype('float64'), f)
    no_of_negatives =  len(list(filter(lambda x: x > 0, sdf.tolist())))/len(sdf.tolist())
    if ('scene' in surface and no_of_negatives>0.6):
        return 1-no_of_negatives
    return no_of_negatives

def dist_metric(mesh,surface):
    verts = mesh.vertices
    min_z = np.min(verts[:, 2])
    v, f = pcu.load_mesh_vf(surface)
    if f.shape[1] >3 :
        f= triangulate(f)
    filtered_vertices = verts[verts[:, 2] == min_z]
    sdf,_,_ = pcu.signed_distance_to_mesh(filtered_vertices, v.astype('float64'), f)
    print("Distance value: ",sdf)

def rotate(p_mesh,surface):
    sdf_values =[]
    min_x,min_y,min_z = np.min(p_mesh.vertices,axis=0)
    max_x,max_y,_ = np.max(p_mesh.vertices,axis=0)
    center = [(min_x+max_x)/2,(min_y+max_y)/2,min_z]
    for i in range(4):
        mesh = p_mesh
        angle = (i*90)/57.295
        R = transformations.rotation_matrix(angle, [0, 0, 1], center)
        mesh.apply_transform(R)
        sdf_values.append(sdf_metric(mesh,surface))
    sdf = min(sdf_values)
    ind = sdf_values.index(sdf)
    angle = (ind*90)/57.295
    R = transformations.rotation_matrix(angle, [0, 0, 1], center)
    p_mesh.apply_transform(R)
    return p_mesh,sdf_values[0],sdf

def angle(vector1, vector2 ):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle_in_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angle_in_radians

def process(name):
    verts,faces,colors = pcu.load_mesh_vfc(os.path.join(resource_path,name,name+".ply"))
    if faces.shape[1]>3:
        faces = triangulate(faces)
    x_min,y_min,z_min = np.min(verts,axis=0)
    x_max,y_max,z_max = np.max(verts,axis=0)
    verts1 = verts[verts[:,1]==y_min][0]
    verts2 = verts[verts[:,0]==x_max][0]
    vector1 = [verts2[0]-verts1[0],verts2[1]-verts1[1],verts2[2]-verts1[2]]
    ang = angle(vector1,[1,0,0]) 
    R = transformations.rotation_matrix( 3.14 - ang, [0, 0, 1], [0,0,0])
    V_homogeneous = np.hstack((verts, np.ones((verts.shape[0], 1))))
    V_rotated_homogeneous = np.dot(R, V_homogeneous.T).T
    verts = V_rotated_homogeneous[:, :3]
    return verts,faces,colors
    


def get_masks(outdir,filename):
    maskpath = os.path.join(outdir,filename+"_masks.pt")
    masks = torch.load(maskpath)
    return masks


def get_features(outdir):
    features = np.load(outdir)
    return features


def get_query_embedding(query, device, clip_model_type):
    clip_model, _ = clip.load(clip_model_type, device)
    text_input_processed = clip.tokenize(query).to(device)
    with torch.no_grad():
        query_embedding = clip_model.encode_text(text_input_processed)
    query_embedding_normalized =  (query_embedding/query_embedding.norm(dim=-1, keepdim=True)).float().cpu()
    return query_embedding_normalized


def normalize_features(mask_features):
    mask_features_normalized = mask_features/np.linalg.norm(mask_features, axis=1)[..., None]
    mask_features_normalized[np.isnan(mask_features_normalized) | np.isinf(mask_features_normalized)] = 0.0
    mask_features_normalized = torch.Tensor(mask_features_normalized).to(device)
    return mask_features_normalized


def get_query_mask(mask_features_normalized, sentence_embedding_normalized):
    per_class_similarity_scores = torch.matmul(mask_features_normalized, sentence_embedding_normalized.T) #(177, 20)
    max_ind = np.argmax(per_class_similarity_scores)
    max_ind = max_ind.item()
    return max_ind



def rotation(mesh,deg,point):
    R = transformations.rotation_matrix(deg, [0, 0, 1], point)
    return mesh.apply_transform(R)


def get_isolated_obj(query,outdir,filename):
    masks = get_masks(outdir,filename)                                 
    feature_path = os.path.join(outdir,filename+"_openmask3d_features.npy")
    features = get_features(feature_path)
    features = normalize_features(features)
    query_embedding = get_query_embedding(query, device, clip_model_type)
    mask_index = get_query_mask(torch.Tensor(features), query_embedding)
    return masks,mask_index