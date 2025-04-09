import trimesh
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from config import *


def get_vox_grid(cls_verts,voxel_size):                  # function to generate the initial level voxel grid of grounding object surface
    x_min = min(cls_verts[:,0])
    y_min = min(cls_verts[:,1])
    x_diff = int((max(cls_verts[:,0])- x_min)/voxel_size)
    y_diff = int((max(cls_verts[:,1])- y_min)/voxel_size)
    vox_grid = np.zeros((x_diff+1,y_diff+1))
    shift_verts = cls_verts
    shift_verts[:,0] = cls_verts[:,0] - x_min
    shift_verts[:,1] = cls_verts[:,1] - y_min
    shift_verts = (shift_verts/voxel_size).astype('int')
    vox_grid[shift_verts[:,0],shift_verts[:,1]]=1
    return vox_grid


def con_op(vox_grid,voxels):                             # function for computing next level grid
    res = np.zeros((vox_grid.shape[0]-voxels+1,vox_grid.shape[1]-voxels+1))
    for i in range(vox_grid.shape[0] - voxels+1):
        for j in range(vox_grid.shape[1]- voxels+1):
            temp = np.average(vox_grid[i:i+voxels,j:j+voxels])
            if(temp >= threshold):
                res[i,j]=1
    return res


def get_output(vox_grid,voxels):                         # function for applying hierarchical convolution operation on the voxel grid
    cnt = 0
    while 1:
        out_arr = vox_grid
        if(voxels <= vox_grid.shape[0] and voxels <= vox_grid.shape[1]):
            vox_grid = con_op(vox_grid,voxels)
        else:
            return out_arr,cnt
        if 1 not in vox_grid or cnt >10 :
            return out_arr,cnt
        else:
            cnt += 1


def get_scale(verts,voxels):                             # function to get the voxel scale of grounding object surface
    x_diff = (max(verts[:,0])-min(verts[:,0]))/voxels
    y_diff = (max(verts[:,1])-min(verts[:,1]))/voxels
    vox_size = round(max(x_diff,y_diff),5)
    return vox_size


def translate_obj(voxels,labels,g_mesh,p_mesh,filt_verts):
    classes = list(np.unique(labels))
    classes.remove(-1)
    voxel_size = get_scale(p_mesh.vertices,voxels)    
    print("voxel_size ",voxel_size)
    cnts = []
    outputs = []
    pts = []
    cent_verts =[]
    areas = []
    for cls in classes:
        indexes = np.where(labels==cls)
        cls_verts = filt_verts[indexes]
        z_axis = cls_verts[:,2].tolist()
        z = round(mode(z_axis)[0],4)
        pts.append([np.min(cls_verts[:,0]),np.min(cls_verts[:,1]),z])
        vox = get_vox_grid(cls_verts,voxel_size)
        output,cnt = get_output(vox,voxels)
        cnts.append(cnt)
        outputs.append(output)
        cent_verts.append(cls_verts)
        areas.append((np.max(cls_verts[:,0])-np.min(cls_verts[:,0]))*(np.max(cls_verts[:,1])-np.min(cls_verts[:,1])))
    return cnts,outputs,voxel_size,pts,cent_verts,areas

def get_center(parts,labels,g_mesh,p_mesh,filt_verts):   # function to identify optimal location
    cnts,outputs,voxel_size,pts,cl_verts,areas = translate_obj(parts,labels,g_mesh,p_mesh,filt_verts)
    max_cnt = max(cnts)
    ind = [i for i, x in enumerate(cnts) if x == max_cnt]
    if (len(ind ) == 1):
        i = cnts.index(max(cnts))
    else:
        i = max([areas[id] for id in ind])
        i = areas.index(i)
    cl_verts = np.array(cl_verts[i])
    cl_verts[:,0] +=pts[i][0]
    cl_verts[:,1] +=pts[i][1]
    ind = [round(np.mean(np.where(outputs[i] == 1)[0])),round(np.mean(np.where(outputs[i] == 1)[1]))]
    if (ind[0] not in np.where(outputs[i] == 1)[0] or ind[1] not in np.where(outputs[i] == 1)[1]):
        ind = [int(mode(np.where(outputs[i] == 1)[0])[0]),int(mode(np.where(outputs[i] == 1)[1])[0])]
    xm,ym = ind[0]+cnts[i],ind[1]+cnts[i]
    center = [pts[i][0]+(xm+0.5) *voxel_size,pts[i][1]+(ym+0.5) *voxel_size]
    mesh = trimesh.Trimesh(cl_verts)
    
    mul_x1 = 1.2 if center[0] < 0 else 0.8
    mul_x2 = 0.8 if center[0] < 0 else 1.2
    mul_y1 = 1.2 if center[1] < 0 else 0.8
    mul_y2 = 0.8 if center[1] < 0 else 1.2

    new_verts= cl_verts[(cl_verts[:,0] > (mul_x1 *center[0])) & ( cl_verts[:,0] < (mul_x2 *center[0])) & ( cl_verts[:,1] > (mul_y1 *center[1])) & ( cl_verts[:,1] < (mul_y2 *center[1]))]
    if(new_verts.shape[0]>0):
        z = np.mean(new_verts,axis=0)
        center.append(z[2])
    else:
        center.append(pts[i][2])
    return center


def place_obj(voxels,g_mesh):
    p_mesh = trimesh.load_mesh("primary_object.ply")
    te = np.round(np.dot(g_mesh.vertex_normals,np.array([0,0,1])),3)        
    ind = list(np.where(te>0.98)[0])
    filt_verts = g_mesh.vertices[ind]                                       # filtering vertices having normals pointed towards roof
    z_values = np.array(filt_verts)[:, 2].reshape(-1, 1)
    z_values_standardized = StandardScaler().fit_transform(z_values)
    dbscan = DBSCAN(eps = 0.1,min_samples=10)                             
    labels = dbscan.fit_predict(z_values_standardized)                      # clustering surfaces based on height
    center = get_center(voxels,labels,g_mesh,p_mesh,filt_verts)             # identifing optimal x, y coordinates
    min_z = np.min(p_mesh.vertices[:, 2])                                   # identifying base surface of primary object
    filtered_vertices = p_mesh.vertices[p_mesh.vertices[:, 2] == min_z]
    obj_cen = np.mean(filtered_vertices,axis=0)                             # center of primary object base surface
    p_mesh.vertices = p_mesh.vertices + (center - obj_cen)                  # transalting the primary object to optimal location 
    return p_mesh


        
       