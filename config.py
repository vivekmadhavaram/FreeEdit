import torch

def_voxels = 4        #filter size
metrics = True        #boolean value to calculate metrics
threshold = 1         #defines the output of convoluation operation 
inf_steps = 25        #parameter for stable diffusion
def_width = 0.12      #default width value if stable diffusion fails to calculate scale (one object can be absent in image)
num_samples = 6       #maximum no of calculated scales
max_samples = 5       #total number of images generated by stable diffusion 
max_scale = 0.22      #max threshold scale value of primary object
scale_v = 0.14        #default scale if stable diffusion is not required
scale_b = True        # Scaling algorithm get executed if value is True, else above scale value is used


resource_path = "/data/vivek/resources/"   #path to Replica/ScanNet input data
output_path = "/data/vivek/output/"        #path to OpenMask3D output
default_prompt = "I will give you a sentence and you need to answer it if needed. Take the resultant sentence and create a short summary of it. Also classify the task in the summary only into either insertion or replacement or deletion. If it is a insertion task, identify primary object and grounding object from the summary. Primary object is the object to be placed and grounding object is the object on which primary object has to be placed. If it is a replacement task, identify object to be replaced and replacing objects from summary. Object to be replaced can be grounding object and replacing object can be primary object. In deletion task, primary object will be None and grounding object will be the object to be deleted. Print only the task category and the two object names in json format like '{task:'',primary object:'',grounding object:''}."
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model_type = 'ViT-L/14@336px'
CONFIG_PATH = "/home/vivek/FreeEdit/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "/home/vivek/FreeEdit/groundingdino_swint_ogc.pth"

