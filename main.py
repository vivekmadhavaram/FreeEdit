import numpy as np
import openai 
import argparse
import re
import torch
import trimesh
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import open3d as o3d
from scipy.stats import mode
import point_cloud_utils as pcu

import os
from insertion import *
from replace import *
from config import *
from delete import *

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--api',required=True)       # openai api key
parser.add_argument('--scene',required=True)     # path to the scene
parser.add_argument('--prompt',required=True)    # edit instruction
parser.add_argument('--primary_object',default=None)    # path to the primary object/replacing object
args = parser.parse_args()
openai.api_key = args.api
scene = args.scene
prompt = args.prompt
obj_path=args.primary_object
response = openai.ChatCompletion.create(model="gpt-4",
                                                messages=[{"role": "system", "content": default_prompt},
                                                {"role": "user", "content": prompt}],
                                                temperature = 0.5,
                                                max_tokens=120)
text = re.findall(r'\{(.*?)\}', response.choices[0].message.content, re.DOTALL)[0].split('\n')

if len(text)==1:
    text = text[0].split(",")
task = [k for k in text if 'task' in k.lower()][0].split(":")[-1]
pobject = [k for k in text if 'primary' in k.lower()][0].split(":")[-1]
gobject = [k for k in text if 'ground' in k.lower()][0].split(":")[-1]

if 'insertion' in task.lower():
    print("Task : ",task,", Primary Object : ",pobject," Grounding Object : ",gobject)
    insert_obj(task,pobject,gobject,scene,def_voxels,obj_path)
    
elif 'replacement' in task.lower():
    print("Task : ",task,", Replacing Object : ",pobject," Object to be replaced : ",gobject)
    replace_obj(gobject,pobject,scene,obj_path)

elif 'deletion' in task.lower():
    print("Task : ",task,", Grounding Object : ",gobject)
    deletion(scene,gobject)

else:
    print(task)

print("\n\n\nEdit operation done\n\n\n")   

