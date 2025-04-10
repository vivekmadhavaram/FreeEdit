# FreeEdit : Towards a Training Free Approach for 3D Scene Editing




<div align="center">

<img src="https://github.com/user-attachments/assets/4fbb1afe-d964-4ae1-b15c-bc95d72cf1ad" alt="Alt text" width="800" height="500"/>

### [Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Madhavaram_Towards_a_Training_Free_Approach_for_3D_Scene_Editing_WACV_2025_paper.pdf) | [Project Page](https://vivekmadhavaram.github.io/FreeEdit_page/) | [Video](https://www.youtube.com/watch?v=hmpQZUqD2tA) | [ArXiv](https://arxiv.org/abs/2412.12766)

#### [Vivek Madhavaram](https://vivekmadhavaram.github.io/vivek_page/)<sup>1</sup> | Shivangana Rawat<sup>2</sup> | [Chaitanya Devaguptapu](https://chaitanya.one/)<sup>2</sup> | [Charu Sharma](https://charusharma.org/)<sup>1</sup> | [Manohar Kaul](https://scholar.google.dk/citations?user=jNroyK4AAAAJ&hl=en)<sup>2</sup>

<sup>1</sup> MLL, IIIT Hyderabad, <sup>2</sup>Fujitsu Research India

</div>

## Setup 
Clone the repository and create a conda environment as below:

```bash
conda create --name FreeEdit python=3.10
conda activate FreeEdit
```
Then install the required packages

```bash
pip install openai==0.28
pip install point-cloud-utils
pip install trimesh
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install clip
pip install pandas
pip install open3d
```

#### Setup [Shap-E](https://github.com/openai/shap-e) for object synthesis
```bash
git clone https://github.com/openai/shap-e
cd shap-e
pip install -e .
cd ..
```
#### Install and run [OpenMask3D](https://github.com/OpenMask3D/openmask3d) 
Follow the procedure mention in the main repo to generate object masks and save them in the output directory. Download the Add the output directory path in the config.py file.

#### Install other packages
```bash
pip install pyyaml
pip install ipywidgets
pip install scikit-learn
pip install diffusers
pip install transformers
pip install numpy==1.24.3
```

#### Setup [GroundingDINO](https://github.com/IDEA-Research/Grounded-Segment-Anything)
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -q -e .
pip install -q roboflow
cd ..
```

### Run FreeEdit
Execute the below command to run the code. Add your OpenAPI key and chnage th prompt accordingly.
```bash
python main.py --api "" --scene office2 --prompt "insert candle on the desk"
```
Replace the prompt for replacement and deletion tasks.
Output will be saved as "Result.ply" which can be loaded into any visualization tool like Blender, Meshlab, etc.





