# FreeEdit : Towards a Training Free Approach for 3D Scene Editing


![image](https://github.com/user-attachments/assets/4fbb1afe-d964-4ae1-b15c-bc95d72cf1ad)

<p align="center">

### [Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Madhavaram_Towards_a_Training_Free_Approach_for_3D_Scene_Editing_WACV_2025_paper.pdf) | [Project Page](https://vivekmadhavaram.github.io/FreeEdit_page/) | [Video](https://www.youtube.com/watch?v=hmpQZUqD2tA) | [ArXiv](https://arxiv.org/abs/2412.12766)

#### [Vivek Madhavaram](https://vivekmadhavaram.github.io/vivek_page/)<sup>1</sup> | Shivangana Rawat<sup>2</sup> | [Chaitanya Devaguptapu](https://chaitanya.one/)<sup>2</sup> | [Charu Sharma](https://charusharma.org/)<sup>1</sup> | [Manohar Kaul](https://scholar.google.dk/citations?user=jNroyK4AAAAJ&hl=en)<sup>2</sup>

<sup>1</sup> MLL, IIIT Hyderabad, <sup>2</sup>Fujitsu Research India

</p>

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






