
## KDPRA: Dual-Molecule Knowledge Distillation with Cross-Attention for Protein–RNA Binding Affinity Prediction

<p align="left">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white" /></a>
  <a href="https://pytorch-geometric.readthedocs.io/"><img src="https://img.shields.io/badge/PyTorch%20Geometric-3C78D8?style=flat&logo=python&logoColor=white" /></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white" /></a>
  <a href="https://scipy.org/"><img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white" /></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8-3776AB?style=flat&logo=python&logoColor=white" /></a>
</p>

## 📖 Table of Contents: 

- [Description](#-description)  <!-- 👉 原 #description → 需匹配标题中的 📝 -->
- [System and hardware requirements](#-system-and-hardware-requirements)
- [Software prerequisites](#-software-prerequisites)
- [Datasets](#-datasets)          <!-- 👉 原 #Datasets → GitHub自动转换大写字母为小写 -->
- [Feature](#-feature)            <!-- 👉 原 #Feature → 需添加连字符 -->
- [Environment Setup](#environment-setup)   <!-- 手动锚点方案 -->
- [Trained Models](#-trained-models) <!-- 👉 原 #The-trained-model → 需匹配标题复数形式 -->


## 📝 Description

**Motivation.**  
Quantifying protein–RNA binding affinity is essential for understanding molecular recognition and regulatory mechanisms. However, existing computational methods are often constrained by the scarcity of experimentally measured affinity data and limited cross-modal interaction modeling, which hinders their ability to capture the complex and heterogeneous binding patterns underlying protein–RNA interactions.

**Results.**  
To address these challenges, we propose **KDPRA**, a protein–RNA binding affinity prediction framework that integrates **dual-molecule knowledge distillation** with a **bidirectional cross-attention fusion mechanism**. Modality-specific teacher models are independently trained for proteins and RNAs, guiding the student model to learn informative structural and semantic representations under limited supervision. In parallel, a bidirectional cross-attention module explicitly models fine-grained residue-level interactions between proteins and RNAs, enabling effective cross-modal feature integration. Extensive experiments demonstrate that KDPRA consistently outperforms existing methods across multiple benchmarks. Moreover, case studies show that KDPRA produces biologically interpretable predictions and accurately estimates protein–RNA binding affinities, highlighting its practical utility.


<img src="./Model/model_overview.jpg" alt="Overview" width="800">

## 🖥️ System Requirements

The experiments were conducted on a Linux server with the following hardware and software configuration.

### Hardware
- **CPU**: AMD Ryzen Threadripper PRO 5975WX (32 cores, 64 threads)
- **Architecture**: x86_64 (64-bit)
- **Memory**: 256 GB RAM
- **GPU**: 4 × NVIDIA GeForce RTX 4090
- **NUMA**: Single NUMA node

### Software
- **Operating System**: CentOS Linux 7 (Core)
- **Kernel Version**: Linux 3.10.0-1160.99.1.el7.x86_64
- **CUDA**: NVIDIA CUDA-enabled GPUs (driver compatible with RTX 4090)
- **Python**: Python 3.8 (recommended)

### Notes
- All experiments and evaluations were performed on NVIDIA GPUs with CUDA acceleration enabled.
- The codebase is designed for 64-bit Linux systems and has been tested on CentOS 7.


## 📦 Software prerequisites 
The following is the list of required libraries and programs, as well as the version on which it was tested (in parenthesis).
* [Python](https://www.python.org/) (3.6)
* [ESM-2](https://github.com/facebookresearch/esm) . (esm2_t36_3B_UR50D)
* [BioPython](https://github.com/biopython/biopython) .
* [DSSP](https://github.com/cmbi/dssp) . (2.3.0)
* [DGL](https://www.dgl.ai/). (0.6.0). 
* [CD-HIT](https://github.com/weizhongli/cdhit/releases). (4.8.1) 
* [Pymol](https://pymol.org/2/). This optional plugin allows one to visualize surface files in PyMOL.
* [torch](https://pytorch.org/). (1.9.0) 
* [torch_geometric](https://pytorch.org/). (2.3.1) 
* [torchvision](https://pytorch.org/). (0.10.0+cu111) 
* pandas. (2.0.1) 

## 📊 Datasets

| FILE NAME              | DESCRIPTION                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `data_dict.pkl`        | Training/validation dataset in dictionary format (Protein name, sequence, and binding affinity label). |
| `data_dict_test.pkl`   | Test dataset in dictionary format, prepared in the same format as `data_dict.pkl` but without labels. |
| `protein_dict.pkl`     | Three-dimensional coordinates of protein Cα (CA) atoms for graph construction. |
| `rna_dict.pkl` | Three-dimensional coordinates of RNA atoms used for RNA graph construction. |





## 🧬 Input Features

| FEATURE NAME        | DESCRIPTION                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| DSSP                | Protein secondary-structure and physicochemical features extracted by DSSP, including secondary-structure states, solvent accessibility, backbone geometry, and hydrogen-bond information. |
| ESM-2               | Residue-level protein sequence embeddings generated by the ESM-2 protein language model, capturing evolutionary and contextual information from large-scale pretraining. |
| Distance Matrices   | Pairwise residue distance matrices computed from 3D structures (Cα–Cα for proteins and P–P for RNA), used to construct contact maps and graph edges for structure-aware modeling. |
| Virtual Node        | Hotspot-aware global graph representation derived from O-ring regions, serving as a virtual node to aggregate high-level structural context across the protein graph. |
| RNA-FM              | Residue-level RNA sequence embeddings generated by the RNA-FM foundation model, capturing contextual nucleotide dependencies and implicit secondary-structure information. |
| GAT-based Encoding  | Structure-aware RNA representations learned by Graph Attention Networks (GAT) over RNA residue graphs constructed from spatial proximity information. |

### 🎓 Teacher Models

| MODEL FILE                | MODALITY | PRETRAINING TASK                    | TRAINING DATASET                                                                 | INPUT FEATURES                                   | ARCHITECTURE              | ROLE IN KDPRA |
|---------------------------|----------|-------------------------------------|----------------------------------------------------------------------------------|--------------------------------------------------|---------------------------|---------------|
| `model256_All_sf2_1_2.pth` | RNA      | RNA–small molecule binding affinity | Curated RNA–small molecule affinity dataset derived from **R-SIM**, containing RNA sequences, small-molecule ligands, and experimentally measured binding affinities | RNA-FM embeddings + RNA distance-based residue graph | GAT + CNN                 | Provides sequence- and structure-aware RNA priors to guide RNA representation learning via feature-level knowledge distillation |
| `Full_model_45.pkl`       | Protein  | Protein–protein binding affinity    | Protein–protein binding affinity dataset derived from **PDBbind v2020**, filtered to remove redundancy and covering diverse functional complex categories | ESM-2 residue embeddings + protein contact graph | GAT + Transformer Encoder | Transfers structure-aware protein interaction knowledge to supervise protein representations in the student model |


## 🛠️ Environment Setup
<a id="environment-setup"></a>

1. **Clone the repository**
   ```bash
   git clone https://github.com/Q1DT/KDPRA.git
   cd KDPRA
2. Create the Conda environment from ``./model/environment.yml``
    ```bash
    conda env create -f environment.yml  
3. Activate the environment:
    ```bash
    conda activate KDPRA
## 🎯 Trained Models

The models with trained parameters are put in the directory `` ./Best_model'``

## Usage
### ⚙ Network Architecture
Our model is implemented in ``KDPRA.py``.
You can run ``train.py`` to train the deep model from stratch.


**Model Training**

Run 
```
python train.py
``` 

