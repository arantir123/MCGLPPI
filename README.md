# MCGLPPI
### Integration of molecular coarse-grained model into geometric representation learning framework for protein-protein complex property prediction 

__Original bioRxiv version link: https://www.biorxiv.org/content/10.1101/2024.03.14.585015v1.abstract__

__Basic Environment Configuation (Windows or Linux, the specific installation time depends on your system configuration):__
* Python 3.9.18
* Pytorch 1.12.1
* CUDA tool kit 11.3.1
* Pytorch Geometric (PyG) for PyTorch 1.12.* and CUDA 11.3
* Pytorch-scatter 2.1.0
* Torchdrug 0.2.1

## Steps about how to generate required data for model pre-training and model downstream predictions ##

__Step1. Generating the coarse-grained (CG) protein files as the initial input of MCGLPPI.__

1. No matter for the pre-training phase or the downstream prediction phase, the framework requires the CG geometric parameters generated by MARTINI2 (https://pubs.acs.org/doi/10.1021/ct700324x) or MARTINI3 (https://www.nature.com/articles/s41592-021-01098-3) for each sample point as the input.

2. As MCGLPPI focuses on the general protein-protein interaction (PPI) complex overall property prediction tasks, the explicit specification of two interaction parts for each original PDB complex file (as a sample point) is required. This can be easily achieved by specifying the chain id of each atom line to A (i.e., interaction part A) or B (i.e., interaction part B) in the corresponding PDB file based on the prior knowledge.

3. The raw scripts for transforming the full-atom PDB file into the CG geometric parameters can be found in http://www.cgmartini.nl/index.php/tools2/proteins-and-bilayers/204-martinize, version 2.4 (for MARTINI2) and https://arxiv.org/abs/2212.01191 (for MARTINI3). On top of this, we provide corresponding further optimized pipeline scripts to generate the CG geometric parameters exactly in line with the input requirement of MCGLPPI, which will updated very soon.

4. Briefly, after the CG transformation (applicable for both MARTINI2 and MARTINI3), for each sample point, three files will be read by the MCGLPPI CG Protein Class Creation script for further creating the CG protein graph described in the manuscript: 1) PDB name-cg.pdb, containing the CG bead lines equivalent to the full-atom PDB file that provides the particle type and coordinate information. 2) PDB name-cg_A.itp, containing the other geometric parameters which can be used to calculate the CG bond angles and dihedrals in interaction part A, etc. 3) PDB name-cg_B.itp: analogous to PDB name-cg_A.itp. We provide a simple demo in demo_samples/1a22/ as an example. The complete datasets provided below also follow this format (with adding the original full-atom PDB file as an extra reference). 



## Quick start ##

__We have provided:__ 

__(1) our pre-processed CG source data pickle file (in downstream_files/PDBBIND),__

__(2) pre-trained CG graph encoder (in pretrained_cgmodels, below is the checkpoint name),__

cgdiff_seed0_gamma0.2_bs64_epoch50_dim256_length150_radius5_extra_step2_0_ls3did_fepoch200_bbfeatsFalse_miFalse.pth

__(3) corresponding running script demos (based on the PDBbind) for a quick start.__ 

__Please follow the illustration in config/ppi_cg/cg_pdbbind_gearnet_gbt.yaml to set the hyper-parameters for the downstream evaluation configurations (supporting both training-from-scratch or fine-tuning from the pre-trained checkpoint).__

__A running example (including the training and evaluation to create the similar results reported):__

__After the environment configuration, usually several hours are needed to finish running the demo code. The evaluation results might be varying according to the actual installed virtual environment and the supporting hardware.__

python cg_steps/cg_downstream_1gpu_10CV_GBT.py -c config/ppi_cg/cg_pdbbind_gearnet_gbt.yaml  

(whether to use the pre-trained CG graph encoder checkpoint can be directly specified by the 'model_checkpoint' argument in above yaml file, if not, excuating training-from-scratch)

__The complete MCGLPPI framework, including the CG geometer parameter generation code, CG pre-training scripts, evaluation scripts of other downstream datasets, all mentioned source data files, and other supporting scripts/materials will be released upon acceptance.__

# MCGLPPI++
### The demo code of implementation of MCGLPPI++

__Basic Environment Configuation (Windows or Linux, the specific installation time depends on your system configuration):__
* Python 3.9.18
* Pytorch 1.12.1
* CUDA tool kit 11.3.1
* Pytorch Geometric (PyG) for PyTorch 1.12.* and CUDA 11.3
* Pytorch-scatter 2.1.0
* Torchdrug 0.2.1

## Quick start ##

__We have provided:__

__(1) our pre-processed CG source data pickle file (in the following link),__

https://drive.google.com/file/d/1NpgCtAmIcyiUjjbcfq7uSfGymjQMR-IK/view?usp=sharing

__(2) pre-trained CG graph encoder (in pretrained_cgmodels, below is the checkpoint name),__

cgdiff_seed0_gamma0.2_bs64_epoch50_dim256_length150_radius5_extra_step2_0_l3did_fepoch200_bbfeatsFalse_miFalse.pth

__(3) corresponding running script demos for a quick start.__ 

__Please follow the illustration in config/ppi_cg_maml/cg_maml_gearnet_reg.yaml to set the hyper-parameters for the downstream evaluation configurations.__

__A running example:__

__After the environment configuration, usually dozens of minutes are needed to finish running the demo code. The evaluation results might be varying according to the actual installed virtual environment and the supporting hardware.__

python cg_maml_steps/cg_downstream_1gpu_maml_reg.py -c config/ppi_cg_maml/cg_maml_gearnet_reg.yaml  

__The complete MCGLPPI++ framework, including complete original data and implementation scripts, will be released upon acceptance.__



