# MCGLPPI
### The demo code of implementation of MCGLPPI

__Basic Environment Configuation (Windows or Linux, the specific installation time depends on your system configuration):__
* Python 3.9.18
* Pytorch 1.12.1
* CUDA tool kit 11.3.1
* Pytorch Geometric (PyG) for PyTorch 1.12.* and CUDA 11.3
* Pytorch-scatter 2.1.0
* Torchdrug 0.2.1

## Quick start ##

__We have provided our pre-processed CG source data pickle file (in downstream_files/PDBBIND), pre-trained CG graph encoder (in pretrained_cgmodels), and corresponding running script demos (based on the PDBbind) for a quick start.__ 

__Please follow the illustration in config/ppi_cg/cg_pdbbind_gearnet_gbt.yaml to set the hyper-parameters for the downstream evaluation configurations (supporting both training-from-scratch or fine-tuning from the pre-trained checkpoint).__

__A running example (including the training and evaluation to create the similar results reported):__

__After the environment configuration, usually several hours are needed to finish running the demo code. The evaluation results could slightly vary according to the actual installed virtual environment and the supporting hardware.__

python cg_steps/cg_downstream_1gpu_10CV_GBT.py -c config/ppi_cg/cg_pdbbind_gearnet_gbt.yaml  

(whether to use the pre-trained CG graph encoder checkpoint can be directly specifiied by 'model_checkpoint' arugment in above yaml file, if not, excuating training-from-scratch)

__The complete MCGLPPI framework, including the CG geometer parameter generation code, CG pre-training scripts, evaluation scripts of other downstream datasets, all mentioned source data files, and other supporting scripts/materials will be released upon acceptance.__



