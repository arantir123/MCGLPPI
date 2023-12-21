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

__We have provided our pre-processed CG source data pickle files (including the pretraining and downstream S4169 datasets, etc.) in downstream_files/PDBBIND for a quick start.__ 

Step1. __Download our pre-processed jsonl files in above link, and put it into the ./data/ folder as the source data file.__

Step2. __Follow the illustration in \_4_run_MpbPPI_ddg_prediction.py, to read the above jsonl file and run&evaluate MpbPPI for downstream ddG predictions in two different data splitting ways (for pre-training, the procedure is similar based on the \_4_run_pretraining_MpbPPI_aadenoising_rgraph.py script).__

__A running example (including the training and evaluation to create the similar results reported in the manuscript):__

__After the environment configuration, usually several hours are needed to finish running the demo code.__

python \_4_run_MpbPPI_ddg_prediction.py --data_split_mode 'CV10_random' (mutation-level tenfold cross-validation)

python \_4_run_MpbPPI_ddg_prediction.py --data_split_mode 'complex' (wide-type PPI complex type-based cross-validation)
