# MCGLPPI
### Integration of molecular coarse-grained model into geometric representation learning framework for protein-protein complex property prediction ###

__The original bioRxiv version link: [https://www.biorxiv.org/content/10.1101/2024.03.14.585015v1.abstract]__

__The accepted version link: [https://www.nature.com/articles/s41467-024-53583-w]__

__Basic Environment Configuation (Windows or Linux, the specific installation time depends on your system configuration):__
* Python 3.9.18
* Pytorch 1.12.1
* CUDA tool kit 11.3.1
* Pytorch Geometric (PyG) for PyTorch 1.12.* and CUDA 11.3
* Pytorch-scatter 2.1.0
* Torchdrug 0.2.1

## Steps about how to generate required data for model pre-training and model downstream predictions ##

__Step1. Generating the coarse-grained (CG) protein complex files as the initial input of MCGLPPI.__

1. No matter for the pre-training phase or the downstream prediction phase, the framework requires the CG geometric parameters generated by MARTINI2 (https://pubs.acs.org/doi/10.1021/ct700324x) or MARTINI3 (https://www.nature.com/articles/s41592-021-01098-3) for each sample point as the input.

2. As MCGLPPI focuses on the general protein-protein interaction (PPI) complex overall property prediction tasks, the explicit specification of two interaction parts for each original PDB complex file (as a sample point) is required. This can be easily achieved by specifying the chain id of each atom line to A (i.e., interaction part A) or B (i.e., interaction part B) in the corresponding PDB file based on the prior knowledge.

3. The raw scripts for transforming the full-atom PDB file into the CG geometric parameters can be found in https://cgmartini.nl/docs/downloads/tools/proteins-and-bilayers.html, version 2.4 (for MARTINI2) and https://arxiv.org/abs/2212.01191 (for MARTINI3). On top of this, we provide further optimized pipeline scripts and corresponding descriptions for generating the CG geometric parameters exactly in line with the input requirement of MCGLPPI, which are provided in /create_cg_geo_params/ path.

4. Briefly, after the CG transformation (applicable for both MARTINI2 and MARTINI3), for each sample point, three files will be read by the MCGLPPI CG Protein Class Production script (as the input) for further creating the CG protein graph described in the manuscript: 1) PDB name-cg.pdb, containing the CG bead lines equivalent to the full-atom PDB file that provides the particle type and coordinate information. 2) PDB name-cg_A.itp, containing the other geometric parameters which can be used to calculate the CG bond angles and dihedrals in interaction part A, etc. 3) PDB name-cg_B.itp: analogous to PDB name-cg_A.itp. We provide a simple demo in demo_samples/1a22/ as an example. The complete datasets provided below also follow this format (with adding the original full-atom PDB files with a format of PDB name-aa.pdb as an extra reference).

5. Thanks to the basic code logic of Torchdrug repository (https://torchdrug.ai/), when the first time to run the pre-training or downstream predictions, our scripts (see below) will automatically parse the aforementioned three files for each sample point, and produce and store a pickle (.pkl) file storing all generated CG Protein Classes for current dataset (e.g., cg2_pdbbind_strictdimer.pkl.gz in https://drive.google.com/file/d/14-0QF0b8JeXUU57yMzTdTPzvsNP_ayUv/view?usp=sharing, corresponding to a complete dataset folder including multiple sub-folders, for each representing a sample point) for the rapid repeated use.

__Step2. CG diffusion-based pre-training using domain-domain interaction (DDI) templates based on the parsed pickle file.__

1. Based on our assumption, pre-training on DDI templates, which are critical subsets of PPIs where the interaction typically occurs between domains rather than the entire proteins, could enhance the model's ability on PPI binding affinity related prediction tasks with a relatively smaller sample amount.
   
2. The curated original dataset (including the full-atom PDB file and corresponding three CG geometric files for each sample point) are provided in:
   
   __For full-atom and MARTINI2:__

         https://drive.google.com/file/d/1nYPwkMhpIrTifOLWL-NDpyLOrnuzczjE/view?usp=sharing

   __For MARTINI3__:

         https://drive.google.com/file/d/1S32LteGRSCCVPBM-Ig8MuVKibBMRqHiT/view?usp=sharing 

   Please note that these zipped files contain 50,359 3DID (https://3did.irbbarcelona.org/) sample points covering 15,983 DDI structure templates. However, we remove any DDI templates from the 3DID dataset that are identical to those present in our downstream datasets, and 41,663 sample points are remained for the main experiments in the manuscript. Based on this, we further provide the parsed pickle file for these 41,663 sample points (in which the original sample point names are included) for a quick start:

   __For MARTINI2:__

         https://drive.google.com/file/d/1FACIyhD-Jn1J6MpN7KuiU-uXo_gxgxh7/view?usp=sharing

   __For MARTINI3:__

         https://drive.google.com/file/d/1ptIM69OkHzsP-fJtsUBF1HTBccM3Iv29/view?usp=sharing 

4. After preparing the parsed pickle file, we can run the corresponding scripts for MARTINI2 or MARTINI3 to pre-train the CG GearNet-Edge protein graph encoder (https://github.com/DeepGraphLearning/GearNet) for the downstream use:

   __MARTINI2 (run 1st and then run 2nd):__

         python cg_steps/cg_pretrain.py -c config/ppi_cg/cgdiff_1st.yaml
   
         python cg_steps/cg_pretrain.py -c config/ppi_cg/cgdiff_2nd.yaml

   __MARTINI3 (run 1st and then run 2nd):__

         python cg3_steps/cg3_pretrain.py -c config/ppi_cg3/cg3diff_1st.yaml
   
         python cg3_steps/cg3_pretrain.py -c config/ppi_cg3/cg3diff_2nd.yaml

   __We provide the pre-trained CG graph encoder (based on the aforementioned 41,663 MARTINI2 3DID subset) as an example (in pretrained_cgmodels, below is the checkpoint name):__

         cgdiff_seed0_gamma0.2_bs64_epoch50_dim256_length150_radius5_extra_step2_0_ls3did_fepoch200_bbfeatsFalse_miFalse.pth

   __The below one is the encoder pre-trained on the further 33,144-sample subset described in the original manuscript:__

         cgdiff_seed0_gamma0.2_bs64_epoch50_dim256_length150_radius5_extra_step2_0_lss3did_fepoch200_bbfeatsFalse_miFalse.pth

5. As a reference of the corresponding pre-training scripts in the original scales, we also clone the original ones in this repository (in SiamDiff, original link: https://github.com/DeepGraphLearning/SiamDiff/tree/main).
   
__Step3. Downstream complex overall property predictions based on w/ or w/o pre-trained CG graph encoder.__

__(1) dG predictions based on the PDBbind strict dimer dataset__

1. This dataset only contains the strict dimer complexes curated from the PDBbind v2020 database (http://www.pdbbind.org.cn/download/pdbbind_2020_intro.pdf). We provide relevant files as follows:
   
   1) __Original data for full-atom and MARTINI2:__
      
            https://drive.google.com/file/d/1o8bDAZdQg-sRKdWpEA_5jRv05l0RwyRv/view?usp=sharing

   2) __Original data for MARTINI3:__

            https://drive.google.com/file/d/1pgPsGvvT3zfvaMfmSj5COqtIUMC1m471/view?usp=sharing 

   3) __All dG labels for corresponding complex structures:__
     
            PDBBINDdimer_strict_index.csv in downstream_files/PDBBIND/

1. We also provide the corresponding pickle files for a quick start:

   __MARTINI2 and MARTINI3:__

         https://drive.google.com/file/d/14-0QF0b8JeXUU57yMzTdTPzvsNP_ayUv/view?usp=sharing

2. After the preparation of the source data for MCGLPPI, the .yaml execution scripts can be used to evaluate the model performance based on different data splitting settings. Please follow the illustration in corresponding scripts to set the hyper-parameters for evaluation configurations (the data splitting file ['index_path' argument] and whether to use the pre-trained graph encoder checkpoint ['model_checkpoint' argument] can be specified in these scripts). The running examples including training and evaluation are as follows:

   __Example 1__ (in a standard tenfold cross-validation (CV) setting):

   __MARTINI2:__

         python cg_steps/cg_downstream_1gpu_10CV_GBT.py -c config/ppi_cg/cg_pdbbind_gearnet_gbt_10CV.yaml

   __MARTINI3:__ 

         python cg3_steps/cg3_downstream_1gpu_10CV_GBT.py -c config/ppi_cg3/cg3_pdbbind_gearnet_gbt_10CV.yaml 

   __Example 2__ (in an overall TM-score-based splitting [<0.45: test set, 0.45~0.55: validation set, >0.55: training set]):

   __MARTINI2:__

         python cg_steps/cg_downstream_1gpu_10CV_GBT.py -c config/ppi_cg/cg_pdbbind_gearnet_gbt_TMscore.yaml

    __MARTINI3:__

         python cg3_steps/cg3_downstream_1gpu_10CV_GBT.py -c config/ppi_cg3/cg3_pdbbind_gearnet_gbt_TMscore.yaml

__(2) dG predictions based on the ATLAS dataset__

1. The used dataset is curated from https://onlinelibrary.wiley.com/doi/full/10.1002/prot.25260 and https://github.com/weng-lab/ATLAS/blob/master/README.md.

   The purpose of examining MCGLPPI on this dataset is to check the feasiability of handling more complex geometric binding patterns (beyond the strict dimers) for identifying similar structures generated by the computational simulation.

3. We provide the relevant original data as follows:

   1) __Original data for full-atom and MARTINI2:__
      
            https://drive.google.com/file/d/1SA7fXpbF2r6co7KkPxn0LSeCe_Noau-J/view?usp=sharing 

   2) __Original data for MARTINI3:__

           https://drive.google.com/file/d/1nai954uqdn47ZlOI5l_wfguIGgX3l4IA/view?usp=sharing 
  
   3) __All dG labels for corresponding complex structures:__

           ATLAS.csv in downstream_files/ATLAS/ 

4. We also provide the corresponding pickle files for a quick start:

   __MARTINI2 and MARTINI3:__

         https://drive.google.com/file/d/1dWUoIPK_F4C_hKY5cG5aM1cQJS13azuv/view?usp=sharing 

5. Running examples:

   __Example 1__ (in a standard tenfold cross-validation (CV) setting):

   __MARTINI2:__

         python cg_steps/cg_downstream_1gpu_10CV_GBT.py -c config/ppi_cg/cg_atlas_gearnet_gbt_10CV.yaml

   __MARTINI3:__ 

         python cg3_steps/cg3_downstream_1gpu_10CV_GBT.py -c config/ppi_cg3/cg3_atlas_gearnet_gbt_10CV.yaml 

__(3) ddG predictions based on the AB-bind dataset__

1. We demonstrate the potential of MCGLPPI on the extension into directly predicting ddG with the simple modifications (see manuscript for processing details). The ddG dataset used is a multiple-point mutation dataset AB-bind (https://pubmed.ncbi.nlm.nih.gov/26473627/), which contains 1101 sample points related to the binding affinity change (i.e., ddG) caused by multiple-point amino acid (AA) mutations on the complex formed from antibody or antibody-like binding. We provide relevant files as follows:
   
   1) __Original data for full-atom and MARTINI2:__

            https://drive.google.com/file/d/1vxGXXhtYJw_QmZi9PBjNH8U-AwYsAj5L/view?usp=sharing 

   2) __All ddG labels for corresponding wild-type (WT) - mutant (MT) complex structure pairs:__

            M1101_label.csv in downstream_files/M1101/ 
  
2. We also provide the corresponding pickle file for a quick start:

   __MARTINI2:__

         https://drive.google.com/file/d/1LgR-CD7H4pUTXlWAcZ3kNzv9r05DVJwe/view?usp=sharing 

4. Running example:

   __Example 1__ (in a WT protein-protein complex type-based fivefold CV setting,

   see https://academic.oup.com/bib/article/24/5/bbad310/7256145 for details,

   the splitting file is retrieved from https://github.com/arantir123/MpbPPI):

   __MARTINI2:__

         python cg_steps_energy_injection/cg_downstream_1gpu_10CV_GBT.py -c config/ppi_cg/cg_m1101_gearnet_gbt_WTtype.yaml 
   
__(4) Protein-protein complex interface classifications based on the MANY/DC dataset__ 

1. This dataset is used to examine the model's ability to distinguish/classify the biological interface from crystal artefacts (https://www.nature.com/articles/s41467-021-27396-0). We provide relevant files as follows:
   
   1) __Original data for full-atom and MARTINI2:__
     
            https://drive.google.com/file/d/18oEzeiqKT7tf7f9krFxCQ8QP-o8lLwar/view?usp=sharing 

   2) __Original data for MARTINI3:__
   
            https://drive.google.com/file/d/19R6QQiT2NDC94Pv15pgolm_-cLfUXryy/view?usp=sharing 
  
   3) __All binary classification labels for corresponding complex structures:__

            MANYDC.csv in downstream_files/MANYDC/ 
  
2. We also provide the corresponding pickle files for a quick start:

   __MARTINI2 and MARTINI3:__

         https://drive.google.com/file/d/1DiS9WX8zqKTWolEg4Ebm-fAziMiJvjqR/view?usp=sharing 

4. Running examples:

   __Example 1__ (in a conventional splitting where 80% MANY data points are selected as training set and complete DC data points are test set):

   __MARTINI2:__

         python cg_steps/cg_downstream_1gpu_GBT_stats_cal.py -c config/ppi_cg/cg_manydc_gearnet_DCtest.yaml

   __MARTINI3:__

         python cg3_steps/cg3_downstream_1gpu_GBT_stats_cal.py -c config/ppi_cg3/cg3_manydc_gearnet_DCtest.yaml 
 
## Quick start 【太长不看版】 ##

__We have provided:__ 

__(1) our pre-processed CG downstream source data pickle files for using MCGLPPI quickly (e.g., pickle files for the PDBbind strict dimer dataset in above link),__

__(2) pre-trained CG graph encoder (in pretrained_cgmodels, below is the checkpoint name),__

      cgdiff_seed0_gamma0.2_bs64_epoch50_dim256_length150_radius5_extra_step2_0_ls3did_fepoch200_bbfeatsFalse_miFalse.pth

__(3) corresponding .yaml execution scripts (e.g., ones based on the PDBbind strict dimer dataset) for a quick start.__ 

__Please follow the illustration in corresponding .yaml files (e.g., config/ppi_cg/cg_pdbbind_gearnet_gbt_10CV.yaml) to set the hyper-parameters for the downstream evaluation configurations (supporting both training-from-scratch or fine-tuning from the pre-trained CG graph encoder checkpoint).__

__A running example (including the training and evaluation to create the evaluation results on the test set):__

__After the environment configuration, usually dozens of minutes are needed to finish running the demo code. The evaluation results might be varying according to the actual installed virtual environment and the supporting hardware.__

      python cg_steps/cg_downstream_1gpu_10CV_GBT.py -c config/ppi_cg/cg_pdbbind_gearnet_gbt_10CV.yaml 

(whether to use the pre-trained CG graph encoder checkpoint can be directly specified by the 'model_checkpoint' argument in above .yaml file, if not, excuating training-from-scratch)

__Please note that some of the used data subsets are small due to the limitation of available complex structures and their experimental labels, the evaluation results on these subsets might vary based on different hardwares and environments. Nevertheless, the better trade-off between the computational overhead and chemical-plausible interaction descriptions within complexes can be guaranteed. Please check the original manuscipt link for complete evaluation result reports.__

# MCGLPPI++
### Meta-learning enables complex cluster-specific few-shot binding affinity prediction for protein-protein interactions ###

__The accepted version link: [https://pubs.acs.org/doi/10.1021/acs.jcim.4c01607]__

__Basic Environment Configuation (Windows or Linux, the specific installation time depends on your system configuration):__
* Python 3.9.18
* Pytorch 1.12.1
* CUDA tool kit 11.3.1
* Pytorch Geometric (PyG) for PyTorch 1.12.* and CUDA 11.3
* Pytorch-scatter 2.1.0
* Torchdrug 0.2.1

## Quick start ##

__We have provided (for how to generate the CG protein complex files as the basic model input, please check the MCGLPPI illustration above):__

__(1) our curated more comprehensive PPI dG dataset (including original data for full-atom and MARTINI2),__

This dataset contains the selected PPI structure samples originated from 1) SKEMPI 2.0 (https://life.bsc.es/pid/skempi2/), 2) PDBbind v2020 (http://www.pdbbind.org.cn/), 3) ATLAS (https://pubmed.ncbi.nlm.nih.gov/28160322/), and 4) Panagiotis et al. databases (https://pubs.acs.org/doi/full/10.1021/pr9009854).

      https://drive.google.com/file/d/1j7kj14wLFpACixuqjCIf1ztErK4sULML/view?usp=sharing
   
__(2) our pre-processed CG source data pickle file (based on MARTINI2),__

      https://drive.google.com/file/d/1NpgCtAmIcyiUjjbcfq7uSfGymjQMR-IK/view?usp=sharing

__(3) all dG labels for corresponding complex structures (including the pre-calculated original training cluster splitting features),__

      MAML_complete_index.csv in downstream_files/MAML/ 

__(4) pre-trained CG graph encoder [in pretrained_cgmodels, using 45,714 (out of 50,359) 3DID sample points for pre-training, below is the checkpoint name],__

      cgdiff_seed0_gamma0.2_bs64_epoch50_dim256_length150_radius5_extra_step2_0_l3did_fepoch200_bbfeatsFalse_miFalse.pth

__(5) corresponding running scripts for a quick start.__

__Please follow the illustration in config/ppi_cg_maml/cg_maml_gearnet_reg.yaml to set the hyper-parameters for the downstream evaluation configurations.__

__A running example:__

__After the environment configuration, usually dozens of minutes are needed to finish running the demo code. The evaluation results might be varying according to the actual installed virtual environment and the supporting hardware.__

      python cg_maml_steps/cg_downstream_1gpu_maml_reg.py -c config/ppi_cg_maml/cg_maml_gearnet_reg.yaml  



