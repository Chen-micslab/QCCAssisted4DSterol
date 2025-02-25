# Introduction
 `QCCAssisted4DSterol` is a collection of Python scripts used in “Isomer-Level Unsaturated Sterolomics Empowered by Quantum Chemistry Calculation-Assisted Large-Scale Collision Cross Section Prediction Deciphers Tissue-Specific Sterol Distribution”.
 The project is divided into three main parts
1. MS/MS calculations for N-Me derived unsaturated sterol lipids
2. QCC-Assited dataset CCS prediction workflow of N-Me derived unsaturated sterol lipids.
3. LC-IM-MS/MS based 4D streolomics data processing and identification.

All functions are implemented in jupyter notebook
## MS/MS calculation
The script is written on the basis of RDkit's built-in functions. The script  recognises double bond positions and generates MS/MS based on N-Me fragmentation patterns. Theoretically applicable to all molecules including C=C bond  (only test sterol lipids).
## Training CCS Predicition Model
The CCS prediction process in the paper is implemented using the scikit-learn API. It employs LASSO for feature selection and uses cross-validation to select the best hyperparameters for the SVR model.
## Data processing and sterol identification
Folder structure for each sample. All sample files are available in `./Search/tissue`
![image](https://github.com/user-attachments/assets/16e9d8b2-0f82-4096-a42b-b7f88223e5fc)
## GNN RT prediction
The RT prediction process references the ”Retention Time Prediction through Learning from a Small Training Data Set with a Pretrained Graph Neural Network“.The code can be found at [Github](https://github.com/seokhokang/retention_time_gnn/)



