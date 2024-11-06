# Rectal-RadioSAM
Rectal-RadioSAM: Large Model-Assisted Multi-Parametric MRI Pipeline for Predicting Response to Neoadjuvant Chemoradiotherapy in Rectal Cancer via No Human Intervention
# Rectal-RadioSAM
## Brief Description
This repository include a program of a multi-parametric MRI-based and large model-assisted automated prediction tool for assessing response to neoadjuvant chemoradiotherapy (nCRT) in rectal cancer.
## How to Execute the Program
To start the program, please follow the instructions step by step.
## User Manual
* Collected MRI images from the four examined modalities underwent comprehensive pre-processing steps. 
* All pre-processed images were uniformly resized to 1024×1024 pixels. 
* Utilize the codebase of a novel two-stage hybrid model that integrates a large segmentation model with the XGBoost algorithm.
* Utilize the codebase of four-channel MedSAM networks to generate corresponding segmentation results.
* Calculate the evaluation metrics and analyse statistical results.
## Architecture of the Model
<img width="534" alt="截屏2024-11-06 19 01 39" src="https://github.com/user-attachments/assets/2103966c-9cbd-41b2-acd5-fbbe88a3789b">

## Samples of Segmentation Results
![image](https://github.com/user-attachments/assets/0111e04b-e0da-4bc6-89be-70820c688821)







