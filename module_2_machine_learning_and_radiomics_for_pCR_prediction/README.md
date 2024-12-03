## Coupled XGBoost model for pCR prediction
- For pCR prediction, radiomic features were extracted using the Pyradiomics library based on each segmentation. MRI semantic features underwent logistic regression to identify variables significantly correlated with the pCR indicator. Subsequently, all selected variables were trained on a coupled XGBoost model to predict pCR versus non-pCR outcomes.
- 10 types of radiomic features and 1 original feature were selected through logistic regression screening. Selected features included First-order (n=6; Minimum, 10th Percentile, Interquartile Range, Kurtosis, Skewness, Median), GLSZM (n=2; Size-Zone Non-Uniformity Normalized, Small Area Emphasis), and Shape (n=2; Elongation, Maximum 2D Diameter Slice) with mean values derived from original images. This selection process yielded 44 radiomic features per patient, computed as averages across all two-dimensional tumor images for final feature values.
- During the prediction process, an XGBoost model was trained using log loss with a learning rate of 0.01.
![image](https://github.com/user-attachments/assets/357a88d8-e655-493d-97c4-b30069d47343)

