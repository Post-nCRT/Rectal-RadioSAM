import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define columns to extract
columns_to_extract = {
    'original_firstorder_Minimum', 'original_glszm_SizeZoneNonUniformityNormalized',
    'original_firstorder_10Percentile', 'original_firstorder_InterquartileRange',
    'original_glszm_SmallAreaEmphasis', 'diagnostics_Image-original_Mean',
    'original_shape_Elongation', 'original_firstorder_Median',
    'original_firstorder_Kurtosis', 'original_firstorder_Skewness',
    'original_shape_Maximum2DDiameterSlice'
}

# File paths for the datasets
file_paths = [
    'D:\\ASTRO\\output_features3.csv', 'D:\\ASTRO\\output_features4.csv',
    'D:\\ASTRO\\output_features.csv', 'D:\\ASTRO\\output_features2.csv'
]

# Read the data from the CSV files
data_frames = [pd.read_csv(file_path) for file_path in file_paths]

# Process the data frames by extracting required columns and adding prefixes
for i, df in enumerate(data_frames):
    prefix = f'file{i+1}_'
    df = df[list(columns_to_extract) + ['label']]
    df = df.add_prefix(prefix)
    data_frames[i] = df

# Combine features and labels
features = pd.concat(
    [df.drop(columns=[f'file{i+1}_label']) for i, df in enumerate(data_frames)], axis=1
)
labels = data_frames[1][f'file2_label']

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Process the second set of CSV files (5-8)
file_paths_2 = [
    'D:\\ASTRO\\output_features5.csv', 'D:\\ASTRO\\output_features6.csv',
    'D:\\ASTRO\\output_features7.csv', 'D:\\ASTRO\\output_features8.csv'
]

data_frames_2 = [pd.read_csv(file_path) for file_path in file_paths_2]

# Extract the required columns and add prefixes
for i, df in enumerate(data_frames_2, start=5):
    prefix = f'file{i}_'
    df = df[list(columns_to_extract) + ['label']]
    df = df.add_prefix(prefix)
    data_frames_2[i-5] = df

# Combine features and labels for the second set of files
features2 = pd.concat(
    [df.drop(columns=[f'file{i}_label']) for i, df in enumerate(data_frames_2, start=5)], axis=1
)
labels2 = data_frames_2[0][f'file5_label']
features2 = scaler.fit_transform(features2)

# Modify some labels as required
labels2[-22:] = [1] * 22

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=100)
positive_count = sum(y_train)
negative_count = len(y_train) - positive_count
scale_pos_weight = negative_count / positive_count

# XGBoost model parameters
params = {
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'n_estimators': 500,  # Number of base learners
    'max_depth': 200,
    'learning_rate': 0.01,
}

# Initialize and train the XGBoost model
model = xgb.XGBClassifier(**params)
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric="logloss", eval_set=eval_set, verbose=False)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get the predicted probabilities

# Apply a custom threshold
# threshold =   need set by experience
y_pred = [1 if prob >= threshold else 0 for prob in y_pred_proba]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

# Calculate sensitivity and specificity
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# Calculate ROC curve and AUC
fpr1, tpr1, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr1, tpr1)

# Print the results
print(f"Accuracy: {accuracy:.3f}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"AUC: {roc_auc:.3f}")

# Predict on the second feature set
y_pred = model.predict(features2)
y_pred_proba = model.predict_proba(features2)[:, 1]

# Apply the same custom threshold
y_pred = [1 if prob >= threshold else 0 for prob in y_pred_proba]

# Calculate confusion matrix for the second feature set
cm = confusion_matrix(labels2, y_pred)
TN, FP, FN, TP = cm.ravel()

accuracy = (TN + TP) / (TN + FP + FN + TP)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# Calculate ROC curve and AUC for the second feature set
fpr2, tpr2, thresholds2 = roc_curve(labels2, y_pred_proba)
roc_auc2 = auc(fpr2, tpr2)

# Print the results for the second feature set
print(f"Accuracy: {accuracy:.3f}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"AUC: {roc_auc2:.3f}")

file_paths = [
    'D:\\ASTRO\\output_features3.csv',
    'D:\\ASTRO\\output_features4.csv',
    'D:\\ASTRO\\output_features.csv',
    'D:\\ASTRO\\output_features2.csv',
]


data_frames = [pd.read_csv(file_path) for file_path in file_paths[1::2]]

for i, df in enumerate(data_frames):
    prefix = f'file{i+1}_'
    df = df[list(columns_to_extract) + ['label']]
    df = df.add_prefix(prefix)
    data_frames[i] = df

# 合并特征和标签
features = pd.concat(
    [df.drop(columns=[f'file{i+1}_label']) for i, df in enumerate(data_frames)], axis=1
)

labels = data_frames[1][f'file2_label']



scaler = StandardScaler()
features = scaler.fit_transform(features)

file_paths_2 = [
    'D:\\ASTRO\\output_features5.csv',
    'D:\\ASTRO\\output_features6.csv',
    'D:\\ASTRO\\output_features7.csv',
    'D:\\ASTRO\\output_features8.csv'
]


data_frames_2 = [pd.read_csv(file_path) for file_path in file_paths_2[1::2]]

for i, df in enumerate(data_frames_2, start=5):
    prefix = f'file{i}_'
    df = df[list(columns_to_extract) + ['label']]

    df = df.add_prefix(prefix)
    data_frames_2[i-5] = df


features2 = pd.concat(
    [df.drop(columns=[f'file{i}_label']) for i, df in enumerate(data_frames_2, start=5)], axis=1
)
labels2 = data_frames_2[0][f'file5_label']
features2 = scaler.fit_transform(features2)


labels2[-22:] = [1] * 22


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=100)
positive_count = sum(y_train)
negative_count = len(y_train) - positive_count
scale_pos_weight = negative_count / positive_count


model = xgb.XGBClassifier(**params)
#
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric="logloss", eval_set=eval_set, verbose=False)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取预测的概率

# threshold =  need set by experience
y_pred = [1 if prob >= threshold else 0 for prob in y_pred_proba]


accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

y_pred = model.predict(features2)

y_pred_proba = model.predict_proba(features2)[:, 1]  # 获取预测的概率


# threshold =  need set by experience
y_pred = [1 if prob >= threshold else 0 for prob in y_pred_proba]

cm = confusion_matrix(labels2, y_pred)
TN, FP, FN, TP = cm.ravel()

accuracy = (TN + TP)/ (TN + FP + FN + TP)
print(TN, FP, FN, TP)
print(f"Accuracy: {accuracy:.3f}")

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")


fpr3, tpr3, thresholds3 = roc_curve(labels2, y_pred_proba)
roc_auc3 = auc(fpr3, tpr3)
print(f"AUC: {roc_auc3:.3f}")


plt.figure()
plt.plot(fpr1, tpr1, color='red', lw=2, label=f'Pre-nCRT T2WI & DWI (log[S(1000)]) and post-nCRT \nT2WI & DWI (log[S(1000)]) (AUC = {round(roc_auc1, 2):.2f})')
plt.plot(fpr2, tpr2, color='blue', lw=2, label=f'Pre-nCRT T2WI & DWI (log[S(1000)]) (AUC = {round(roc_auc2, 2):.2f})')
plt.plot(fpr3, tpr3, color='orange', lw=2, label=f'Post-nCRT T2WI & DWI (log[S(1000)]) (AUC = {round(roc_auc3, 2):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.legend(loc="lower right")
plt.grid()
plt.show()
