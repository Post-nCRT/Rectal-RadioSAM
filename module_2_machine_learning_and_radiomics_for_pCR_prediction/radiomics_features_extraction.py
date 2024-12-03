import os
import numpy as np
import SimpleITK as sitk
import csv
from radiomics import featureextractor

# Set paths for the original image folder and segmentation image folder
# original_image_folder = "./data/train/npy1/MRI_Abd/imgs"
# segmentation_folder = "./data/train/npy1/MRI_Abd/segs"
original_image_folder = "./data/test/npy1/MRI_Abd/imgs"
segmentation_folder = "./data/test/npy1/MRI_Abd/segs"

# Create a feature extractor
params = {
    'binWidth': 25,  # Interval for gray levels
    'resampledPixelSpacing': None,
    'interpolator': sitk.sitkBSpline
}
extractor = featureextractor.RadiomicsFeatureExtractor(**params)

# Save features to a CSV file
output_csv_path = "./output_features1.csv"

# Open the CSV file in append mode
with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
    print(sorted(os.listdir(original_image_folder)))
    # Iterate over the files in the original image folder
    for filename in sorted(os.listdir(original_image_folder)):
        # Load the original image
        original_image_path = os.path.join(original_image_folder, filename)
        original_image = np.load(original_image_path)
        original_image = original_image[:, :, 2]
        original_image = np.expand_dims(original_image, axis=-1)
        original_image_sitk = sitk.GetImageFromArray(original_image)
        
        # Load the segmentation image
        segmentation_path = os.path.join(segmentation_folder, 'seg_' + filename)
        segmentation = np.load(segmentation_path)
        segmentation = np.expand_dims(segmentation, axis=-1)
        segmentation_sitk = sitk.GetImageFromArray(segmentation)
        
        # Extract features
        feature_vector = extractor.execute(original_image_sitk, segmentation_sitk)
        
        # Remove entries where values are tuples or dictionaries
        feature_vector = {key: value for key, value in feature_vector.items() if not isinstance(value, (tuple, dict))}
        
        # List of feature names to remove
        features_to_remove = ['diagnostics_Versions_PyRadiomics',
                              'diagnostics_Versions_Numpy',
                              'diagnostics_Versions_SimpleITK',
                              'diagnostics_Versions_PyWavelet',
                              'diagnostics_Versions_Python',
                              'diagnostics_Image-original_Hash',
                              'diagnostics_Mask-original_Hash',
                              'diagnostics_Image-original_Dimensionality']
        
        # Remove specific keys
        feature_vector = {key: value for key, value in feature_vector.items() if key not in features_to_remove}
        header = list(feature_vector.keys())
        feature_vector = [feature_vector]
        
        # Write feature values row by row
        dictWriter = csv.DictWriter(f, header, lineterminator='\n')
        if filename.endswith('MRI_Abd_rectal_0001-000.npy'):
            dictWriter.writeheader()
        dictWriter.writerows(feature_vector)
        print('save')

print("Features have been saved to the CSV file:", output_csv_path)
