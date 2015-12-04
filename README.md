# SMAI_Project
Automatic Tag Prediction for Stack Over Flow Questions

## Run Instructions

- bash scripts/full_run.sh

## Detailed explanation

- We have a two folders positive dataset and negative dataset in the respective folders. 
- We are know making a file for each tag using the tagid in the code mallet_input.py, inside the full_phase_1 folder.
- Next, we have to run mallet on it. So, we run feature_vector.py on it and get the format required to run SVM on it and we are string the data in full_phase_2.
- Now, we run svm using svm_run.py and store the models for each tag in its file ( represented by its ID [name of the file] ) in full_phase_3 folder.
