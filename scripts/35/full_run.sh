# Training and creating models
python mallet_input.py
bash filter.sh
python feature_vector.py
python svm_run.py
rm -r text.vectors

# Testing
python mallet_input_test.py
bash filter_test.sh
python feature_vector_test.py
python svm_run_test.py
python average_cal.py

#Done - Accuracies printed on stdout
