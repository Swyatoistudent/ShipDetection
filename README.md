# ShipDetection
I used the U-Net model architecture (link), which was implemented using TensorFlow. Images of size () were provided as input to the model. For training, 2000 examples from the training set were used, containing one or more ships, as well as 20 examples without ships. The validation set was generated from the training data (validation_split=0.2).

File Descriptions:

data_prep.py: Data preprocessing
model.py: Model architecture
training.py: Model training process
test_model.py: A file where you can load weights and test the model on any image or images from the test set.

Model weights link: https://drive.google.com/file/d/1lgWueFzZ3yC62hwM_0E2QHjbxXFIpHiI/view?usp=sharing

In the jupyter notebook, there is a process of dataset analysis, model training, and visualization of metrics during training.

Possible improvements include:

1) Enhanced Data Analysis: Perform a more thorough analysis of the dataset.

2) Class Weight Assignment: Consider assigning class weights to improve the model's training quality.

3) Training on a Larger Dataset: Train the model on a larger dataset.
