# Image_Classifier_PYTORCH

# Project Overview

Here the task was to build an image classifier for flowers. The flowers dataset on which the training is done can be found
here: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html . The image classifier should provide some options for
example if the training is done on CPU or GPU, the learning rate, the transfer learning network etc. 

# Packages

For this jupyter notebook to run you need these common packages: Pyhton 3.6, jupyter notebook, numpy, matplotlib, torch,
PIL.

# Files

assets directory: pictures for the jupyternotebook.

Image Classifier Project.ipynb: jupyter notebook which explaines all the steps of the classification process.

model.py: neural network architecture

predict_function.py: function to make prediction

train_function.py: function to train the model

train.py and predict.py : function to use with keyboard input to classify a flower.

cat_to_name.json : a file to map the categories of the flowers to their names

# Conclusion, Findings

The image classifier reached an accuracy of 0.745 of the test set. This was done by using the pretrained densenet121.

# Licensing, Authors, Acknowledgements

This repository is licensed under the MIT-License. Big thanks to udacity for providing the data and the code to start.
