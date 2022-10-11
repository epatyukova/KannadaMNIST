# KannadaMNIST

This package contains CNN model for Kannada digits classification https://www.kaggle.com/c/Kannada-MNIST

I use hand-build CNN, SGD optimiser with learning rate scheduling (StepLR) and early stopping. I also use data augmentation (rotation, shift-crop-resize, affine) to improve generalisation.

In order to improve accuracy of prediction pseudo-labeling is implemented as advised in discussion of the competition. Also soft voting of three models trained with different portions of the test set attached to the initial training set is used (slightly improves the score). I also tired to use SWA (https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/) but it did not work. Cross-validation was not used due to the limited availability of computational resources.

Best accuracy on Kaggle which I was able to achieve is 98.280% (public) and 98.2% (private).

Accuracy of the KNN classifier on the same dataset is 91%, and accuracy of the kernelised SVM is 95.5%.

Some parameters for training and evaluation can be specified in config.yaml. 

# Installing dependencies

pip install -r requirements.txt

# Model Training: trains the model from scratch.

python3 train.py

# Model Evaluation: produces submission.csv file using previously trained model.

python3 evaluate.py
