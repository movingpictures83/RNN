# RNN
# Language: Python
# Input: TXT
# Output: Prefix
# Tested with: PluMA 1.1, Python 3.6
# Dependencies: Tensorflow 1.14

PluMA plugin to assemble and run a Recurrent Neural Network (Williams et al, 1986).  The plugin follows closely the Tensorflow tutorial at (https://www.tensorflow.org/guide/keras/rnn).

The plugin takes input in the form of a tab-delineated parameter file of keyword-value pairs.  The following keywords are acceptable:

classnames: Possible classification classes for images (TXT)
trainset: Training set image files (CSV)
testset: Test set image files (CSV)
tensor: List of tensors (CSV)
dense: List of dense models (CSV)
optimize: Optimization method
metric: Metric to maximize
epochs: Epochs to run model

The output prefix will be used for two files:

prefix.final.csv: Classifications of all images
prefix.probs.csv: Probabilities of each classification
