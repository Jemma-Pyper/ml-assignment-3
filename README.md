## Machine Learning 441 Assignment 3 Planning

### Incremental Class Learning

Incremental Class Learning - address imbalance for multi-class problems.

* Add classes in order from least representative to most representative.
* Search for problems with or without class imbalance (use 2/3 datasets for consistency?).
* If no class imbalance is present, downsample to create class imbalance.

### 1. Baseline Neural Network

Train NN without incremental class learning and with no methods to correct class imbalance.

* Tune hidden layer size to avoid overfitting and underfitting.

### 2. Incremental Class Learning Algorithm

* Start with the 2 least frequent classes.
* Start training with no hidden neurons.
* When performance stagnates, decide if the model is underfitting.

  * If so, add a hidden layer with 1 hidden neuron.
* Add more hidden neurons while underfitting is observed.
* As soon as the model starts to overfit / classification error is satisfactory:

  * Add the next output neuron for the next least representative class label.
* Continue until trained on all class labels and satisfied with performance.

### Comparison

Compare the performance of:

1. Baseline NN
2. Incremental Class Learning NN

Questions:

* Does incremental learning help?
* Explain why or why not?

### Important

**ADD DETAIL throughout the report.**
