# Convolutional Neural Networks on MNIST Dataset
Every notebook in this repo uses Tensorflow and Keras

## Installation

Clone this repository:

```bash
git clone https://github.com/shoryasethia/Modified-CNN-MNIST
```

### Download Dataset from [here](https://www.tensorflow.org/datasets/catalog/mnist)

### Or run the code
```
import tensorlow as tf
from tensorflow.keras import datasets, models, layers

(X_train, y_train), (X_test,y_test) = datasets.mnist.load_data()
```
Digit Recognition via MNIST Dataset is like "Hello World" of Deep Learning or Classification tasks. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. People beleive that if the architecture works poorly on MNIST it won't work on other dataset at all, but if it works on MNIST doesn't ensure that it would work on other datasets as well. 


> The table below collects the some benchmarks on MNIST. Note that I have tested these results on my system. You are welcome to validate the results using the code provided by me. Test accuracy may differ due to the number of epoch, batch size, etc. To correct/Add this table, please create a new issue.

Architecture | Model | Total Parameters |Notebook | Accuracy | 
|------------|-----|-----|--------|---------------|
| 3 Conv + 3 FC + softmax   | [🔗](https://github.com/shoryasethia/Modified-CNN-MNIST/blob/main/MNIST-numbers-cnn.h5) | 27594 |[.ipynb](https://github.com/shoryasethia/Modified-CNN-MNIST/blob/main/tensorflow-mnist-cnn.ipynb)    | 0.9822000265121460 |
| 2 Conv + 2 FC + softmax | [LeNet-5](https://github.com/shoryasethia/Modified-CNN-MNIST/blob/main/LeNet-DigitRecognition.h5)  | 28844 | [.ipynb](https://github.com/shoryasethia/Modified-CNN-MNIST/blob/main/LeNet-5-digit.ipynb) | 0.9883000254631042 |
| 11 Conv + 4 FC + softmax | [Vgg16]() | 6558346 | [.ipynb](https://github.com/shoryasethia/Modified-CNN-MNIST/blob/main/cnn-mnist.ipynb)| 0.9894999861717224 |


