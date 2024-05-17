# Convolutional Neural Networks on MNIST Dataset

## Installation

Clone this repository:

```bash
git clone https://github.com/shoryasethia/Tensorflow-CNN-MNIST
```

## Implementation

### [Dataset](https://www.tensorflow.org/datasets/catalog/mnist)

The MNIST Data set is usually used to measure the efficiency of an algorithm in classifying images, so it was chosen to be the data set to be classified.
### Or run the code
```
import tensorlow as tf
from tensorflow.keras import datasets, models, layers

(X_train, y_train), (X_test,y_test) = datasets.mnist.load_data()
```

