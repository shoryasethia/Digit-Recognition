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
Architecture | Model      | notebook | Accuracy | 
|------------|-----|-------------|---------------|
| 3 Conv + 3 FC + softmax   | [🔗](https://github.com/shoryasethia/Modified-CNN-MNIST/blob/main/MNIST-numbers-cnn.h5) | [.ipynb](https://github.com/shoryasethia/Modified-CNN-MNIST/blob/main/tensorflow-mnist-cnn.ipynb)    | 0.9822000265121460 |
| 2 Conv + 2 FC + softmax | [LeNet-5](https://github.com/shoryasethia/Modified-CNN-MNIST/blob/main/LeNet-DigitRecognition.h5)  | [.ipynb](https://github.com/shoryasethia/Modified-CNN-MNIST/blob/main/LeNet-5-digit.ipynb) | 0.9883000254631042 |


