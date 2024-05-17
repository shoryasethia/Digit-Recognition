# Convolutional Neural Networks on MNIST Dataset

## Installation

Clone this repository:

```bash
git clone https://github.com/shoryasethia/Tensorflow-CNN-MNIST
```

## Implementation

### [Dataset](https://www.tensorflow.org/datasets/catalog/mnist)

### Or run the code
```
import tensorlow as tf
from tensorflow.keras import datasets, models, layers

(X_train, y_train), (X_test,y_test) = datasets.mnist.load_data()
```
## Sequential I used is 
```
cnn = models.Sequential([
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.AveragePooling2D((2, 2)), 
    
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.AveragePooling2D((2, 2)),
    
    layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```
## Model Summary
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 64)        640       
                                                                 
 average_pooling2d (Average  (None, 13, 13, 64)        0         
 Pooling2D)                                                      
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 32)        18464     
                                                                 
 average_pooling2d_1 (Avera  (None, 5, 5, 32)          0         
 gePooling2D)                                                    
                                                                 
 conv2d_2 (Conv2D)           (None, 3, 3, 16)          4624      
                                                                 
 max_pooling2d (MaxPooling2  (None, 1, 1, 16)          0         
 D)                                                              
                                                                 
 flatten (Flatten)           (None, 16)                0         
                                                                 
 dense (Dense)               (None, 64)                1088      
                                                                 
 dense_1 (Dense)             (None, 32)                2080      
                                                                 
 dense_2 (Dense)             (None, 16)                528       
                                                                 
 dense_3 (Dense)             (None, 10)                170       
                                                                 
=================================================================
Total params: 27594 (107.79 KB)
Trainable params: 27594 (107.79 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
## Accuracy on Testing Data is
```
0.982200026512146
```
Which is very good for a classification task

## Saved Model can be accessed and used from [here](https://github.com/shoryasethia/Tensorflow-CNN-MNIST/blob/main/MNIST-numbers-cnn.h5)
