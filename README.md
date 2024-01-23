# Neural-Network-Mnist-Dataset
Repository dedicated to the classification of the minist database using neural networks.

## Installation

Clone this repository:

```bash
git clone https://github.com/shoryasethia/MNIST-Integer-NeuralNetwork
```

## Implementation

### [Dataset](https://www.tensorflow.org/datasets/catalog/mnist)

The MNIST Data set is usually used to measure the efficiency of an algorithm in classifying images, so it was chosen to be the data set to be classified.
### Or run the code
```
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))

test=datasets.MNIST("", download=True,train=False,
                    transform=transforms.Compose([transforms.ToTensor()]))
```

