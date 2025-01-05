# VIN Code Symbol Classifier Documentation

## Data Used
- **Dataset**: [EMNIST](https://www.kaggle.com/datasets/crawford/emnist "EMNIST") (Balanced training and testing datasets).
- **Preprocessing**:
  - Lowercase letters were removed.
  - The letters `I`, `O`, and `Q` were removed from the data because they are not used in VIN codes.
- **Classes**: The dataset includes 33 classes comprising digits (`0-9`) and uppercase English letters (excluding `I`, `O`, and `Q`).
- **Datasets Volume**: Each class contains 2400 examples of data in the training dataset and 400 examples in the testing dataset.

## Model Architecture
The model is a Convolutional Neural Network (CNN), structured as follows:

```python
class SymbolClassifier(nn.Module):
    def __init__(self):
        super(SymbolClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=6, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=6, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 33)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

- **Convolution Layers**: Two convolutional layers with `ReLU` activation functions.
- **Pooling Layers**: Max-pooling layers are used after each convolutional layer to reduce feature map dimensions.
- **Parameter Tuning**: The model`s hyperparameters (e.g., kernel size, number of channels) were adjusted to get better accuracy.

## Training and Evaluation
Epoch [1/10], Loss: 0.5109, Accuracy: 90.56818181818181%
Epoch [2/10], Loss: 0.2451, Accuracy: 91.98484848484848%
Epoch [3/10], Loss: 0.1986, Accuracy: 92.71969696969697%
Epoch [4/10], Loss: 0.1751, Accuracy: 93.4090909090909%
Epoch [5/10], Loss: 0.1559, Accuracy: 92.8030303030303%
Epoch [6/10], Loss: 0.1425, Accuracy: 93.18181818181819%
Epoch [7/10], Loss: 0.1315, Accuracy: 93.33333333333333%
Epoch [8/10], Loss: 0.1217, Accuracy: 93.56060606060606%
Epoch [9/10], Loss: 0.1151, Accuracy: 93.51515151515152%
Epoch [10/10], Loss: 0.1073, Accuracy: 93.43939393939394%

## Usage Instructions
To use the model for inference, run the script `inference.py`.

### Running Directly from Command Line:
- Use `inference.py` with the following command:

  ```bash
  python inference.py --input [path_to_folder]
  ```
- The script outputs have the format: 
`[character ASCII index in decimal format], [POSIX path to image sample]`.
