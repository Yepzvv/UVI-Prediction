# UV Index Prediction with TransNN

This project introduces a Transformer-based neural network model TransNN designed to predict the UV index, providing valuable insights for public health and environmental monitoring.

## Overview
TransNN utilizes the Transformer architecture to capture temporal dependencies and complex patterns in UV radiation data, enhancing the accuracy of UV index predictions.

## Model Architecture

Below is a diagram illustrating the architecture of our Transformer-based model for UV index prediction.

![framework](https://github.com/Yepzvv/UVI-Prediction/assets/171041111/0fff390c-6cdc-438c-8f85-0fc751d60ebe)

## Getting Started

### Prerequisites
- Python 3.6 or later
- PyTorch
- NumPy
- scikit-learn

You can install the required packages using pip:
```
pip install torch numpy scikit-learn
```

### Installation
Clone the repository and navigate into the project directory:
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

## Data
The dataset includes historical UV index records along with associated meteorological factors such as temperature, humidity, and cloud cover.

## Model Description
The `TransNN` model is adapted from the standard Transformer architecture, featuring:
- Input dimension matching the number of features in the dataset.
- A specified number of attention heads and encoder layers.
- Linear layers for regression output.

## Usage
Run the `main.py` with the following command:
```bash
python main.py
```

## Results
The model achieves the following performance metrics on the test set:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R^2) Score

## Acknowledgements
- Thank contributors, acknowledge any datasets or libraries used in the project.
