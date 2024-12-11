# Plant Disease Detection Model

This project focuses on building a deep learning model to classify plant diseases using image data. The dataset is sourced from the `New Plant Diseases Dataset (Augmented)`.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Code Description](#code-description)
- [Usage](#usage)
- [Model Details](#model-details)
- [Saving and Loading](#saving-and-loading)
- [Acknowledgements](#acknowledgements)

## Project Overview

The project trains a Convolutional Neural Network (CNN) model using TensorFlow and Keras to classify plant diseases into one of 38 predefined classes. The model predicts plant diseases based on leaf images and provides the predicted class along with a confidence score.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Ipywidgets
- Open Datasets
- Pickle

Install the required packages using:

```bash
pip install tensorflow numpy pandas matplotlib ipywidgets opendatasets
```

## Dataset

The dataset contains images of plants categorized by disease type and health status:

- Training images are located in `./new-plant-diseases-dataset/.../train`.
- Validation images are located in `./new-plant-diseases-dataset/.../valid`.

### Dataset Preprocessing

The code checks for missing or invalid image files and filters out hidden files.

## Code Description

1. **Data Loading and Preprocessing**:

   - The dataset is loaded using `image_dataset_from_directory`.
   - Images are resized to `(256, 256)` and normalized using `Rescaling`.
   - Normalization is applied using the `Rescaling` layer in TensorFlow, which scales pixel values from the range `[0, 255]` to `[0, 1]`. This helps improve the stability and efficiency of the training process.

2. **Model Architecture**:

   - The CNN model consists of convolutional, pooling, and dense layers with a final `softmax` activation for classification.

3. **Visualization**:

   - Sample images from the dataset are displayed with Matplotlib.

4. **Training**:

   - The model is trained with `sparse_categorical_crossentropy` as the loss function and `Adam` optimizer.

5. **Prediction**:

   - A file upload widget allows users to test the model with custom images.

## Usage

### Training the Model

Run the script to train the model on the training dataset. The training progress, including accuracy and loss, is logged.

### Testing the Model

Upload an image using the file upload widget. The model predicts the disease class and displays the confidence score.

### Example:

```python
model.evaluate(example_batch_images, example_batch_labels, verbose=False)
```

### Visualization:

Use Matplotlib to visualize sample images and their corresponding labels.

## Model Details

- **Input Shape**: `(256, 256, 3)`
- **Output Shape**: `38 classes`
- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: `Adam`
- **Normalization**: Pixel values are normalized to the range `[0, 1]` using the `Rescaling` layer to improve model training.

### Saving and Loading

- Save the trained model:
  ```python
  model.save('models/plant_disease_model.h5')
  ```
- Save the weights:
  ```python
  model.save_weights('models/plant_disease_weights.h5')
  ```
- Save training history:
  ```python
  with open('training_history.pkl', 'wb') as file:
      pickle.dump(history.history, file)
  ```

## Acknowledgements

The dataset and inspiration for this project come from the Plant Pathology community and Kaggle contributors.

