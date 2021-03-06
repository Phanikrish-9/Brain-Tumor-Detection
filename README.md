# Brain Tumor Detection using Convolutional Neural Network
The project is for a research of Medical engineering at Hospital of National Taiwan University.
## Why this project?
In clinical diagnosis, checking brain tumor among a large amount of MRI images usually takes doctors much time. For example, in samples of this project, a patient has about 200 MRI images, but tumor tissues only appears in 15 images. Therefore, this project aims to automatically detect tumor tissues in large amount of MRI image data.

### Our targets:

 * automatically detect if tumor tissue appears in the MRI image
 * automatical brain tumor segmentation in MRI image

## Requirements
* NumPy >= 1.12.0
* keras >= 2.1.0
* TensorFlow >= 1.4 (the project hasn't been tested in other version)
## File Description
* ***Brain tumor.py*** includes all parameters.
* ***Brain tumor.ipynb*** includes functions regarding loading, batching data, CNN model architecture and log files saving.
## Usage
 * Step1. adjust parameters in Brain tumor.py including data filenames, training parameters.
 * Step2. adjust model parameters in Brain tumor.ipynb
 * Step3. start training
 * Step4. inference and demo
