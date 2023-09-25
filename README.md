# Avatar-Generation-from-Real-Images-using-Deep-Learning


_Generate Cartoon Images using [DC-GAN](https://arxiv.org/abs/1511.06434)_

Deep Convolutional GAN is a generative adversarial network architecture. It uses a couple of guidelines, in particular:

- Replacing any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Using batchnorm in both the generator and the discriminator.
- Removing fully connected hidden layers for deeper architectures.
- Using ReLU activation in generator for all layers except for the output, which uses tanh.
- Using LeakyReLU activation in the discriminator for all layer.
### GAN Model

1. Define Generator and Discriminator network architecture
2. Train the Generator model to generate the fake data that can fool Discriminator
3. Train the Discriminator model to distinguish real vs fake data
4. Continue the training for several epochs and save the Generator model
![Model]https://github.com/vinyvn/Avatar-Generation-from-Real-Images-using-Deep-Learning/blob/main/assets/images/GAN-architecture.png
### Dataset Setup
### Dataset Setup

[Cartoon Set](https://google.github.io/cartoonset/) which is a collection of random 2D cartoon avatar images.
Download the dataset using the shell script.

```
sh download-dataset.sh
```

This will download the dataset in `data/` directory.
If you want to train the model in Google Colab, upload the dataset folder to Google Drive. The destination path should be `projects/cartoons/`
### Model Training

Check out the model being trained to generate cartoon images.
![Training]https://github.com/vinyvn/Avatar-Generation-from-Real-Images-using-Deep-Learning/blob/main/assets/images/GAN-training.gif
### Model Prediction
https://github.com/vinyvn/Avatar-Generation-from-Real-Images-using-Deep-Learning/blob/main/assets/images/GAN-output.png    
Avatar Generator Streamlit App
This Streamlit application allows you to generate cartoon avatars from your uploaded photos using a Deep Convolutional Generative Adversarial Network (DCGAN) model.

How to Use
Upload Your Photo: Click on the "Choose an image..." button to upload your photo. The image should be in JPG, JPEG, or PNG format.

Select the Epoch Value: Use the number input field to enter the epoch value. This determines which version of the trained model to use for avatar generation. You can experiment with different epochs to see how the results vary.

Generate Avatar: Click the "Generate Avatar" button to create a cartoon avatar based on your uploaded photo.

View Generated Avatar: Once generated, the avatar will be displayed in the app.

About the Model
The avatar generation model is based on a Deep Convolutional Generative Adversarial Network (DCGAN). It has been trained on a dataset of cartoon avatars and has learned to generate cartoon-style images.

Requirements
Python 3.7+
TensorFlow (for loading and running the DCGAN model)
Streamlit (for creating the user interface)
Pillow (PIL) for image processing
How to Run
Make sure you have the required Python libraries installed. You can install them using pip:
pip install streamlit tensorflow pillow
Clone this repository or download the script.

Run the Streamlit app using the following command:
streamlit run avatar_generator.py
The app will open in your default web browser, and you can start generating avatars.
Credits
DCGAN model for avatar generation
 [(https://github.com/aakashjhawar/AvatarGAN)https://github.com/aakashjhawar/AvatarGAN]



