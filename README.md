# pix2pix-Pytorch

Conditionsal Generative Adversarial Networks are an extensions to normal GANs where the targeted generated image is conditioned on some condition. This condition helps us in controlling what output we want to generate.
 Pix2Pix is an **image-to-image translation** process based on **Conditional GAN** where a target image is generated, that is conditioned on a given input image. 
Here in this project Pytorch implementation of Pix2Pix model from scratch has been done. Aim is to generate **streetmap images** of a corresponding **satellite image** using Pix2Pix model.
The project implementation follows the original paper on Pix2Pix


# Dataset
The dataset is publicly available in Kaggle and can be downloaded from here [https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset]<br>
It consists of satellite images along with their ground truth streetmap images<br>
Dataset is divided into train and validation folders with each set having 1096 images<br><br>
![pix2pix_eg](https://github.com/arka57/pix2pix-Pytorch/assets/36561428/a1f81ce8-1e4a-4902-a510-11c36f74ef3a)<br><br>
Example of a satellite image and its corresponding ground truth streetmap image

# Model Architecture
Here a conditional Generative adversarial network is implemented where generated images are conditioned on some input and here it is the input image itself<br><br>
![8109b440-49ad-491c-b72a-3f0768a6256b](https://github.com/arka57/pix2pix-Pytorch/assets/36561428/ad7f220f-7c74-4fa0-9a80-a95e7437150e)<br><br>
Above figure shows the conditional GAN architecture . Here y is the condition on which the generator and discriminator are conditioned. In these case y is the input image(satellite image) iteslf.
<br><br>

The **generator** resembles a U-net architecture where input to generator is RGB image and it tries to generate another RGB image of same shape
The **discriminator** is Patch GAN which outputs a 30X30 matrix where each cell classifies that part of the image whether it is real or fake generated<br><br>
![unet](https://github.com/arka57/pix2pix-Pytorch/assets/36561428/794d6973-5fae-43cb-b7b4-7a0c2231aedd)
<br><br>Above figure shows the U-net architecture where we go through downsampling followed by upsampling and adding the previous skip connections<br><br>
![discriminator](https://github.com/arka57/pix2pix-Pytorch/assets/36561428/e67ffa70-2e4b-4308-a568-82b421142505)
Above figure shows the patch GAN architecture where output is 30x30<br><br>
Along with normal **adversarial** loss for training the generator block uses an added **L1** loss function for evaluating how close the generated streetmap are close to actual streetmap

# Hyperparameters
EARNING_RATE = 2e-4<br>
BATCH_SIZE = 16<br>
NUM_WORKERS = 2<br>
IMAGE_SIZE = 256<br>
CHANNELS_IMG = 3<br>
L1_LAMBDA = 100<br>
LAMBDA_GP = 10<br>
NUM_EPOCHS = 500<br>

# Result

# References
Pix2Pix paper:  https://arxiv.org/abs/1611.07004

