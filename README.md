# Generative Modeling Using GAN and Diffusion Algorithms 
Image augmentation using Generative Machine Learning



### 1. Image Generation: Generate synthetic images of protoplanetary disks (PPDs) hosting exoplanets using Generative modeling:
a. We train a Generative Adversarial Network(GAN) to generate new images of PPDs

b. We train a Diffusion model to generate new images 

Training data: Both the models were trained using ~ 100,000 images produced using FARGO3D hydrodynamics simulations + RADMC3D radiative transfer code 

### 2. Image Rotation: Generative MODEL to rotate the protoplanetary disk images from any arbitrary orientation to face-on images: we adopt the PIX2PIX code
 

The following image demonstrates an example case, where the GAN model rotates the input image to generate a face-on image

![sample_image](https://github.com/sauddy/Generative_Models/assets/46558389/06ec2fa4-8b82-4a23-8479-e335ce7140dc)

The following animation shows the training of the PIX2PIX Gan model where it rotates the input disks (oriented randomly in the sky)
to face-on-images
https://github.com/sauddy/Generative_Models/assets/46558389/0d614a57-402d-4069-b49d-be822d50e46e

### 3. Attention based Image Rotation: Generative MODEL to rotate the protoplanetary disk images from any arbitrary orientation to face-on images: We modify the PIX2PIX code and add an ATTENTION module to improve performance

The following image demonstrates an example case, where the GAN with self_attenstion model rotates the input image to generate a face-on image



### 4. Radiative Transfer:  Generate synthetic radiative transfer images from Hydro output images 





