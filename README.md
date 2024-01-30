# Genarative Modelling
Image augmentation using Generative Machine Learning



### 1. Generate synthetic images of protoplanetary disks (PPDs) hosting exoplanets using Generative modeling:
a. We train a Generative Adversarial Network(GAN) to generate new images of PPDs

b. We train a Diffusion model to generate new images 

Training data: Both the models were trained using ~ 100,000 images produced using FARGO3D hydrodynamics simulations + RADMC3D radiative transfer code 

### 2. Generative MODEL to rotate the protoplanetary disk images from any arbitrary orientation to face-on images: we adopt the PIX2PIX code
a.  Rotate the protoplanetary disk images from any arbitrary orientation to face-on images 

b.  Generate synthetic radiative transfer images from Hydro output images (Done in a separate notebook) 

### 3. Generative MODEL to rotate the protoplanetary disk images from any arbitrary orientation to face-on images: We modify the PIX2PIX code and add an ATTENTION module to improve performance

The following animation shows the training of the PIX2PIX Gan model where it rotates the input disks (oriented randomly in the sky)
to face-on-images




https://github.com/sauddy/Generative_Models/assets/46558389/0d614a57-402d-4069-b49d-be822d50e46e

