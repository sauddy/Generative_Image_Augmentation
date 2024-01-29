# Genarative Modelling
Image augmentation using Generative Machine Learning



## 1. Generate synthetic images of protoplanetary disks hosting exoplanets using Generative modelling:
a. We train a Generative Adversarial Network to generate new images
b. We train a Diffusion model to generate new images 

Training data: Both the models were trained using ~ 100,000 images produced using FARGO3D hydrodynamics simulations + RADMC3D radiative transfer code 

## 2. Generative MODEL to rotate the protoplanetary disk images from any arbitrary orientation to face-on images: 
## we adapt the PIX2PIC code
a.  Rotate the protoplanetary disk images from any arbitrary orientation to face-on images 
b.  Generate synthetic radiative transfer images from Hydro output images (Done in a separate notebook) 

## 3. Generative MODEL to rotate the protoplanetary disk images from any arbitrary orientation to face-on images: 
## We adapt the PIX2PIX and add an ATTENTION module to improve performance

The following animation shows the training of the PIX2PIX Gan model where it rotates the input disks (oriented randomly in the sky)
to face-on-images

https://github.com/sauddy/Generative_Models/assets/46558389/552fb7e9-f802-48bb-92c0-c4f9eddd5a56

