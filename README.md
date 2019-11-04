# GANs

# General

## What is a GAN?
A General Adversarial Network is a method for approximating the underlying distribution of some data. So for example if you have a lot of images of cats, which is some kind of a distribution, a GAN would try to produce images of cat. The goal at the end is that you can't distinguish the "fake" images created by the GAN from the real images. 

## How does a basic GAN work?
In a GAN you have a Generator and a Discriminator who play a game. The Generator tries to produce real looking images. The Discriminator is shown real and fake images from the Generator and tries to distinguish between them. 


# Generative Adversarial Nets 
This was the classic paper introducing GANs by Ian Goodfellow. It stated the idea of using a Generator and a Discriminator playing a min-max Game for the first time. In this paper they showed that for a fixed G, the optimal Discriminator is: D<sub>G*(x) = p<sub>data / (p<sub>data+p<sub>g)
