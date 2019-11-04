# GANs

# Needed basic knowledge
## Entropy
Entropy is the expected Information of an event, which is disproportional to the uncertainty of it. Assume you toss a coin. The result of the event doesn't give you that much of information, because the uncertainty is just 0.5. If you instead get the result of rolling a dice, the nformation of this event is much higher.  

## Measures of Divergence
Divergence in statics measures the "distance" between two distributions. So if we for example have one distribution where all the data is placed in one point and in the other distribution it is uniformly spaced, the divergence would be really high. If we have in contrast two Gaussian with close mean and variance, the divergence would be low.

### Kullback-Leibner Divergence
This is a non-symmetrical divergence, which means if you measure the divergence from p to q, you would get a different result than measuring the divergence from q to p. 

## Jensen-Shannon Divergence

## Earth Mover Distance

# General

## What is a GAN?
A General Adversarial Network is a method for approximating the underlying distribution of some data. So for example if you have a lot of images of cats, which is some kind of a distribution, a GAN would try to produce images of cat. The goal at the end is that you can't distinguish the "fake" images created by the GAN from the real images. 

## How does a basic GAN work?
In a GAN you have a Generator and a Discriminator, which are neural networks, who play a game. The Generator tries to produce real looking images. The Discriminator is shown real and fake images from the Generator and tries to distinguish between them.  

## What is the basic loss formulation for a GAN?
In the basic Loss-Formulation

## What are alternatices for GANs?
Restricted BM, deep BM and Deep Belief Networks

## Why are GANs so hard to train?

## What is the latent space?

# Papers

## Generative Adversarial Nets - 2014
This was the classic paper introducing GANs by Ian Goodfellow. It stated the idea of using a Generator and a Discriminator playing a min-max Game for the first time. In this paper they showed that for a fixed G, the optimal Discriminator is: D<sub>G</sub>(x) = p<sub>data</sub> / (p<sub>data</sub>+p<sub>g</sub>).
Furthermore they showed that the global minimum of the loss is achieved, when pg=pd and the that under perfect conditions the GAN converge to the optimum. Minimizing there proposed loss is equivalent to minimizing the Jensen-Shannon divergence. 

## Conditional GAN - 2014
They additionally fed the label y to G and D by adding the labels to the noise vector for G and to the input image for D. In this way they were able to generate Images conditioned on a label. The image quality was slightly worse than in the original. 

## DCGAN - 2016
It proved difficult to scale GANs using CNN. So the authors did a lot of empirical work to find some guidelines for architectures for training Deep convolutional GANS. DCGAN is not a special architecture, which I thought at the beginning reading about it, but rather a family of architectures following the guidelines.
The Guidelines are:
1) Replace Max Pool with strided Convolution
2) Use Batchnorm (Except G Output Layer and D input Layer)
3) Eliminate Fully Connected Layers on top of conv. features
4) Use leaky relu as activation functions

They also used the trained discriminator as a feature extractor for classification and achieved good results. 
Furthermore they did some analysis of the latent space. They showed that the trainsitions are smooth in latent space, which means the network doesnt memorize. For memorization you would see sudden changes. 
They also did vector arithmetic on latent space to see for example man+glasses-man+woman = images of woman + glasses

## Learning from Simulated and Unsupervised Images through Adversarial Training
The authors used a GAN to refine a simulated image to look more real to use it for training while maintaining the annotation. Concretly they used images of eyes with the annotation where they are looking.  
For this they combined a loss for the "realness" and one for maintaining the corectness of the annotation. 
Especially they used two techniques to make the results better:
1) They used local adversarial loss, using image patches to train the network. With this they wanted to avoid, that the network concentrates too much on single features
2) Use a history of faked images to update discriminator. The logic behind this is that G sometimes uses the same tricks again to try to fool D, so by giving D the history he shouldnt fall for it again 
