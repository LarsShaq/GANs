# Everything about GANs as simple as possible

## Needed basic knowledge
### Entropy
Entropy is the expected Information of an event, which is disproportional to the uncertainty of it. You can imagine that if an event is really rare, the information of that event happening is really high. Formally it is: E(I(X)) = sum[-p(x)log(p(x))] , where I(x) = -log(P(x)) and the other part is just the formular for expectation. Some intuition why the information is the negative log:
1) log(1) = 0 -> An event which always happens has no information
2) log(x) >= 0 -> Information always positive
3) -log(p) = log(1/p) -> Information monotonically decreasing with p
4) If x1, x2 independent, information should add up: I(x1,x2) = -log(P(x1,x2)) = -log(P(x1)P(x2)) = -log(P(x1)) - log(P(x2)) = I(x1) + I(x2)  

### Measures of Divergence
Divergence in statics measures the "distance" between two distributions. So if we for example have one distribution where all the data is placed in one point and in the other distribution it is uniformly spaced, the divergence would be really high. If we have in contrast two Gaussian with similar mean and variance, the divergence would be low.

#### Kullback-Leibler Divergence
Formally the KL-Divergence for distributions p and q is: D(p||q) = sum[p(xi)(log(p(xi))-log(q(xi)))]
Intuitively it measures how much information we loose by approximating distribution p with q. 
This is a non-symmetrical divergence, which means if you measure the divergence from p to q, you would get a different result than measuring the divergence from q to p. 

#### Jensen-Shannon Divergence
Formally: JSD(p||q) = 0.5 KL(p||r) + 0.5 KL(q||r) where r=(p+q)/2. This is kind of a middle ground between the non-symmetric KL(p||q) and KL(q||p). 

#### Earth Mover Distance

### Mutual Information
Mutual information measures the amount of information learned from knowledge of one random variable Y about another random variable X. Formally: I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) where H is the entropy.
It can also be formulated as the KL between the joint distribution of p(x,y) and the joint distribution if the variables were independent p(x)p(y). So if the variables are independent the joint distribution is exactly p(x)p(y) and the KL divergence is 0, so the mutual information as well. If having X you know everything about Y, than the Entropy/information H(X|Y) is zero, because there is no uncertainty given Y. This comes down to P(x|y) having a small entropy. 

## General

### What is a GAN?
A General Adversarial Network is a method for approximating the underlying distribution of some data. So for example if you have a lot of images of cats, which is some kind of a distribution, a GAN would try to produce images of cat which look like the images you have. The goal at the end is that you can't distinguish the "fake" images created by the GAN from the real images. 

### What is the main principle of a GAN?
In a GAN you have a Generator and a Discriminator, which are neural networks, who play a game. The Generator tries to produce real looking images. The Discriminator is shown real and fake images from the Generator and tries to distinguish between them. 
The Generator starts from some random numbers z and performs common neural network operations on it while upsampling it to the image resolution. Then the Discriminator, which is similar to common classification neural networks gets fed with the generated fake images and real images. If the image is real, D should output 1, if not, 0.

### What is the basic loss formulation for a GAN?
In the basic Loss-Formulation D tries to maximize the expected value over the real distribution of log(D(x)) and the expected value over the fake distribution of log(1-D(G(z))). The Generator tries to minimize the expected value over his generated samples of log(1-D(G(z))). A few things to notice: 
- The expected value of a function f(x) over a distribution is: sum[p(x)f(x)] (in discrete setting). We don't explicitly use a p(x), but by sampling over the data points (or latent space z) we get an approximation of the expected value of the function. So in practise, just think of the loss as log(D(x))+log(1-D(G(z))) calculated over a minibatch. 
- In practise instead of minimizing log(1-D(G(z))) the Generator tries to maximize log(D(G(z))). This has to do with the gradients. In the beginning, when G is bad, D can easily distinguish between fake and real. So log(1-D(G(z))) becomes approximately log(1), which if you look at the log function has a small gradient. The log of a small value thus has a high gradient for learning, so we use log(D(G(z))).
- In Neural Networks we always try to minimize a cost function. To turn the max poblems we just formulated into min problems you just multiply it by -1. You can see that now we have -log(D)*y + -log(1-D)*(1-y) where y=1 for real data and y=0 for fake data, which is the cross entropy loss formulation. 

### What are alternatices for GANs?
//Todo
Restricted BM, deep BM and Deep Belief Networks, VA

### Why are GANs so hard to train?
//Todo
### What is the latent space?
//Todo
### What are some metrics for measuring the performance of GANs?
//Todo

## Papers

### Generative Adversarial Nets - 2014
This was the classic paper introducing GANs by Ian Goodfellow. It stated the idea of using a Generator and a Discriminator playing a min-max Game for the first time. In this paper they showed that for a fixed G, the optimal Discriminator is: D<sub>G</sub>(x) = p<sub>data</sub> / (p<sub>data</sub>+p<sub>g</sub>).
Furthermore they showed that the global minimum of the loss is achieved, when pg=pd and the that under perfect conditions the GAN converge to the optimum. Minimizing there proposed loss is equivalent to minimizing the Jensen-Shannon divergence. 

### Conditional GAN - 2014
They additionally fed the label y to G and D by adding the labels to the noise vector for G and to the input image for D. In this way they were able to generate Images conditioned on a label. The image quality was slightly worse than in the original. 

### DCGAN - 2016
It proved difficult to scale GANs using CNN. So the authors did a lot of empirical work to find some guidelines for architectures for training Deep convolutional GANS. DCGAN is not a special architecture, which I thought at the beginning reading about it, but rather a family of architectures following the guidelines.
The Guidelines are:
1) Replace Max Pool with strided Convolution
2) Use Batchnorm (Except G Output Layer and D input Layer)
3) Eliminate Fully Connected Layers on top of conv. features
4) Use leaky relu as activation functions

They also used the trained discriminator as a feature extractor for classification and achieved good results. 
Furthermore they did some analysis of the latent space. They showed that the trainsitions are smooth in latent space, which means the network doesnt memorize. For memorization you would see sudden changes. 
They also did vector arithmetic on latent space to see for example man+glasses-man+woman = images of woman + glasses

### Learning from Simulated and Unsupervised Images through Adversarial Training - 2017
The authors used a GAN to refine a simulated image to look more real to use it for training while maintaining the annotation. Concretly they used images of eyes with the annotation where they are looking.  
For this they combined a loss for the "realness" and one for maintaining the corectness of the annotation. 
Especially they used two techniques to make the results better:
1) They used local adversarial loss, using image patches to train the network. With this they wanted to avoid, that the network concentrates too much on single features
2) Use a history of faked images to update discriminator. The logic behind this is that G sometimes uses the same tricks again to try to fool D, so by giving D the history he shouldnt fall for it again 

### Improved Techniques for Training GANs - 2016
Training GANs is finding a Nash Equilibrium between D and G. A Nash Equilibrium is a point where both parties have a optimal strategy for this point, and if one partner diverges from the strategy he will be off worse. Achieving this Equilibrium is very difficult, since it is a non-convex, high dimensional problem with continous parameters. 
In this paper, the authors developed some useful techniques for training GANs better:

1) Feature Matching
To avoid that G overfits on D, a new objective function is propesed for G. The "fake" outputs of G should follow a certain statistic, which is given by the the calculated activations of a layer of D. These activations/ features should be similiar between real and fake data. So the distance between this expected activations for the real data and for the created is the objective function. D is trained as usually.

2) Minibatch Discrimination
*One of the main failure modes for GAN is for the generator to collapse to a parameter setting where it always emits the same point. When collapse to a single mode is imminent, the gradient of the discriminator may point in similar directions for many similar points. Because the discriminator processes each example independently, there is no coordination between its gradients, and thus no mechanism to tell the outputs of the generator to become more dissimilar to each other. Instead, all outputs race toward a single point that the discriminator currently believes is highly realistic. After collapse has occurred, the discriminator learns that this single point comes from the generator, but gradient descent is unable to separate the identical outputs. The gradients of the discriminator then push the single point produced by the generator around space forever, and the algorithm cannot converge to a distribution with the correct amount of entropy. An obvious strategy to avoid this type of failure is to allow the discriminator to look at multiple data examples in combination, and perform what we call minibatch discrimination.*
To avoid that, they make sure D gets some information about the similarity of examples in a batch, so that it can use this additional information to discriminate more easily. You can imagine that if it sees all the examples look the same, it can say its from G because mode collapse happend. To achieve that they do the following:
They take a feature vector from some Layer of D. They multiply this feauture vector by a tensor to get a matrix M. (What is this Tensor?) They calculate the L1 distance of the rows of this Matrix to the rows of the matrix from every other sample of the batch. Then they take the negative exponential of it (I assume to make the values smaller to fit better in range of activations). Now they have for for every pair of samples in the batch B (number of rows of the Matrix M) numbers of similarity. For every sample they add up the pairs to all the other samples to obtain B values, which is the final vector they concatenate again to the feature vector. They do this for the real data and the G data batch.

3) Historical Averaging
They add a additional loss to the two players: the distance between the parameters and the historical average of the parameters. They found this was able to solve non-convex toy problems.

4) One-sided label smoothing
*Label smoothing, a technique from the 1980s recently independently re-discovered by Szegedy et. al, replaces the 0 and 1 targets for a classifier with smoothed values, like .9 or .1, and was recently shown to reduce the vulnerability of neural networks to adversarial examples.*
Here they use one-sided smoothing, meaning the negative label still stays 0. They provide an explanation for that, which I haven't fully understood yet.

5) Virtual Batch Normalization
In common Batch Norm, the output of a network depends also on the other sample in the batch. (Why is that here a problem?)
So they suggest using a reference batch. The reference batch are just some samples choosen at the beginning. Then they always run the reference batch and the current batch through the network. For every sample of the current batch they use the values of the reference batch and the current sample to normalize. As you can easily see, this has the disadvantage of having to run forward pass twice. So they only use it on G.

This paper also had the idea of the inception score to get an automatic score for the realness of the data (See above). 

### InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets - 2016
This paper shows how to have single variables of the latent space have an explicit effect on the generated output. So for example if you generate a number image, one variable of the latent space would change which number is shown and a second variable is responsible for rotating the number a little. 
To achieve this, they split the usual noise vector at the beginning in a noise part and a so called latent code c. The latent code can have categorial variables (like integeres 1-10) or continues (like -1 to 1). In the traditional GAN there is no garuantee that the latent code has any meaning, the GAN could simply ignore it. To make sure the code has a significant meaning, they try to maximize the mutual information (see above) between the code and the generated output. So they add -lambda I(c,G(z,c)) to the loss function, where lambda is hyperparameter(- because we want to maximize instead of minimize).
For maximizing I(c,x), where x is given by G(z,c), you need to minimize the entropy of p(c|x) (make information of c given x small, because by knowing x you already know a lot about c). In practise you don't have access to p(c|x). But you can obtain a lower bound by an auxiliary distibution Q. 
This auxiliary distribution Q is a neural network. It simply shares all convolutional layers of D and adds a final fully connected layer at the end. The mutual information part in the loss always converges faster than the normal GAN, so it doesnt really add an overhead to the traditional GAN.   

## Open questions

### Could you start with z at image resolution and just use dilated convolution? If so, could you make z at the beginning similar to real images to make it easier for G at the beginning?
