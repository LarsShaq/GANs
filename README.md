# GANs

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
This paper shows how to have single variables of the latent space have an explicit effect on the generated output without any supervision. So for example if you generate a number image, one variable of the latent space would change which number is shown and a second variable is responsible for rotating the number a little. 
To achieve this, they split the usual noise vector at the beginning in a noise part and a so called latent code c. The latent code can have categorial variables (like integeres 1-10) or continues (like -1 to 1). In the traditional GAN there is no garuantee that the latent code has any meaning, the GAN could simply ignore it. To make sure the code has a significant meaning, they try to maximize the mutual information (see above) between the code and the generated output. So they add -lambda I(c,G(z,c)) to the loss function, where lambda is hyperparameter(- because we want to maximize instead of minimize).
For maximizing I(c,x), where x is given by G(z,c), you need to minimize the entropy of p(c|x) (make information of c given x small, because by knowing x you already know a lot about c). In practise you don't have access to p(c|x). But you can obtain a lower bound for the mutual information by approximmating p(c|x) with an auxiliary distibution Q. The lower bound is then given by log(Q(c'|x)) where they show that you can sample c' from P(X|c).    
This auxiliary distribution Q is approximated by a neural network. It simply shares all convolutional layers of D and adds a final fully connected layer at the end. In practise you have two fully connected layers at the end, one for categorial latent codes and one for continous. For the categorial latent code you use a softmax non-linearity like in the traditional classification problems, which is pretty much what it is. For continous variables they found that a factored Gaussian is sufficient. That means that you make an estimate for the my and the variances (or sometimes just left one) of the Gaussian and compare it to the real value. Factorial means that you multiply the Gaussians of all the continous variables, which in log space is a sum.           
The mutual information part in the loss always converges faster than the normal GAN, so it doesnt really add an overhead to the traditional GAN. 

### Perceptual Losses for Real-Time Style Transferand Super-Resolution - 2016
Here they use dont use GAN but a single feedforward network to make style transfers. The interesitng part is the perceptual loss they use, which is used in many GAN papers as well. If you just use L1 loss on pixels, two images that are conceptually the same but differ in the exact pictures can have a high loss. The features of a network are a better choice, since it corresponds to more higher level attributes that should be the same. Therofre they use a separate loss network to compute the feature loss of their transfer network. Especially the loss network is a vgg pretrained on Imagenet. 
From this they define two features losses:
1) Feature Reconstrucion Loss which is a L2 loss of different layers of the network. For this they feed both vgg and the transfer network the same input. The output should be conceptually the same, like both having a ceratin dog. If you use lower layer features the image become very similar, with higher level features the content and overall structure are kept. 
2) Style Reconstruction Loss which forces the produced image on the style of a new image. For this they compute the Gram Matrix, which basically is an elementwise product over all spatial locations of each channel with the ones from the channel of the other image. It basically is like covariance. This loss only keeps style. (think about why)  

### Conditional Image Synthesis with Auxiliary Classifier GANs (ACGAN) - 2017
It is very hard to generate images from a distribution with a lot of variety such as Image Net with is 1000 class labels. In this paper they are able to achieve 128x128 images of Image Net. As in Conditional GAN and other papers it prooved useful to feed G and D additional data. In their approach, they feed G additional data but for D they let it deconstruct the additional data. In this case the additional data were the class labels, and such a additional loss term was added to the GAN formulation. It is not that much of a crazy idea or change of the GAN, but it showed to give really good results. Especially they trained 100 different GANs each for 10 classes to generate all the classes of Image Net. They analyzed the results with some new metrics they thought useful, but I wont go into detail of that. 

### The Effectiveness of Data Augmentation in Image Classification using DeepLearning - 2017
They compare different augmentation strategies like classic transformations, GAN and learning a style transfer neural network. But for GAN they only use cycle gan not creating other realistic images, but rather images in a different style. Traditional and the learned transfomrations on a neural net work the best. 

### Deconvolution and Checkerboard Artifacts
Good reading material: https://distill.pub/2016/deconv-checkerboard/

### Synthesizing retinal and neuronal images with generative adversarial nets - 2018

### SYNTHETIC DATA AUGMENTATION USING GAN FOCLASSIFICATIONR IMPROVED LIVER LESION CLASSIFICATION - 2018
In medical applications there is often a lack of data. They were able to improve there classification accuracy significantly by using synthetically augmentated images from GAN. They trained a GAN for each of the three classes separatly. Since they didnt have that much images to train the GAN, they used a lot of augmentated images (flipped, rotated etc) to train the GAN.

### GAN Augmentation: Augmenting Training Datausing Generative Adversarial Networks - 2018
They tested the benefits of data augmentation on medical images. They used progressive growing of GANs network to produce the images. The results were that the augmentation with GAN almost always benfited the segmentation training. The smaller the original dataset, the better the results. In one case on MRI images the additional data worsened the performance when using all original data. The GAN was the most beneficial for underrepresented classes. The traditional and GAN augmentation helped each other, bringing better results using them both then the combined single results.

### Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks Jun-Yan (CycleGAN)- 2018
![alt text](https://github.com/LarsShaq/GANs/blob/master/images/cycleGAN.png)
In this paper they dont need paired training examples for image translation. Rather they are able to convert images from one distribution, e.g. segmentation map to another e.g. images. Best explained with the image. THe additional thing to the normal GAN loss for G and F is the cycle consisitency loss, making sure that F(G(x))=x and G(F(y))=y by taking L1 norm. Instead of normal GAN loss use least square loss and use the trick of taking historyof images for D.

### Pros and Cons of GAN Evaluation Measures - 2018
Compares a lot of different measures, just will mention a few. Inception score is classic, but only works for classification stuff by using Inception net trained on image net. Frechet Inception Score compares statistics of feature space of generated samples and real ones. To detect mode collapse: Birthday Paradox test, needs manual inspection for duplicates. 
Classifier Two-sample test: generator tested on held out test set, which is used to train a new D. 
FCN: feed generated input to segmentation net and measure difference between gt and calculated segmentation map
+++ a lot to read, read again when needed+++

### An empirical study on evaluation metrics ofgenerative adversarial networks - 2018
Compare some common evaluation metrices: Inception score, Kernel MMD, Wasserstein Distance, Frechet Inception Distance, 1NN classifier. Results:
- Inception SCore: good correlation with quality and diversity. But only evaluates generator distribution and not the similarity to real distribution. So any sharp real looking picture would add high to the score. Unable to detect overfitting.
- Kernel MMD: works suprisingly well when operates in feauter space of pretrained net. Recommended by authors.
- Wasserstein Distance: works well in suitable featuer space, but high complexity
- Frechet Inception Distance: not a lot of info there, works good
- 1NN classifier: Really good measure as well. Recommended together with Kernel
Published Code to use the metrices. Furthermore authors find that overfitting of GAN to training data does not happen only if very few examples. 

### Spectral Normalization for Generative Adversarial Networks
They find a new way of making D Lipschitz constant. It outperfroms WGAN-GP while also being computationally more efficient. They tested it with normal GAN loss and hinge loss, hinge loss better results. Spectral normalization computationally costly to calculate but they provide an efficient algorithm for approximating. Need to read again for all the theoretical details. But seems promising, especially since one of the best results so far used it, like bigGAN.  

### Progresive Growing of GANS for improved quality, stability, and variation - 2018
The main contribution is that they use progressive growing of GANs for creating high resolution images. They always use a G and D with symmetric architecture. They start at images of 4x4 resolution and with each stage they smoothly add layers to G and D. The logic is that there is less class information and fewer modes at low resolution making the training more stable. The next problem of higher resolution than is always easier mapping from lower resolution to slightly higher then directly mapping to high resolution. Training time also is lower. 
They furthermore introduce some tricks to improve the training:
- They increase variation using minibatch standard deviation. This idea is similar to minibatch discrimination from Improved Techniques for Training GANs. Here they use a simpler, better approach: They compute the standard deviation for each feature in each spatial location over the minibatch. They then average these estimates over all features and spatial locations to arrive at a single value.  We replicate the value and concatenate it to all spatial locations and over the minibatch, yielding one additional (constant) feature map and add it somewhere to D. 
- Since GANs are prone to the escalation of signal magnitudes as a result of unhealthy competition between the two networks they take measures against this. Especially they observe that mode collapses tend to happen very quickly, over the course of a dozen minibatches.  Commonly they start when the discriminator overshoots,  leading to exaggerated gradients,  and an unhealthycompetition follows where the signal magnitudes escalate in both networks. Usually a common measure are variants of batch normalization but they were originally introduced to eliminate covariate shift which doesnt not seem to be a problem according to their observations. Here they use two approaches. First, they intialize the weights following a normal distribution and use the logic from He initializer at runtime to scale the weights. This has to do with some attribute of Adam and RMSProp optimizer. (Look at it again) Secondly they normalize the feature vector of G  after each convolution to keep the magnitude from going out of control.
Furthermore they propose some new metric. 

### BAGAN: Data Augmentation with Balancing GAN - 2018
![alt text](https://github.com/LarsShaq/GANs/blob/master/images/BAGAN.png)
In this paper they use GANs for Data Augmentation for classes which are underrepresentated. A big problem for this is that you only have a few images of the underrepresentated class, most often not enough to train the GAN well. So they made use of the entire training set, because the GAN can leverage the images from other classes for learning general featues of image and then better create images of the sparse class. For creating images of a certain class, it is necessary to train the GAN conditioned on a certain label like in ACGAN. In ACGAN you output a real/fake label and a class label simultanoeusly. But if you have underrepresented classes this approach is flawed: If G produces real images of the seldom class, D is off good to discriminate it as a false image, since its seldomly real. So the class label of the seldom class and the real loss are kind of contradictory. To avoid this, they changed the loss formulation such that it is not two different losses anymore. Instead D outputs either fake or the class label, which makes it impossiple these two contradict each other. 
Additionally they use Autoencoders to further improve the training. First, they train an autoencoder, which decoder part is like G and the encoder part is partly like D on the training data. Then they can use the corresponding parts of the autoencoder to initialize G and D. This is especially useful for G, since now the latent code Z corresponds to the encoded part from the autoencoder. They use this fact to produce class conditioned latent code to feed G. They model a class in the latent code with a multivariate normal distribution. By running all class images in the encoder, they can calculate the means and variances, which then stay fixed for training. 
In the results they trained a classifier on a dataset, dropping real images of a class and augmenting it with the fake ones. They usually get better accuracies than with ACGAN and normal GAN trained on each class, but the results didnt seem that impressive. The image quality seems better then ACGAN though. 

### Image-to-Image Translation with Conditional Adversarial Networks (pix2pix) - 2018
There has been some GAN architectures for transforming one image into another one, but specialized for certain applications (like segmentation-map to image). Here the authors provide a general framework for translating one image into another, like sketch to image, segmetnation to image or map to satellite image. For this they use two losses for the GAN:
The first is the traditional GAN loss conditioned on the input image x. This means G gets images x and vector z as input and produces a "translated" image y. Then the tanslated fake or real image and the original image gets fed into D to discriminate. The second loss is the L1 Norm between the fake and the real image. It has been known that using L1/L2 loss is able to produce images which accurately match the low frequency of the image (thus not the sharp details like edges, but eg the right objects in the image). So to produce blurry outputs, you would not need a GAN. Therefore the GAN should especially enforce the production of real looking details. So they use the GAN to enforce "realness" on NxN sized image patches. For each patch they get an output of D (you can easily do this with convolution) and average across these results. Such an discriminator assumes the image as a Markov Random Field, assuming independece between pixels distance by patch diameter. This PatchGAN can be understood as a form of texture/style loss.
Furthermore they use dropout to generate randomness in G instead of z. The logic behind this is that G easily ignores z when training making the translation deterministic. *But they notice that it is still not very stochastic, which is an open question.* 
For the architecture of G they use a U-Net. This essentiale has a up and down path with skip connections. These skip connections were used for the low level information which are often shared between input and output, like edges.
Decent results can already be achieved by relatively small datasets.

### Photographic Image Synthesis with Cascaded Refinement Networks - 2017
This paper achieves slightly better results than pix2pix without the use of GANs. They just use a single feedforward network. They experimented with a lot of different settings and found three key characteristics important for creating realistic images: 
1) Global Coordination -> Global structures such as symmetrie are important. Therfore the network they designed has a low to high refinement structures. In the low dimension the global features can be cooradinated and then progressively get upscaled. 
2) High resolution images -> up to 2k
3) Memory -> The models need to have a high capacity
The architecutre is as said above a refinement structure. They downsample the segmentation mask as the first input. Then the features get upsampled and combined with a higher resolution segmentation mask as input for a next step. They do not use upconvolution as its said to build characteristic artifacts. They make use of perceptual loss using vgg19 fed with the original image as the net to match the feautures too. To be able to generate diverse results from a single semgentation map they output k images simultaneously and add a new loss: The basic idea is that they calculate the loss of all k images simultaneously and take the loss of the best performing image. *By considering only the best synthesized image,  this lossencourages  the  network  to  spread  its  bets  and  cover  thespace of images that conform to the input semantic layout.The loss is structurally akin to thek-means clustering ob-jective,  which only considers the closest centroid to eachdatapoint and thus encourages the centroids to spread andcover the dataset.* They expend this loss but didnt take time for that.
One thin I additonally found interesting: They experimented  with automatic measures for image realness (for example using pretrained seman-tic segmentation networks) and found that they can all befooled by augmenting any baseline to also optimize for thee valuated measure; the resulting images are not more real-istic but score very highly

### High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs (pix2pixHD) - 2018
They optimite the exisiting pix2pix framework by using a coars-to-fine G, a multi-scale D and a robust adversarial learning objective:
1) Coarse-ToFine:
They first train a global generator network on lower reosultion images (Witht the architecture used by Johnson et al.. Check out the architecture!). Then they combine this wit a local enhancer network for training them jointly on higher resolution images. The global networks than gets the input 2x downsampled and the features of the two networks get added element-wise before the residual blocks of the local enhancer.
2) Multi-Scale Discriminator
For discriminating hig resolution images D needs to be a deeper network or have larger kernels to have a perceptive field big enough, which potentially could lead to overfitting. They use three D, which all use the identic network. The real and syntethized images get downscaled 2/4 times for feedig it to the other D. So the coarser D have a higher perceptive field so it has a more global view. It makes also the training easier since you can add the finer D when you add the finer G.
3) Improved loss
They add a feature matching loss from varias layers of D with L1 Norm.

The authors also try to incorporate the knowledge about instance into the network, since segmentation maps only give class id but no object ids. Since you dont know how many objects will be in an image you cant easily add the objects ids as an additional input. Therefore they feed the network a boundary map, where objects in general are different.

To be able to have control over the image and change different objects, they feed an additional feature map to G. This feautre map is created by an autoencoder network. For every instance they compute an average pooling of the feature of this instance and expand this averaged value to the full object instance. 

They also experiment with perceptual loss, which improves it slighlty.

### Mask-Guided Portrait Editing with Conditional GANs - 2019
![alt text](https://github.com/LarsShaq/GANs/blob/master/images/MaskGuidedGAN.png)
The goal of this here created framework is to convert one image into another with help of the segmentation map, especially manipulate portraits in this case.
The input to the network is thus a source image, a target image and a corresponding soruce and target feature map. 
The overall structure exists of three parts:
1) Local Embedding Network
Given the single parts of a face from the segmentation map, encoders learn for every part a feature representation.
2) mask guided sub GAN
For creating the face image they need to add the feautres from 1) to the target mask. For this they find the center position of each component from the target map and for every component create a Tensor where they place the features at the center location. After that they concatenate theses Tensors.
3) Background Fusing Sub-Network
By just copying the the background to the foreground there often are noticeable artifacts at the boundary. They feed the background to an encoder to get an background feature tensor. They then combine this with the foreground to get the final picture.

The loss of some is the sum of 4 total losses:
1) Local Reconstruction loss for the encoder of the different components, which is the MSE loss.
2) Global Reconstruction
They use a training method where they feed the network with paired and unpaired images alternatively. For the paired images, the result should be the same as the input, which leads to another MSE loss for the whole image. 
3) GAN loss
They use the GAN training loss with additional feature matching loss smiliar to pix2pixHD.
4) A face parsing loss
They train a face parsing network. This is used to check if the generated images have the same mask as the target mask, which then uses pixel wide cross netropy loss. This loss is also only avaible during the paired image step.

### Toward Multimodal Image-to-Image Translation (BicycleGAN) - 2018
![alt text](https://github.com/LarsShaq/GANs/blob/master/images/BicycleGAN.png)
In this paper they expand pix2pix to be able to create more diverse picuters. 
As the authors in pix2pix noted, simply adding noise to the latent code does not produce diverse results, as the noise simply gets ignored. 
In this paper they map the latent code to the ouput but also train an encoder to come back from the ouput to the latent code. This discourages two different latent codes from generating the same output. 
They have two approaches which at the end they combine to the final result:
1) Conditional Variational Autoencoder GAN
They encode the ground truth image with an encoder. This enoced imformation gives G a noisy version of what the output should look like. They combine this encoded output with the input to G. Furthermore they wanna make sure that the encoded latent distribution is close to a standard normal distribution. The reason for this is that they can sample during inference from the latent code. If they would not enforce the latent code to follow a distirbution, the distribution at inference time without an output image will be unknown. So if they sample from the latent code, they have a lot of sparse points in the latent space, which will lead to bad outputs. The authors enforce a normal distribution by minimizing the KL divergence between them. 
2) Conditional Latent Regressor GAN
This approach is similar to InfoGAN. They use a random latent code and produce an output. Then they run the output trough an encoder to try to recover the initial latent code.
The hybride model of the two is then the BicycleGAN, which tries to compensate for the disadvantage of the single versions. The disadvantage of the cVAE-GAN is that a random latent code may not yield realistic result, due to bad converges to the gaussian or because D didnt have a chance to see sampled results during training. In the cLR-GAN the benefit of seeing ground-truth in-out pairs is not there. 
The weights of G and the encoder of cVAEGAN and cLRGAN are tied. 

The results are more diverse than Pix2Pix and look even more realisitc than pix2pix. The authors notice that the perfect size of the latent code depend on the dataset. 

### DATA AUGMENTATION GENERATIVE ADVERSARIAL NETWORKS - 2018
![alt text](https://github.com/LarsShaq/GANs/blob/master/images/DAGAN.png)
Data augmentation methods could be expanded by just learning a transformation which keeps the class distinguishable but creates a different image. In this paper they create a GAN for learning transformations of images for data augmentation. The key idea is that they feed the discriminator always also the corresponding input image. So in one iteration they feed the generated image plus the input image and in the other they feed the input iamge plus another real image from the same class. This way they make sure that G learns to produce a different image but from the same class distribution. One thing I found interesintg is that they fed the classifier for training at the end the real/fake label: *The real or fake label was also passed to the network, to enable the network to learn how best to emphasise true over generated data. This last step proved crucial to maximizing the potential of the DAGAN augmentations.*

### Towards Open-Set Identity Preserving Face Synthesis - 2018
They have some stuff important for code for mask guided portrait editign

### GAN DISSECTION: VISUALIZING AND UNDERSTANDING GENERATIVE ADVERSARIAL NETWORKS - 2018
![alt text](https://github.com/LarsShaq/GANs/blob/master/images/GANDissection.png)
When CNN became successful, there was a lot of work done of understanding and visualzing what is going on inside of them. This paper adresses the same questions, just about GANs. How are objects represented internally?
Therefore they use two methods:
1) Characterizing Units by Dissection
Here they take a feature map of a certain layer of G and upsample it. Then they threshold the value and make IOU with segmetnation masks of different classes. Then they can find for each unit (a unit here is a channel of the feature map) which class it best represents and for each class which unit it best represents. But just because teh activatio of a unit correlates with object, it doesnt mean that unit is responsible for creating that object. Any output will jointly depend on several parts of the represnetation. 
2) Measuring Causal Relationship Using INtervention
Here they look which combination of units are responsible for generating objects. Therefore they test if they turn a certain set of units on/off, if the object will disappear. They measure this by measuring the difference in the segmentation mask for this class. Since exhaustively searching over all combination of sets it not feasible they design this as an optimization problem. 
Results:
- they find units which correspond well to single objects, like a couch in an image
- many units present parts of an object, like body and head of a person
- as with CNNs different layers account for different type of information. The first layer is entangled, not corresponding to anything in specific. The middle layer match semantic objects and objects parts. The last layers match local pixel patterns
- they can improve images by finding units corresponding to artificats in an image and setting them to zero. This manual change beats state of the art GAN models. Furthermore they expand this to find automatically units: For a unit they generate 200000 images and select 10000 images that maximize the activation of unit u. And this subset is compared to the real images using the FID metrix. The ones with the worst score get ablated. Of the 20 worst choosen this way, 10 correspong to the manually choosen ones
- by finding corresponding units for an object they can delete that object frmo the scene. But interestingly if the object is critical to the concept of the scene, like a table for a conference room, it is not possible to delete that object.
- reversiley it is possible to add objects to a scene, but only where it makes sense. A door couldnt be placed in the sky for example. There is some kind of relationship between the units that forbids that

### Generating Photo-Realistic Training Data to ImproveFace Recognition Accuracy - 2018
In this paper they use syntehtically created images of faces to improve a face recognition system. Especially they designed a GAN where the latent code is divided into person realted features and person indepent features. They use progressive growing for generating good images. Additionally they use the ACGAN method to condition on the person identity. 
Instead of condition on y they extend the framework to lean an embedded representation of y which shall follow a gaussian distirbution. With this they dont have just the discrete values of y to conditon on but continous values from a distribution which lets them sample new identities as well. They make the embedded code follow a posterior by using another discriminator (Adversarial  autoencoders papers). This embedded is zId the latent code for id realted feuatres. To make sure the other part zNID doenst have person-id realted feature they use the mutual information loss.
They use the generated images to add to a small, medium and large dataset. It had significant imporvemnets to the small dataset but on the large one not as big as you could expect. Furthermore there seems to exist a good ratio between fake and real images for training. 

### The unusual Effectieness of Averaging in GAN training - 2019
They show the benefit of averaging the weights of G during training especially using exponential moving average. They give a lot of theoretical stuff I havent read yet, like thath one reason for non-convergence is cycling around an optimal solution. Paper hows promising results and easy to implement.  

### Self-Attention Generative Adversarial Networks - 2019

### The relativistic discriminator: a key element missing from standard GAN - 2019
In standard generative adversarial network (SGAN), the discriminator estimates the probability that the input data is real. The generator is trained to increase the probability that fake data is real. We argue that it should also simultaneously decrease the probability that real data is real because 1) this would account for a priori knowledge that half of the data in the mini-batch is fake, 2) this would be observed with divergence minimization, and 3) in optimal settings, SGAN would be equivalent to integral probability metric (IPM) GANs.
We show that this property can be induced by using a relativistic discriminator which estimate the probability that the given real data is more realistic than a randomly sampled fake data. We also present a variant in which the discriminator estimate the probability that the given real data is more realistic than fake data, on average.
Empirically, we observe that 1) RGANs and RaGANs are significantly more stable and generate higher quality data samples than their non-relativistic counterparts, 2) Standard RaGAN with gradient penalty generate data of better quality than WGAN-GP while only requiring a single discriminator update per generator update (reducing the time taken for reaching the state-of-the-art by 400%), and 3) RaGANs are able to generate plausible high resolutions images (256x256) from a very small sample (N=2011), while GAN and LSGAN cannot; these images are of significantly better quality than the ones generated by WGAN-GP and SGAN with spectral normalization. 

### Large Scale GAN Training for High Fidelity Natural Image Synthesis - 2019
They train conditional GAN on Image Net resulting in state of the art results. Specifically they use a lot of recent techniques to make large scale training possible and show that the GANS benefit a lot from scaling.
- baseline is SA-GAN architecture
- hinge loss GAN objective
- class information to G with class-conditional BatchNorm and to D with projection
- employ Spectral Norm in G
- 2 D steps per G step
- for evaluation use moving average of G's weights
- orthogonal initialization
- found progresive growing unnessecary even gor 512x512
- found immediately big benefits of increasing batch size (up to 8 times) -> become unstable and collapse. Therfore early stopping
- increasing capacity of model by using more channels improved a lot to
- use skipt connections from z to various layers of G
- deeper model is better, but needed to change architecutre design
- Trading Variety and Fidelity with truncation trick: During training they sampled z from a normal distribution. During evaluation they can trade off quality vs. variety by truncating the normal distribution they sample, resampling outliers into the truncated shape
They further analyse the instability of training in G and D:
- The collapse in G can be seen by looking ath the top three singular values of weight matrix. But constraining these values didnt help stability
- gradient penalty in D helps with instability but performance degrades. Furhtermore D is memorizing/overfitting towards the collapse point. They test if G memorizes training points which is not the case
In Appendix: They tested differnt latent sampling staregies. They test some strategies to avoid collapse, like training D even more often than G, learning rate, freezing G beofre collapse etc.. Analyse spikes in D's spectra which have noise which up close look like impulse response. 
### ELASTIC-INFOGAN: UNSUPERVISED DISENTANGLED REPRESENTATION LEARNING IN IMBALANCED DATA - 2019
InfoGan assumes for the categorial latent cariables uniform distribution. For a lot of artificial datasets like MNIST, this is true, there is app. an equal number of image from 0-9. But in real world dataset there is an imbalance in the different object categories. Therefore InfoGAN in not able to disentangle different objects in these dataset effectifely. This paper tackles this problem by introducing two ideas:
1) Instead of assuming an uniform categorial distirbution, the categorial distribution gets learned by the network. Therefore the laten variable ci have to be sampled in a way that is differentiable and therfore applieable for Gradien Descent. They use some Gumbel-Softmax stuff for that (didnt bother about the details).
2) They wanna make sure that the categorial variables correspond to object identeties, such as one variable corresponds to the digit in MNIST. For this they apply random transformations on the images which keep the object identety, like changing contrast or slight rotation. The estimate of the latent variable should be the same, because it should only bother about object identity and not the other stuff. (didnt bother about details). 

### MaskGAN: Towards Diverse and Interactive Facial Image Manipulation -2019
// To read
### Generative Adversarial Networks: A Survey andTaxonomy - 2019
Short survey of recent architecture and losses. Spectral NOrmalization highly recommended.

### Improved Precision and Recall Metric for AssessingGenerative Models -2019
Improved metric based on comparing images in feature space. Realness score. Could be interesing for evaluating the quality of images.
### A Style-Based Generator Architecture for Generative Adversarial Networks -2019
use different architecture for G to achieve state of the art results. Instead of feeding a noise vecotr as the inout, they learn a constant vector. This then goes through some affine transfromation and goes into different layers of G. Also noise is added to different layers of G. Didnt read the details.

### DermGAN:Synthetic Generation of Clinical Skin Images with Pathology -2019
Used synthetic images of skin as data augmentation. Based on pix2pix with nothing really new. Results were sam as base class, just for some difficutl classes better.

### Self-Supervised GANs via Auxiliary Rotation Loss - 2019
They used rotation classification as an auxilairy loss to improve unsupervised trainig. They rotated some image of G and asked D to tell teh degree of rotation. The loss of D only depeends on what it says on the real images. The authors especially mention that fortgetting of D is a big problem in training GANs because it is an online learning problem where the data D is trainied on always changes. Self-supervised training can help with that.

### Extremely Weak Supervised Image-to-Image Translation forSemantic Segmentation -2019
Basically combine pix2pix und cycle gan to be able to use less paired images to achieve good results. Use the best paired images to use doing k mean clustering in feature space. 

### Attribute-Guided  Sketch  Generation
Not that useful on first glance

Delete units to make segmentation better. 


## Some basics

### VAE 
long paper covering the basics: An Introduction to Variational Autoencoders

### Entropy
Entropy is the expected Information of an event, which is disproportional to the uncertainty of it. You can imagine that if an event is really rare, the information of that event happening is really high. Formally it is: E(I(X)) = sum[-p(x)log(p(x))] , where I(x) = -log(P(x)) and the other part is just the formular for expectation. Some intuition why the information is the negative log:
1) log(1) = 0 -> An event which always happens has no information
2) log(x) >= 0 -> Information always positive
3) -log(p) = log(1/p) -> Information monotonically decreasing with p
4) If x1, x2 independent, information should add up: I(x1,x2) = -log(P(x1,x2)) = -log(P(x1)P(x2)) = -log(P(x1)) - log(P(x2)) = I(x1) + I(x2)  

### Spectral Norm
//TODO https://de.wikipedia.org/wiki/Spektralnorm

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
Inception Score
MS-SSIM
Learned Perceptual Image Patch Similarity

