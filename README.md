DCGAN with Autoencoder  
  
I implement Deep Convolutional Generative Adversarial Networks(DCGAN) with Autoencoder.  
  
Sorry, My English ability is low.  
So the explanation below may be difficult to read.  
Please contact me if you have any questions.  
  
# 1.introduction  
DCGAN is presented by Radford et al[1] last year.  
The purpose of DCGAN is to train "generator" which generates a image from 100 dimension uniform random numbers.  
One of the use of DCGAN is to generate manga face.(https://github.com/mattya/chainer-DCGAN)  
By the way, it is difficult to train DCGAN.  
[1] adobpt batch normalization and adjust the parameter of ADAM optimizer to train DCGAN.  
  
In this page, I introduce a way to train DCGAN which may more stable than [1].  
Due to this, We adjust only learning rate instead of adjusting many parameters or adding batch normalization layer.  
  
# 2.DCGAN and Autoencoder  
It was difficult to train deep networks about 10 years ago.  
But Hinton et al[2] found that layer wise pretraining helps to train deep networks  
So I think that pretraining may help to train DCGAN.  
Autoencoder is one of the pretraining way.  
It compress a input image (encoding) and decompress the encoded input (decoding).  
Decoder of autoencoder is similar to the generator of DCGAN.  
I think that weights of decoder may be better initial weights of generator.  
And weights of discriminator of DCGAN can be pretrained with the fixed weights of generator.  
  
# 3.DCGAN  
The weights of generator is set to the weights of pretrained decoder.  
Then pretrain weights of discriminator of DCGAN with the fixed weights of generator.  
Finally train DCGAN normally.  

# 4.Implementation details  
There are three important points.  
First, The number of output layer of discriminator is not 1 but 4.  
"4" is the number of convolution layers.  
This helps generator to learn weights stably.  
I am inpired by DeepID2+[3].  
Second, learning rate is set smaller value than default value of TensorFlow AdamOptimizer learning_rate (0.001).  
Output may be gray or lattice image if learning rate of autoencoder or generator is high.  
Training will fail if learning rate of discriminator is high.  
Please wait 10000 steps even if the output image is not good.(batch size = 10) 
  
![img_ok_ng](https://github.com/suzuichi/DCGAN_with_Autoencoder/blob/master/img_ok_ng.jpg)  
 
Third, activation functions are tanh, relu and sigmoid.  
The activation function of last layer of autoencoder and generator is tanh.  
Tea activation funciont of the others layer of autoencoder and generator is relu.  
The activation function of output layer of discriminator is sigmoid.  
I don't use elu activation function in generator because generator may output noisy or strange image.  
  
# 5.Result  
The image below is the output of generator.  
The left one is the output of generator without pretraining.  
The right one is the output of generator with pretraining using Autoencoder.  
  
![img_result](https://github.com/suzuichi/DCGAN_with_Autoencoder/blob/master/img_result.jpg)  
  
# 6.Layout of networks  
1.Autoencoder  
![img_result](https://github.com/suzuichi/DCGAN_with_Autoencoder/blob/master/img_AE.jpg)  
  
2.generator
![img_result](https://github.com/suzuichi/DCGAN_with_Autoencoder/blob/master/img_DCGAN_AE.jpg)  
  
3.Discriminator of DCGAN  
![img_result](https://github.com/suzuichi/DCGAN_with_Autoencoder/blob/master/img_DCGAN.jpg)  
  
# 7.reference  
[1]Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).  
[2]Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "A fast learning algorithm for deep belief nets." Neural computation 18.7 (2006): 1527-1554.  
[3]Sun, Yi, Xiaogang Wang, and Xiaoou Tang. "Deeply learned face representations are sparse, selective, and robust." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.  