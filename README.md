# K-Sparse-AutoEncoder

## What have we done

* We have create an MNIST classification solver with mini batches and more.
* After the training step we reached 96% accuracy on the test data-set.
* The network architecture was constructed out of 3 layer:
  * 784 to 30 Neurons
  * 30 to 10 Neurons
  * 10 to 10  Neurons
* On this step we took the same network and created an Auto-Encoder
  * 784 to 100 Neurons (encode layer)
  * 100 to 784 Neurons (decode layer)
* Implemented K-Sparse logic on middle layer (encode layer)
* We visualized the weights of the encode layer for different K valu

## Sparse AutoEncoder
* k-Sparse Autoencoders finds the k highest activations in z (hidden layer) and zeros out the rest.
* The error is then backpropogated only through the k active nodes in z.
* In the following example k equals 2

![Alt text](images/Autoencoder_2.png?raw=true "Title")

## Our results for different K value

![Alt text](images/k=10.PNG?raw=true "Title")

![Alt text](images/k=25.PNG?raw=true "Title")

![Alt text](images/k=40.PNG?raw=true "Title")

![Alt text](images/k=70.PNG?raw=true "Title")

## How to use

* main_mnist.py - is the main runnable example, you can easily choose between running a simple MNIST classification or a K-Sparse AutoEncoder.
* auto_encoder_3.ipynb - this is the Jupiter example, we used it to show the K-Sparse code and graphs in an easy fashion. 
