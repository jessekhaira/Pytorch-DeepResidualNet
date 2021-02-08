# Pytorch-DeepResidualNet

## Description 
The purpose of this project was to implement the residual net architecture used on the CIFAR-10 dataset with Pytorch described in the paper introducing residual networks in 2015, and replicate the main results from the paper (IE: A deep network should achieve equivalent or better performance than a shallow network, not worse). 

The full analysis and the model implementation can be seen in the Jupyter Notebook attached. 

## Results
I implemented a 20 layer ResNet and a 56 layer ResNet in Pytorch. As described in the paper, the depth of the networks was equivalent to 6n + 2, where n is the number of residual layers inside of a single residual block (of which there were three, with {16,32,64} filters respectively). 

After training the networks on the CIFAR-10 dataset, it was found that the 56 layer ResNets validation accuracy converged to a value of ~92.5% and the 20 layer ResNets validation accuracy converged to a value of ~91.4%

The results are visualized below.

<img src="./ResNet.png" width = "700">


## References
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
