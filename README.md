# Pytorch-DeepResidualNet

## Description 
The purpose of this project was to implement residual neural networks as presented in the paper published in 2015, and replicate the main result from the paper that deeper networks should achieve a comparable or better performance then shallower networks. The implementation was done in python using pytorch. 

The analysis was carried out in a Jupyter Notebook and can be seen [here](https://github.com/13jk59/Pytorch-DeepResidualNet/blob/master/Deep%20Residual%20Net%20Pytorch.ipynb). 

## Results
A 20 layer and 56 layer resnet were implemented in pytorch, and both trained for 200 epochs. As described in the paper, the depth of the networks was equivalent to 6n + 2, where n is the number of residual layers inside of a single residual block (of which there were three, with {16,32,64} filters respectively). I followed the same data preprocessing and augmentation steps the authors described in the paper, but instead of normalizing the images to a range between -1 and 1, I normalized to a range between 0 and 1. 

After training each network on the CIFAR-10 dataset for 200 epochs, it was found that the 56 layer ResNets validation accuracy converged to a value of ~92.5% and the 20 layer ResNets validation accuracy converged to a value of ~91.4%. This indeed was very similar to the results the authors obtained, and proved that through utilizing skip connections, deeper networks could indeed perform better then shallower networks. 

The results are visualized below.

<img src="./ResNet.png" width = "700">


## References
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
