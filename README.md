# Pytorch-DeepResidualNet

## Description 
The purpose of this project was to implement the residual net architecture used on the CIFAR-10 dataset with Pytorch described in the paper introducing residual networks in 2015, and replicate the main results from the paper (IE: A deep network should achieve comparable or better performance than a shallow network, not worse). 

## Summary
A 20 layer ResNet and a 56 layer ResNet were implemented in Pytorch. As described in the paper, the depth of the networks was equivalent to 6n + 2, where n is the number of residual layers inside of a single residual block (of which there were three, with {16,32,64} filters respectively).

The results from the paper were replicated on the dataset, with the 56 layer ResNets validation accuracy saturating at a value of ~92.5% and the 20 layer ResNets validation accuracy saturating at approx ~91.4%. 

