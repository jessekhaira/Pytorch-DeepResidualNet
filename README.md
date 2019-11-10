# Pytorch-DeepResidualNet

## Description 
The purpose of this project was primarily to learn and practice using Pytorch. The dataset used in this project was CIFAR-10. Although the objective of this project wasn't to match the state of the art results on this dataset, respectable results were still obtained in approximately ~30 minutes of training on Kaggles GPU (~90% train accuracy/~80% validation accuracy). 

## Model Summary
A standard ResNet model was implemented, as summarized in the image below. 

![alt text](https://github.com/13jk59/Pytorch-DeepResidualNet/blob/master/modelSummary.png)


## Notes 
One of the key things I wanted to incorporate into my training procedure was being able to visualize the gradient flow through the layers of my neural network. A sample gradient flow from one of the mini-batches is shown in the image below.

![alt text](https://github.com/13jk59/Pytorch-DeepResidualNet/blob/master/gradientFlow.png)

I also wanted to incorporate some simple unit tests to ensure my model was training as it should. One test that worked well was after a certain number of mini-batches, I would ensure that my parameter values had updated after performing an optimization step.

Both of these methods allowed me to debug my network much easier and streamlined the training procedure. 

