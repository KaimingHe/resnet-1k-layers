# Deep Residual Networks with 1K Layers

By [Kaiming He](http://kaiminghe.com), [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en), [Shaoqing Ren](http://home.ustc.edu.cn/~sqren/), [Jian Sun](http://research.microsoft.com/en-us/people/jiansun/).

Microsoft Research Asia (MSRA).

## Table of Contents
0. [Introduction](#introduction)
0. [Notes](#notes)
0. [Usage](#usage)



## Introduction

This repository contains re-implemented code for the paper "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027). This work enables training quality **1k-layer** neural networks in a super simple way.

**Acknowledgement**: This code is re-implemented by Xiang Ming from Xi'an Jiaotong Univeristy for the ease of release.

Related papers:

	[a]	@article{He2016,
			author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
			title = {Identity Mappings in Deep Residual Networks},
			journal = {arXiv preprint arXiv:1603.05027},
			year = {2016}
		}
	
	[b] @article{He2015,
			author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
			title = {Deep Residual Learning for Image Recognition},
			journal = {arXiv preprint arXiv:1512.03385},
			year = {2015}
		}
	

	
## Notes

0. This code is based on the implementation of Torch ResNets (https://github.com/facebook/fb.resnet.torch).

0. The experiments in the paper were conducted in Caffe, whereas this code is re-implemented in Torch. We observed similar results within reasonable statistical variations.

0. To fit the 1k-layer models into memory without modifying much code, we simply reduced the mini-batch size to 64, noting that results in the paper were obtained with a mini-batch size of 128. Less expectedly, the results with the mini-batch size of 64 are slightly better:

	mini-batch |CIFAR-10 test error (%): (median (mean+/-std))
	:---------:|:------------------:
	128 (as in [a]) | 4.92 (4.89+/-0.14)
	64 (as in this code)| **4.62** (4.69+/-0.20)

0. Curves obtained by running this code with a mini-batch size of 64 (training loss: y-axis on the left; test error: y-axis on the right):	
![resnet1k](https://cloud.githubusercontent.com/assets/11435359/14414142/68714c82-ffc0-11e5-8b1b-657fdb3d96a6.png)
	
## Usage

0. Install Torch ResNets (https://github.com/facebook/fb.resnet.torch) following instructions therein.
0. Add the file resnet-pre-act.lua from this repository to ./models.
0. To train ResNet-1001 as of the form in [a]:
```
th main.lua -netType resnet-pre-act -depth 1001 -batchSize 64 -nGPU 2 -nThreads 4 -dataset cifar10 -nEpochs 200 -shareGradInput false
```
**Note**: ``shareGradInput=true'' is not valid for this model yet.
