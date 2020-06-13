# Online Tensor Robust Principal Component Analysis

I produced these algorithms that recover low rank data structures which have been corrupted by sparse noise as part of my undergraduate [thesis](https://openresearch-repository.anu.edu.au/handle/1885/170630). The trpca algorithm is similar to [this](https://github.com/canyilu/Tensor-Robust-Principal-Component-Analysis-TRPCA) but in python instead of matlab and works for tensors of any dimension. The otrpca algorithm is similar to [this](http://www.merl.com/publications/docs/TR2016-004.pdf), but works for tensors of any dimension (so can be applied to colour images, instead of just black and white). In the example below, the images in the left column are the original images, in the middle column are the original images with the clouds (and a bit extra) removed using otrpca, and in the right column are what was removed. 

![](/example.png?raw=true)

This was done using 18 satellite images of the same area that were all cloud-affected.