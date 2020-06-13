# Online Tensor Robust Principal Component Analysis

I produced these algorithsm that recovering data structures that have been corrupted by noise as part of my undergraduate [thesis](https://openresearch-repository.anu.edu.au/handle/1885/170630). The trpca algorithm is similar to [this](https://github.com/canyilu/Tensor-Robust-Principal-Component-Analysis-TRPCA) but in python instead of matlab and works for tensors of any dimension. The otrpca algorithm is similar to [this](http://www.merl.com/publications/docs/TR2016-004.pdf), but this one works for tensors of any dimension (so can be applied to colour images, instead of just black and white). Here's an example that was made using 15 cloud-affected satellite images:

![](/example.png?raw=true)