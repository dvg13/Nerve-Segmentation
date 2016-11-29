# Nerve-Segmentation

I am enclosing a handful of the files for a project to semantically segment nerves in ultrasound images.
This leaves out some of data processing steps.  
 
Here's a brief description of the files:  

model.lua:  contains a number of neural net architectures.  The idea was to make the components interchangeable.  
There are two basic models for a semantic segmentation model - the first follows a paper by Long (2014)
https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

the second hypothetically follows a paper by Noh(2015)
http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf
but due to memory the decoder network is a lot more limited than the symetrical encoder/decoder scheme in the paper.

My initial process was to produce predicted segmentations and then run a classifier.  Where the classifier determined that the nerve wasn't found, it
would black out the segmentation.  This was necessary because my segmentation network did a poor job of distinguishing the images where the nerve was not present.  
Additionally, the challenge was scored using a dice co-efficent.  Where the ground truth mask was all zeros, the dice co-efficient
was said to be 1 when the predicted mask was all zeros, and 0 if it had any non-zero values.  As a result you needed to zero out a predicted mask is you didn't believe
there was a nerve - otherwise instead of getting a small penalty for small values in the segmentation mask, you would get a disporportionate, maximum penalty.  

Additionally, I tried to include this classification procedurely directly into the models.  This file contains two approaches to doing this.  In the first, the network simply outputs a segmentation output and a classification output, with both modules being back-ends to the encoder.  This seemed to make sense as it adds a penalty to the network where the encoded image is being misclassified.  In the second, the segmentation output is multiplied by the classification output.  This more closely models the task and gives the network two paths to weakening/strengthening a misclassified mask - it can do so through the segmentation network or through the classification network.     
  
double_hourglass.lua:  This network gave me my best results.  The inspiration is a paper for pose estimation
<https://arxiv.org/pdf/1603.06937.pdf> - but memory requirements again forced me to have a much more limited version of it.  This allows for
intermediate supervision between the two hourglasses.  

train_full.lua:  This is a pretty standard training script for torch.  It takes the images from a memory-mapped file as opposed to the raw images.  I would have
moved the constants from the top of the code to command line variables if i was posting or sharing it or should definitely have anyway.    

evaluate_segmentations.py:   

This script takes a couple of approaches to classifying predicted segmentation masks.  It does thesholding based on both the maximum intensity found in the predicted mask, as well as the total intensity.  It also does a random forest based on an "embedding" from the network.  This corresponded to the layer of the network where the image is the smallest - or between the encoder and the decoder modules.  I did this in python to take advantage of the super-easy random forest, as well as not to spend as much time looking up how to do things in lua.  

Ultimately, after experiment with networks that built in the classification aspect, I used a straight segmentation network (with a limited double hourglass) that did the thresholding based on the total intensity of the predicted mask.  

image_manip.lua:  Script to provide augmentations to the images.  It's all pretty basic - nothing engineered particularly for these imaages.  The goal was to apply multiple augmentations in each augmented image to try to get a more efficient effect.  Given more time I would have tested the different classes of augmentations individually.  
