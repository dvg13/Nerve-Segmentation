# Nerve-Segmentation

I am enclosing a handful of the files from my efforts to segment nerves for the Kaggle Ultrasound Nerve Segmentation
challenge <https://www.kaggle.com/c/ultrasound-nerve-segmentation>

This leaves out the code that handeled the data processing.
 
Here's a brief description of the files / approach:  

## model.lua:  
contains a number of neural net architectures.  The idea was to make the components interchangeable.  
There are two basic models for a semantic segmentation model:
1) Long (2014) <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>
2) Noh (2015) <http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf>

Note that due to memory the decoder network is a lot more limited than the symetrical encoder/decoder scheme in the paper.

My initial process was to produce predicted segmentations and then run a classifier over the outputs.  Where the classifier determined that the nerve wasn't found, it would black out the segmentation.  This was necessary because my segmentation network did a poor job of distinguishing the images where the nerve was not present.  Additionally, the challenge was scored using the dice co-efficent.  Where the ground truth mask is all zeros, the dice score is undefined, and for the challenge this was dealt with by giving a loss of 1 when there were predicted pixels and 0 when there were none.  As this metric non-symmetrically punishes false positives, you can improve your score by reducing these.  

I tried to include this classification procedure directly into the models.  This file contains two approaches to doing this.  In the first, the network simply outputs a segmentation output and a classification output, with both modules being back-ends to the encoder.  This adds a penalty to the network where the encoded image is being misclassified.  In the second, the segmentation output is multiplied by the classification output.  This more closely models the task and gives the network two paths to weakening/strengthening a misclassified mask - it can do so through the segmentation network or through the classification network.     
  
## double_hourglass.lua:  
This network gave me my best results.  The inspiration is a paper for pose estimation:
<https://arxiv.org/pdf/1603.06937.pdf>.
Memory requirements again forced me to implement this in a more limited form.  The code allows for
intermediate supervision between the two hourglasses, though i got my best results without this.  

## train_full.lua:  
This is a pretty standard training script for torch.  It takes the images from a memory-mapped file as opposed to the raw images.  I would have moved the constants from the top of the code to command line arguments if i was posting or sharing it.

## evaluate_segmentations.py:   
This script takes a couple of approaches to classifying predicted segmentation masks.  It does thesholding based on both the maximum intensity found in the predicted mask, as well as the total intensity.  It also employs a random forest that takes an "embedding" from the network as input.  This embedding corresponded to the bottleneck layer of the network - between the encoder and the decoder in the network. 

Ultimately, after experiment with networks that built in the classification aspect, I used a straight segmentation network (with a limited double hourglass) that did the thresholding based on the total intensity of the predicted mask.  

## image_manip.lua:  
Script to provide augmentations to the images.  It's all pretty basic - nothing engineered particularly for these imaages.  The goal was to apply multiple augmentations in each augmented image to try to get a more efficient effect.  Given more time I would have tested the different classes of augmentations individually.  
