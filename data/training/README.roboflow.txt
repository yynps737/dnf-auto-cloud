
DNF - v1 2023-06-06 3:57pm
==============================

This dataset was exported via roboflow.com on June 6, 2023 at 8:01 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 649 images.
Player-gold-monster-boss are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 40 percent of the image
* Random brigthness adjustment of between -30 and +30 percent
* Random exposure adjustment of between -20 and +20 percent
* Random Gaussian blur of between 0 and 1 pixels

The following transformations were applied to the bounding boxes of each image:
* Salt and pepper noise was applied to 5 percent of pixels


