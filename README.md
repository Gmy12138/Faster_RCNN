# Faster-RCNN
### pytorch 1.0 and python 3.6 is supported
A PyTorch implementation of Faster-RCNN, with support for training, inference and evaluation.

## Introduction
The method of Faster-RCNN was used to perform defect detection on NEU surface defect database, and We adopted data enhancement methods such as flipping. Finally achieved a satisfactory result.


## Installation
##### Clone and install requirements
    $ git clone https://github.com/Gmy12138/Faster_RCNN.git
    $ cd Faster_RCNN/
    $ pip install -r requirements.txt

##### Download pretrained weights
    We used two pretrained models in our experiments, VGG and ResNet101.
    Download them and put them into the data/pretrained_model/.
  * VGG16:[VGG16](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)
  * ResNet101:[ResNet101](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)
    
##### Download NEU-DET dataset
    $ Download address    http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
    $ cd data/
    $ Put the dataset in the data folder
    
##### Compile the cuda dependencies using following simple commands:
    $ cd lib
    $ python setup.py build develop
 
     
    
## Test
Evaluates the model on NEU-DET test.


| Model        |Image Size| ROI Pooling Ways  |Data Enhancement    | mAP (min. 50 IoU) |
|:------------:|:--------:|:-----------------:|:------------------:|:-----------------:|
| Faster-RCNN  |300       |      pool         |YES                 | 66.9              |
| Faster-RCNN  |300       |      align        |YES                 | 69.9              |


## Inference
Uses pretrained weights to make predictions on images. The VGG16 measurement marked shows the inference time of this implementation on GPU 2080ti.


| Model      |Backbone      |  Image Size     | GPU      | FPS      | parameters (10<sup>6</sup>)|FLOPs (10<sup>9</sup>)|
|:----------:|:------------:|:---------------:|:--------:|:--------:|:--------------------------:|:--------------------:|
|Faster-RCNN | VGG16        |     300         | 2080ti   |          |           136.79           |     127.64           | 



