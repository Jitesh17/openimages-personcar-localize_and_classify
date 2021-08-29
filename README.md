# OpenImages Person-Car: localise and classify

## Model name
1. Faster RCNN (with backbone as ResNet of depth 50)
2. [Cascade RCNN](https://arxiv.org/pdf/1906.09756.pdf) : A variant of Faster RCNN

## Model description
1. **Faster RCNN** is one of the most widely used architecture because of its superior accuracy and ability to detect smaller-sized objects. Its architecture starts with the input image inserted into a CNN backbone such as VGG16 and outputs a feature map. RPN(Region Proposal Network) uses these feature maps, which outputs possible regions containing an object(Objectness classification and Bounding Box regressor) using 9 anchors on a resized input image. Different size region proposals are resized to the same size by ROI pooling layer and then sent to fully connected layer for Class classification and locating the bounding box with different heads.
2. **Cascade RCNN** is a multi-stage extension of the Faster R-CNN architecture, as shown in the figure below. It is the same as Faster R-CNN architecture till the first stage, and then the bounding box head of the first stage is used as region-of-interest for the second stage, and the same follows for the third stage. The accuracy increases with the stages. The increase in accuracy is also demonstrated in the result below.
![](https://i.imgur.com/UXttTPX.png)


## Links to dataset and framework
* Dataset: https://evp-ml-data.s3.us-east-2.amazonaws.com/ml-interview/openimages-personcar/trainval.tar.gz
* Framework:
    1. [Pytorch](https://pytorch.org/)
    2. [Detectron2](https://github.com/facebookresearch/detectron2)

## Primary Analysis
* The dataset has 2239 images with 10800 instances of "person" and 5972 instances of "car" making a total of 16772 instances.
* The annotation is in coco format with bounding boxes and category-id for each instance.
* Checked annotations on few images 

## Assumptions
* I divided the dataset into training and validation datasets with a ratio of 95:5, which is (**2127 images**, 15838 instances) for training data and (**112 images**, 934 instances) for validation data.
* I will use pre-trained weights on the coco dataset of 81 classes and perform transfer learning for faster and better results.
* For augmentation, I will use random horizonta flip with probability of 0.5 and random brightness with intencity range of (0.9, 1.1).

##  Training Graphs

| Legends| | 
| -------- | -------- |
| ![](https://i.imgur.com/TPxgEvs.png)     |![](https://i.imgur.com/vpqhRF1.png)|
| ![](https://i.imgur.com/NNTHBRy.png) | ![](https://i.imgur.com/pcM0nMD.png) | 
| ![](https://i.imgur.com/IXlz4xN.png)| ![](https://i.imgur.com/LXixlL6.png) | 
| ![](https://i.imgur.com/zHo8sNL.png) | ![](https://i.imgur.com/USI5qe7.png)| 
| ![](https://i.imgur.com/RbmYz93.png)| ![](https://i.imgur.com/gX3ZOVi.png) | 


## Inference

The Faster RCNN model's inference on the first four images from the validation dataset(112 images) is below. The green bounding box is for predicted "car", a purple bounding box is for predicted "person" and the red bounding box is for ground truths.

| image_000000004.jpg                  | image_000000013.jpg                  |
| ------------------------------------ | ------------------------------------ |
| ![](https://i.imgur.com/Kx3XBE2.jpg) | ![](https://i.imgur.com/i58SrfJ.jpg) |

| image_000000062.jpg                  | image_000000100.jpg                  |
| ------------------------------------ | ------------------------------------ |
| ![](https://i.imgur.com/hQVOiq2.jpg) | ![](https://i.imgur.com/xerbZGN.jpg) |


## Evaluation

Evaluation at 10,000 iteration on 112 images.

| Model names           | AP        | AP50      | AP-car    | AP-person | AR-car    | AR-person |
| --------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Faster RCNN           | 40.29     | 71        | **56.52** | 34.06     | 56.86     | 46.95     |
| Faster RCNN with Aug  | 39.35     | 70.18     | 45.97     | 32.72     | 56.22     | 45.32     |
| Cascade RCNN          | 42        | 70.84     | 49        | 35        | 59        | 49.3      |
| Cascade RCNN with Aug | **42.65** | **70.88** | 49.39     | **35.91** | **59.39** | **49.3**  |

## Conclusion
1. **Cascade RCNN with Augmentaion** performed the best among the 4 models with average precicion of 42.65.
2. The augmentations used currently are not good enough.

## Recommendations
1. Augmentation didn't affect the result significantly therefore better augmentations should be choosen.
2. Hyperparameter tuning(i.e. training with different learining rates to observe variation in the metrics) should be performed to increase the accuracy. 
3. Faster RCNN with a deeper backbone CNN, i.e., ResNet of depth 101 (currently used ResNet of depth is 50), would take more time to train but improve accuracy.
