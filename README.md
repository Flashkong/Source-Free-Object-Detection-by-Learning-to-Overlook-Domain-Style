# Source-Free Object Detection by Learning to Overlook Domain Style (CVPR 2022 ORAL Paper)

This is the offical implementation of our CVPR 2022 work 'Source-Free Object Detection by Learning to Overlook Domain Style'. We aim to solve the source-free object detetion problem from a novel perspective of overlooking target domain style. The original paper can be found [here](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Source-Free_Object_Detection_by_Learning_To_Overlook_Domain_Style_CVPR_2022_paper.html).

Our paper has been selected for an **ORAL** presentation, the presentation video can be found [here](https://www.youtube.com/watch?v=A7vBStzBZLY).

If you find it helpful for your research, please consider citing:

```
@InProceedings{Li_2022_CVPR,
    author    = {Li, Shuaifeng and Ye, Mao and Zhu, Xiatian and Zhou, Lihua and Xiong, Lin},
    title     = {Source-Free Object Detection by Learning To Overlook Domain Style},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {8014-8023}
}
```
## Supplement
**We use the standard Faster R-CNN as the teacher and student. For PASCAL VOC to Clipart and PASCAL VOC to Watercolor, we use Resnet101 as our backbone. For Cityscapes to Foggy-Cityscapes and KITTI to Cityscape, we use VGG16 (without batchnorm) as our backbone.**

**Note that for the Foggy-Cityscapes dataset, we use the foggy level of 0.02. For Clipart dataset, we use all 1K images for both training and testing.**

## Requirements

- Python 2.7.12
- PyTorch 1.0.0
- torchvision 0.2.2

Please install other requirements by `pip install -r requirements.txt`

## Style enhancement module

First, the style enhancement module needs to be trained.

### Prepare Data

Run the following commands to copy target domain training images. Images will be stored in `Enhance/data/*`

```bash
cd Enhance
# an example for pascal voc -> clipart under my environment
python extract_data.py --file_path /home/lishuaifeng/data/clipart/VOC2007/ImageSets/Main/traintest1k.txt --images_folder /home/lishuaifeng/data/clipart/VOC2007/JPEGImages --scenario_name voc2clipart --image_suffix jpg
```
### Prepare encoder
Download the pre-trained vgg16 encoders from [here](https://drive.google.com/file/d/1d0lR7Nkt_iRjH-xY7a2Aq8nuUvm_jAuc/view?usp=sharing), where 'vgg16_ori.pth' is extracted from [rbgirshick/py-faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch#pretrained-model) and 'vgg16_cityscape.pth' is extracted from the source detector trained on Cityscapes. Put them into the folder named 'pre_trained'.
```bash
cd Enhance
mkdir pre_trained
cd pre_trained
# put the pre_trained vgg encoders here
```

### Train the style enhancement module
For Pascal VOC -> Clipart, Pascal VOC -> Watercolor and KITTI -> Cityscapes, run the following commands to train the style enhancement module. 'vgg16_ori.pth' is used as the encoder.
```bash
cd Enhance
# an example for pascal voc -> clipart
python train.py --scenario_name voc2clipart --content_dir data/voc2clipart --style_dir data/voc2clipart --vgg pre_trained/vgg16_ori.pth --save_dir models/voc2clipart
```
For Cityscapes -> Foggy-Cityscapes, run the following commands to train the style enhancement module. 'vgg16_cityscape.pth' is used as the encoder.
```
python train.py --scenario_name city2foggy --content_dir data/city2foggy --style_dir data/meanfoggy --vgg pre_trained/vgg16_cityscape.pth --save_dir models/city2foggy
```
### Download our trained models
If you don't want to train the style enhancemtn module by yourself, you can download our trained models from [here](https://drive.google.com/file/d/1klPyu_ql9tQZPKmwlSsF4Z4hbLy5q2fa/view?usp=sharing).
### Test the style enhancement module
Run the following commands to generate style enhanced images.

It should be noted that the test file is only used for debugging the style enhancement module, not for generating style enhanced images when overlooking target domain style.
```bash
cd Enhance
# an example for pascal voc -> clipart
python test.py --vgg pre_trained/vgg16_ori.pth --decoder models/voc2clipart/decoder_iter_160000.pth --fc1 models/voc2clipart/fc1_iter_160000.pth --fc2 models/voc2clipart/fc2_iter_160000.pth --content_dir data/voc2clipart --style_dir data/voc2clipart --output output/voc2clipart --alpha 1.0
# an example for cityscape -> foggy cityscape
python test.py --vgg pre_trained/vgg16_cityscape.pth --decoder models/city2foggy/decoder_iter_160000.pth --fc1 models/city2foggy/fc1_iter_160000.pth --fc2 models/city2foggy/fc2_iter_160000.pth --content_dir data/city2foggy --style_dir data/meanfoggy --output output/city2foggy --alpha 1.0
```

## Overlooking style module

### Compilation

Compile the cuda dependencies using following simple commands:

```bash 
cd Overlook/lib
python setup.py build develop
```
It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version. For more information, please refer to [rbgirshick/py-faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0#compilation).

### Prepare data
link your dataset dir to the dir named 'data':
```bash
cd Overlook
ln -s [your dataset dir] data
```
To prepare the datasets, please refer to [krumo/Detectron-DA-Faster-RCNN](https://github.com/krumo/Detectron-DA-Faster-RCNN#usage-example), [tiancity-NJU/da-faster-rcnn-PyTorch](https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch) and [VisionLearningGroup/DA_Detection](https://github.com/VisionLearningGroup/DA_Detection#data-preparation).

### Train the source detector
Run the following commands to train the source detector.
```bash
cd Overlook
# Pascal VOC -> clipart
python train_source.py --dataset voc2clipart --net res101 --cuda --bs 8
# Pascal VOC -> watercolor
python train_source.py --dataset voc2wc --net res101 --cuda --bs 8
# Cityscape -> Foggy-Cityscapes
python train_source.py --dataset city2foggy --net vgg16 --cuda --bs 8
# KITTI -> Cityscape
python train_source.py --dataset KC --net vgg16 --cuda --bs 8
```
To test the source detector's performance on the target domain, run the command:
```bash
# an example for pascal voc -> clipart
python test_source.py --dataset voc2clipart --tm target --lm [your model path] --net res101 --cuda
```

### Copy models
Copy the trained enhancement models to the 'models' folder. 
```bash
cd Overlook
mkdir models
mkdir models/enhance
# for Cityscapes -> Foggy-Cityscapes, copy the style image
cp -r ../Enhance/data/meanfoggy models/enhance
# copy the models for each scenario
cp -r ../Enhance/pre_trained models/enhance
cp -r ../Enhance/models/voc2clipart models/enhance
cp -r ../Enhance/models/voc2wc models/enhance
cp -r ../Enhance/models/city2foggy models/enhance
cp -r ../Enhance/models/KC models/enhance
```
The file tree of the 'enhance' folder is as follows:
```
./Overlook/models/enhance
├── city2foggy
│   ├── decoder_iter_160000.pth
│   ├── fc1_iter_160000.pth
│   └── fc2_iter_160000.pth
├── KC
│   ├── decoder_iter_160000.pth
│   ├── fc1_iter_160000.pth
│   └── fc2_iter_160000.pth
├── meanfoggy
│   └── meanfoggy.jpg
├── pre_trained
│   ├── vgg16_cityscape.pth
│   └── vgg16_ori.pth
├── voc2clipart
│   ├── decoder_iter_160000.pth
│   ├── fc1_iter_160000.pth
│   └── fc2_iter_160000.pth
└── voc2wc
    ├── decoder_iter_160000.pth
    ├── fc1_iter_160000.pth
    └── fc2_iter_160000.pth
```
### Train and test the overlooking style module
If you have two GPUs , run the following commands to train and test the overlooking style module. It will train on GPU:0 and test on both GPU:0 and GPU:1 simultaneously.
```bash
# an example for Pascal VOC -> Clipart
python traintest_target.py --dataset voc2clipart --net res101 --rs True --checksession_source [your source detector session] --checkepoch_source [your source detector epoch] --checkpoint_source [your source detector point] --bs 1 --cuda --epochs 3 --random_style --style_add_alpha 1.0 --encoder_path models/enhance/pre_trained/vgg16_ori.pth --decoder_path models/enhance/voc2clipart/decoder_iter_160000.pth --fc1 models/enhance/voc2clipart/fc1_iter_160000.pth --fc2 models/enhance/voc2clipart/fc2_iter_160000.pth
# an example for Cityscapes -> Foggy-cityscapes
python traintest_target.py --dataset city2foggy --net vgg16 --rs True --checksession_source [your source detector session] --checkepoch_source [your source detector epoch] --checkpoint_source [your source detector point] --bs 1 --cuda --epochs 1 --style_add_alpha 0.4 --style_path models/enhance/meanfoggy/meanfoggy.jpg --encoder_path models/enhance/pre_trained/vgg16_cityscape.pth --decoder_path models/enhance/city2foggy/decoder_iter_160000.pth --fc1 models/enhance/city2foggy/fc1_iter_160000.pth --fc2 models/vgg16/city2foggy/fc2_iter_160000.pth
```
If you only have one GPU, run the following commands to train the overlooking style module.
```bash
# an example for Pascal VOC -> Clipart
python train_target.py --dataset voc2clipart --net res101 --rs True --checksession_source [your source detector session] --checkepoch_source [your source detector epoch] --checkpoint_source [your source detector point] --bs 1 --cuda --epochs 3 --random_style --style_add_alpha 1.0 --encoder_path models/enhance/pre_trained/vgg16_ori.pth --decoder_path models/enhance/voc2clipart/decoder_iter_160000.pth --fc1 models/enhance/voc2clipart/fc1_iter_160000.pth --fc2 models/enhance/voc2clipart/fc2_iter_160000.pth
# an example for Cityscapes -> Foggy-cityscapes
python train_target.py --dataset city2foggy --net vgg16 --rs True --checksession_source [your source detector session] --checkepoch_source [your source detector epoch] --checkpoint_source [your source detector point] --bs 1 --cuda --epochs 1 --style_add_alpha 0.4 --style_path models/enhance/meanfoggy/meanfoggy.jpg --encoder_path models/enhance/pre_trained/vgg16_cityscape.pth --decoder_path models/enhance/city2foggy/decoder_iter_160000.pth --fc1 models/enhance/city2foggy/fc1_iter_160000.pth --fc2 models/vgg16/city2foggy/fc2_iter_160000.pth
```
And run the following command to test the trained models one by one.
```bash
# an example for Pascal VOC -> Clipart
python test_target.py --dataset voc2clipart --tm target --lm [your model path] --net res101 --cuda
```

## Acknowledgment

The implementation is built on the python implementation of Faster RCNN [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) and Arbitrary Style Transfer [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN).