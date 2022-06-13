# coding=utf-8
# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os

from utils import load_dict

import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, drowGT
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from model.utils.logger import make_print_to_file

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--tm', dest='test_model',
                        help='test model with source test data, or test model with target test data',
                        default='target', type=str)
    parser.add_argument('--lm', dest='load_name',
                        help='path for load model',
                        default='', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    # log and diaplay
    parser.add_argument('--log_dir', dest='log_dir',
                        help='directory to store log',
                        default="logs")
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    make_print_to_file(args.log_dir,os.path.basename(__file__))

    print('Called with args:')
    print(args)

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    # city->foggy city
    elif args.dataset == "city2foggy":
        print('loading our dataset...........')
        args.s_imdb_name = "cityscape_2007_train_s500"
        args.t_imdb_name = "cityscape_2007_train_t"
        args.s_imdbtest_name = "cityscape_2007_test_s500"
        args.t_imdbtest_name = "cityscape_2007_test_t"

        if args.test_model=='source':
            args.imdbval_name=args.s_imdbtest_name
        elif args.test_model=='target':
            args.imdbval_name=args.t_imdbtest_name
        args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    elif args.dataset == "city2bdd":
        print('loading our dataset...........')
        args.s_imdb_name = "bdd_cityscape_2007_train_s"
        args.t_imdb_name = "bdd_2007_train"
        
        args.s_imdbtest_name = "bdd_cityscape_2007_test_s500"
        args.t_imdbtest_name = "bdd_2007_valall"

        if args.test_model=='source':
            args.imdbval_name=args.s_imdbtest_name
        elif args.test_model=='target':
            args.imdbval_name=args.t_imdbtest_name
        args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    
    # KITTI->cityscape
    elif args.dataset == "KC":
        print('loading our dataset...........')
        args.s_imdb_name = "KITTI_2007_trainall"
        args.t_imdb_name = "KITTI_cityscape_2007_train_s"
        args.s_imdbtest_name = "KITTI_2007_train500"
        args.t_imdbtest_name = "KITTI_cityscape_2007_test_s500"

        if args.test_model == 'source':
            args.imdbval_name = args.s_imdbtest_name
        elif args.test_model == 'target':
            args.imdbval_name = args.t_imdbtest_name
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        # args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    # cityscape->KITTI
    elif args.dataset == "CK":
        print('loading our dataset...........')
        args.s_imdb_name = "KITTI_cityscape_2007_train_s"
        args.t_imdb_name = "KITTI_2007_trainall"
        args.s_imdbtest_name = "KITTI_cityscape_2007_test_s500"
        args.t_imdbtest_name = "KITTI_2007_train500"

        if args.test_model == 'source':
            args.imdbval_name = args.s_imdbtest_name
        elif args.test_model == 'target':
            args.imdbval_name = args.t_imdbtest_name
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        # args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    # SIM -> cityscape
    elif args.dataset == "SIM2City":
        print('loading our dataset...........')
        args.s_imdb_name = "SIM_2012_train_s"
        args.t_imdb_name = "SIM_cityscape_2007_train_s"
        args.s_imdbtest_name = ""
        args.t_imdbtest_name = "SIM_cityscape_2007_test_s500"

        if args.test_model == 'source':
            args.imdbval_name = args.s_imdbtest_name
        elif args.test_model == 'target':
            args.imdbval_name = args.t_imdbtest_name
        # args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "voc2cartoon":
        args.s_imdb_name = "cartoon_voc_2007_trainval+cartoon_voc_2012_trainval"
        args.t_imdb_name = "cartoon_2007_train_cartoon"
        args.s_imdbtest_name = "cartoon_voc_2007_test"
        args.t_imdbtest_name = "cartoon_2007_test_cartoon"
        if args.test_model == 'source':
            args.imdbval_name = args.s_imdbtest_name
        elif args.test_model == 'target':
            args.imdbval_name = args.t_imdbtest_name
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    # Pascal -> watercolor
    elif args.dataset == "voc2wc":
        args.s_imdb_name = "watercolor_voc_2007_trainval+watercolor_voc_2012_trainval"
        args.t_imdb_name = "wc_2007_train_wc"
        args.s_imdbtest_name = "watercolor_voc_2007_test"
        args.t_imdbtest_name = "wc_2007_test_wc"
        if args.test_model == 'source':
            args.imdbval_name = args.s_imdbtest_name
        elif args.test_model == 'target':
            args.imdbval_name = args.t_imdbtest_name
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    # pascal -> comic
    elif args.dataset == "voc2comic":
        args.s_imdb_name = "comic_voc_2007_trainval+comic_voc_2012_trainval"
        args.t_imdb_name = "comic_2007_train_comic"
        args.s_imdbtest_name = "comic_voc_2007_test"
        args.t_imdbtest_name = "comic_2007_test_comic"
        if args.test_model == 'source':
            args.imdbval_name = args.s_imdbtest_name
        elif args.test_model == 'target':
            args.imdbval_name = args.t_imdbtest_name
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    # pascal -> clipart
    elif args.dataset == "voc2clipart":
        args.s_imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.t_imdb_name = "clipart_2007_traintest1k"
        args.s_imdbtest_name = "voc_2007_test"
        args.t_imdbtest_name = "clipart_2007_traintest1k"
        if args.test_model == 'source':
            args.imdbval_name = args.s_imdbtest_name
        elif args.test_model == 'target':
            args.imdbval_name = args.t_imdbtest_name
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    np.random.seed(cfg.RNG_SEED)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print('Using config:')
    # print(pprint.pformat(cfg))

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)
    print('The total number of roidb is {:d}'.format(len(roidb)))

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,args.load_name)
    print("loading class_agnostic from %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.class_agnostic = checkpoint['class_agnostic']
    print('load class_agnostic successfully! class_agnostic is ' + str(args.class_agnostic))

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("loading checkpoint %s" % (load_name))
    load_dict(fasterRCNN,checkpoint['model_s'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load student model successfully!')
    del checkpoint

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    save_name = load_name.split('/')[-1]
    save_name = save_name.split('.')[0]
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    print('output dir: ' + output_dir)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                             imdb.num_classes, training=False, normalize=False,mosaic=False,target=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    if vis:
        thresh = 0.05
        if not os.path.exists(os.path.join(output_dir, 'images')):
            os.makedirs(os.path.join(output_dir, 'images'))
    else:
        thresh = 0.0

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')
    print('det_file will be stored in ' + det_file)

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):
        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    if args.cuda:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
            gtboxes=gt_boxes[0].t()[:4].t()
            gtboxes/=im_info[0][2]
            im2show=drowGT(im2show, gtboxes,gt_boxes, imdb.classes)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        if i%50==0:
            print('im_detect: {:d}/{:d}, detect time:{:.3f}s, nms time:{:.3f}s   \r' \
                  .format(i + 1, num_images, detect_time, nms_time))

        if vis:
            if i % 5 == 0:
                img_path = os.path.join(output_dir, 'images', 'result' + str(i) + '.jpg')
                cv2.imwrite(img_path, im2show)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)
    print('tested model: ' + args.load_name)
    end = time.time()
    print("test time: %0.4fs" % (end - start))
