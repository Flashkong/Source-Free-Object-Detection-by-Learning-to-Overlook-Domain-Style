# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by ShuaiFengLi, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths

def parse_args():
    import argparse
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=2, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=10, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=1e-4, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained source model
    parser.add_argument('--rs', dest='resume_source',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession_source', dest='checksession_source',
                        help='checksession to load source model',
                        default=1, type=int)
    parser.add_argument('--checkepoch_source', dest='checkepoch_source',
                        help='checkepoch to load source model',
                        default=1, type=int)
    parser.add_argument('--checkpoint_source', dest='checkpoint_source',
                        help='checkpoint to load source model',
                        default=0, type=int)

    # resume trained target model
    parser.add_argument('--rt', dest='resume_target',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession_target', dest='checksession_target',
                        help='checksession to load target model',
                        default=1, type=int)
    parser.add_argument('--checkepoch_target', dest='checkepoch_target',
                        help='checkepoch to load target model',
                        default=1, type=int)
    parser.add_argument('--checkpoint_target', dest='checkpoint_target',
                        help='checkpoint to load target model',
                        default=0, type=int)

    # SOAP
    parser.add_argument('--alpha', dest='alpha',
                        help='teacher model update weight',
                        default=0.999, type=float)

    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
    parser.add_argument('--log_dir', dest='log_dir',
                        help='directory to store log',
                        default="logs")

    # style
    parser.add_argument('--random_style', dest='random_style',
                        help='whether use random_style',
                        action='store_true')
    parser.add_argument('--style_add_alpha', dest='style_add_alpha',
                        help='style add alpha',
                        default=0.5, type=float)
    parser.add_argument('--encoder_path', dest='encoder_path',
                        help='encoder_path',
                        default="models/vgg16/city2foggy/vgg16_35.pth", type=str)
    parser.add_argument('--decoder_path', dest='decoder_path',
                        help='decoder_path',
                        default="models/vgg16/city2foggy/decoder_iter_1200.pth", type=str)
    parser.add_argument('--style_path', dest='style_path',
                        help='style_path',
                        default="models/vgg16/city2foggy/style.jpg", type=str)
    parser.add_argument('--fc1', dest='fc1',
                        help='fc1',
                        default="models/vgg16/city2foggy/fc1_iter_1200.pth", type=str)
    parser.add_argument('--fc2', dest='fc2',
                        help='fc2',
                        default="models/vgg16/city2foggy/fc2_iter_1200.pth", type=str)
    
    # Hyperparameter
    parser.add_argument('--thresh', dest='thresh',
                        help='thresh',
                        default=0.8, type=float)
    parser.add_argument('--gw_global_weight', dest='gw_global_weight',
                        help='gw_global_weight',
                        default=0.1, type=float)
    parser.add_argument('--gw_ins_weight', dest='gw_ins_weight',
                        help='gw_ins_weight',
                        default=0.1, type=float)
    parser.add_argument('--label_weight', dest='label_weight',
                        help='labei_weight',
                        default=1, type=float)
    parser.add_argument('--gw_add_rate', dest='gw_add_rate',
                        help='gw_add_rate',
                        default=0.5, type=float)

    parser.add_argument('--gpu_default_id', dest='gpu_default_id',
                        help='gpu_default_id',
                        default=0, type=int)
    # test
    parser.add_argument('--tm', dest='test_model',
                        help='test model with source test data, or test model with target test data',
                        default='target', type=str)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args

def test(args,load_name,gpu_id):
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
    torch.cuda.set_device(gpu_id)
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
        #print('loading our dataset...........')
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
        #print('loading our dataset...........')
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
        #print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        pass
    np.random.seed(cfg.RNG_SEED)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)
    # print('The total number of roidb is {:d}'.format(len(roidb)))

    #print("loading class_agnostic from %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.class_agnostic = checkpoint['class_agnostic']
    #print('load class_agnostic successfully! class_agnostic is ' + str(args.class_agnostic))

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        #print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    #print("loading checkpoint %s" % (load_name))
    load_dict(fasterRCNN,checkpoint['model_s'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    #print('load student model successfully!')
    del checkpoint

    device = torch.device("cuda:"+str(gpu_id))
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.to(device)
        im_info = im_info.to(device)
        num_boxes = num_boxes.to(device)
        gt_boxes = gt_boxes.to(device)

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.to(device)

    start = time.time()
    max_per_image = 100

    vis = args.vis

    save_name = load_name.split('/')[-1]
    save_name = save_name.split('.')[0]
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    #print('output dir: ' + output_dir)
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
    #print('det_file will be stored in ' + det_file)

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
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(device) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(device) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device)
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
            print('im_detect: {:d}/{:d}, detect time:{:.3f}s, nms time:{:.3f}s   \r' .format(i + 1, num_images, detect_time, nms_time))
            pass
        if vis:
            if i % 5 == 0:
                img_path = os.path.join(output_dir, 'images', 'result' + str(i) + '.jpg')
                cv2.imwrite(img_path, im2show)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    #print('Evaluating detections')
    acc = imdb.evaluate_detections(all_boxes, output_dir)
    print('tested model: ' + load_name + '.  The total test num is {:d}'.format(len(roidb)))
    end = time.time()
    #print("test time: %0.4fs" % (end - start))
    del fasterRCNN
    del all_boxes
    return acc

def train(args,trained_models,gpu_id):
    import torch
    torch.cuda.set_device(gpu_id)
    from torch.autograd import Variable
    import torch.nn as nn

    import argparse
    import os
    
    import numpy as np
    import time
    import pprint
    import pdb

    import torch.nn.functional as F
    import itertools
    from torch.utils.data.sampler import Sampler

    from roi_data_layer.roidb import combined_roidb
    from roi_data_layer.roibatchLoader import roibatchLoader

    from model.utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient
    from model.utils.logger import make_print_to_file
    from model.utils.config import cfg, cfg_from_file, cfg_from_list
    from model.faster_rcnn.vgg16_style import vgg16
    from model.faster_rcnn.resnet_style import resnet

    from utils import FocalLoss
    from utils import load_dict as load_dict
    from utils import inv_lr_scheduler
    from gromovWasserstein import gromovWasserstein

    class sampler(Sampler):
        def __init__(self, train_size, batch_size):
            self.num_data = train_size
            self.num_per_batch = int(train_size / batch_size)
            self.batch_size = batch_size
            self.range = torch.arange(0, batch_size).view(1, batch_size).long()
            self.leftover_flag = False
            if train_size % batch_size:
                self.leftover = torch.arange(
                    self.num_per_batch * batch_size, train_size).long()
                self.leftover_flag = True

        def __iter__(self):
            rand_num = torch.randperm(
                self.num_per_batch).view(-1, 1) * self.batch_size
            self.rand_num = rand_num.expand(
                self.num_per_batch, self.batch_size) + self.range

            self.rand_num_view = self.rand_num.view(-1)

            if self.leftover_flag:
                self.rand_num_view = torch.cat(
                    (self.rand_num_view, self.leftover), 0)

            return iter(self.rand_num_view)

        def __len__(self):
            return self.num_data

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    # city->foggy city
    elif args.dataset == "city2foggy":
        print('loading our dataset...........')
        args.s_imdb_name = "cityscape_2007_train_s"
        args.t_imdb_name = "cityscape_2007_train_t"
        args.s_imdbtest_name = "cityscape_2007_test_s500"
        args.t_imdbtest_name = "cityscape_2007_test_t"

        args.imdb_name = args.t_imdb_name
        args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    elif args.dataset == "city2bdd":
        print('loading our dataset...........')
        args.s_imdb_name = "bdd_cityscape_2007_train_s"
        args.t_imdb_name = "bdd_2007_train"
        
        args.s_imdbtest_name = "bdd_cityscape_2007_test_s500"
        args.t_imdbtest_name = "bdd_2007_valall"

        args.imdb_name = args.t_imdb_name
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

        args.imdb_name = args.t_imdb_name
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        # args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    # cityscape->KITTI
    elif args.dataset == "CK":
        print('loading our dataset...........')
        args.s_imdb_name = "KITTI_cityscape_2007_train_s"
        args.t_imdb_name = "KITTI_2007_trainall"
        args.s_imdbtest_name = "KITTI_cityscape_2007_test_s500"
        args.t_imdbtest_name = "KITTI_2007_train500"

        args.imdb_name = args.t_imdb_name
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        # args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    # SIM -> cityscape
    elif args.dataset == "SIM2City":
        print('loading our dataset...........')
        args.s_imdb_name = "SIM_2012_train_s"
        args.t_imdb_name = "SIM_cityscape_2007_train_s"
        args.s_imdbtest_name = ""
        args.t_imdbtest_name = "SIM_cityscape_2007_test_s500"

        args.imdb_name = args.t_imdb_name
        # args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "voc2cartoon":
        args.s_imdb_name = "cartoon_voc_2007_trainval+cartoon_voc_2012_trainval"
        args.t_imdb_name = "cartoon_2007_train_cartoon"
        args.s_imdbtest_name = "cartoon_voc_2007_test"
        args.t_imdbtest_name = "cartoon_2007_test_cartoon"
        args.imdb_name = args.t_imdb_name
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    # Pascal -> watercolor
    elif args.dataset == "voc2wc":
        args.s_imdb_name = "watercolor_voc_2007_trainval+watercolor_voc_2012_trainval"
        args.t_imdb_name = "wc_2007_train_wc"
        args.s_imdbtest_name = "watercolor_voc_2007_test"
        args.t_imdbtest_name = "wc_2007_test_wc"
        args.imdb_name = args.t_imdb_name
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    # pascal -> comic
    elif args.dataset == "voc2comic":
        args.s_imdb_name = "comic_voc_2007_trainval+comic_voc_2012_trainval"
        args.t_imdb_name = "comic_2007_train_comic"
        args.s_imdbtest_name = "comic_voc_2007_test"
        args.t_imdbtest_name = "comic_2007_test_comic"
        args.imdb_name = args.t_imdb_name
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    # pascal -> clipart
    elif args.dataset == "voc2clipart":
        args.s_imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.t_imdb_name = "clipart_2007_traintest1k"
        args.s_imdbtest_name = "voc_2007_test"
        args.t_imdbtest_name = "clipart_2007_traintest1k"
        args.imdb_name = args.t_imdb_name
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    
    main_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + "/" + 'overlook'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + '/target'):
        os.makedirs(output_dir + '/target')
        
    args.log_dir=output_dir
    make_print_to_file(args.log_dir, os.path.basename(__file__))

    print('Called with args:')
    print(args)

    print('Using config:')
    print(pprint.pformat(cfg))
    # pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    
    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    print('The total number of target roidb is {:d}'.format(len(roidb)))

    print('output dir is : %s'%(output_dir))

    sampler_batch = sampler(train_size, args.batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                             imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)
    if args.cuda:
        cfg.CUDA = True

    print('start training the target model')

    device = torch.device("cuda:"+str(gpu_id))

    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    if args.cuda:
        im_data = im_data.to(device)
        im_info = im_info.to(device)
        num_boxes = num_boxes.to(device)
        gt_boxes = gt_boxes.to(device)

    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.resume_source:
        load_name = os.path.join(main_dir,
                                 'faster_rcnn_source_{}_{}_{}.pth'
                                 .format(args.checksession_source, args.checkepoch_source, args.checkpoint_source))
        print("loading class_agnostic from %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.class_agnostic = checkpoint['class_agnostic']
        print('load class_agnostic successfully! class_agnostic is ' +
              str(args.class_agnostic))

    if args.resume_target:
        load_name = os.path.join(output_dir, 'target',
                                 'faster_rcnn_target_{}_{}_{}.pth'
                                 .format(args.checksession_target, args.checkepoch_target, args.checkpoint_target))
        print("loading class_agnostic from %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.class_agnostic = checkpoint['class_agnostic']
        print('load class_agnostic successfully! class_agnostic is ' +
              str(args.class_agnostic))

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False,
                           class_agnostic=args.class_agnostic, args=args,imdb=imdb)
        fasterRCNN_T = vgg16(imdb.classes, pretrained=False,
                             class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, args=args,pretrained=False,
                            class_agnostic=args.class_agnostic,imdb=imdb)
        fasterRCNN_T = resnet(imdb.classes, 101, pretrained=False,
                              class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    fasterRCNN_T.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    lr_faster_rcnn = lr

    params = []
    if args.net == 'vgg16':
        classifier_decay_rate = 0.1
        for key, value in dict(fasterRCNN.named_parameters()).items():
                if value.requires_grad:
                    if 'bias' in key:
                        if 'RCNN_cls_score' in key or 'RCNN_base1' in key or 'RCNN_base2' in key or 'RCNN_base3' in key:
                            params += [{'params': [value], 'lr': classifier_decay_rate*(cfg.TRAIN.DOUBLE_BIAS + 1),
                                        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                        else:
                            params += [{'params': [value], 'lr': (cfg.TRAIN.DOUBLE_BIAS + 1),
                                        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                    else:
                        if 'RCNN_cls_score' in key or 'RCNN_base1' in key or 'RCNN_base2' in key or 'RCNN_base3' in key:
                            params += [{'params': [value], 'lr': classifier_decay_rate *
                                        1, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
                        else:
                            params += [{'params': [value], 'lr': 1,
                                        'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    else:
        classifier_decay_rate = 0.1
        for key, value in dict(fasterRCNN.named_parameters()).items():
                if value.requires_grad:
                    if 'bias' in key:
                        if 'RCNN_cls_score' in key :
                            params += [{'params': [value], 'lr': classifier_decay_rate*(cfg.TRAIN.DOUBLE_BIAS + 1),
                                        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                        else:
                            params += [{'params': [value], 'lr': (cfg.TRAIN.DOUBLE_BIAS + 1),
                                        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                    else:
                        if 'RCNN_cls_score' in key :
                            params += [{'params': [value], 'lr': classifier_decay_rate *
                                        1, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
                        else:
                            params += [{'params': [value], 'lr': 1,
                                        'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    for param_t in fasterRCNN_T.parameters():
        # pdb.set_trace()
        param_t.requires_grad = False

    if args.optimizer == "adam":
        lr_faster_rcnn = lr_faster_rcnn * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    param_lr_rate = []
    for param_group in optimizer.param_groups:
        param_lr_rate.append(param_group["lr"])

    if args.cuda:
        fasterRCNN.to(device)
        fasterRCNN_T.to(device)

    iters_per_epoch = int(train_size / args.batch_size)

    if args.resume_source:
        print("loading checkpoint %s" % load_name)
        load_dict(fasterRCNN, checkpoint['model'])
        load_dict(fasterRCNN_T, checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
        del checkpoint

    if args.resume_target:
        print("loading checkpoint %s" % (load_name))
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        load_dict(fasterRCNN, checkpoint['model_s'])
        load_dict(fasterRCNN_T, checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_faster_rcnn = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
        del checkpoint

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
        fasterRCNN_T = nn.DataParallel(fasterRCNN_T)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(args.log_dir + '/target')

    total_steps = 0
    total_train_size = args.max_epochs * train_size
    init_lr = lr_faster_rcnn

    gw = gromovWasserstein(args.gw_add_rate)
    print("thresh is:"+str(args.thresh))

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        fasterRCNN_T.train()

        loss_temp_fasterRCNN = 0
        start = time.time()

        data_iter = iter(dataloader)
        for step in range(1, iters_per_epoch + 1):

            total_steps += 1

            optimizer, lr_faster_rcnn = inv_lr_scheduler(param_lr_rate, optimizer, total_steps, init_lr=init_lr,
                                                         gamma=0, power=-1, weight_decay=0.0005)
            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).zero_()
                num_boxes.resize_(data[3].size()).zero_()

            im_data2 = im_data.detach().clone()

            fasterRCNN.zero_grad()
            with torch.no_grad():
                rois_t, cls_score_t,pooled_feat_t ,base_feat_t= \
                        fasterRCNN_T(im_data, im_info, gt_boxes, num_boxes,rois_t=None, student=False, target=True)

            _,  cls_score_s, pooled_feat_s,base_feat_s = \
                    fasterRCNN(im_data2, im_info, gt_boxes, num_boxes,rois_t=rois_t, student=True, target=True)

            cons_loss = -F.log_softmax(cls_score_s, 1).mul(F.softmax((cls_score_t.detach()), 1)).sum(0).sum(0) / cls_score_s.size(0)

            thresh = args.thresh
            cls_prob_t = F.softmax(cls_score_t, 1)
            max_prob_t, labels_t = cls_prob_t.max(1)
            index = (max_prob_t>=thresh).nonzero().squeeze(1)
            try:
                label_loss = F.cross_entropy(cls_score_s[index],labels_t[index].detach())
            except:
                label_loss = torch.zeros(1).to(device)


            # gw loss
            cls_prob_s = F.softmax(cls_score_s,1)
            max_prob_s, labels_s = cls_prob_s.max(1)
            cls_prob_s_d = cls_prob_s
            cls_prob_t_d = cls_prob_t
            feat_s=torch.bmm(cls_prob_s_d.unsqueeze(2), pooled_feat_s.unsqueeze(1)).view(-1, cls_prob_s_d.size(1) * pooled_feat_s.size(1))
            feat_t=torch.bmm(cls_prob_t_d.unsqueeze(2), pooled_feat_t.unsqueeze(1)).view(-1, cls_prob_t_d.size(1) * pooled_feat_t.size(1))
            
            unit_matrix = torch.eye(len(imdb.classes)).to(device)
            unit_matrix[0][0]=0 # ignore background
            
            bk_index = (max_prob_t<thresh).nonzero().squeeze(1)
            labels_t[bk_index]=0
            labels_t_onehot = unit_matrix[labels_t]
            mask = labels_t_onehot.mm(labels_t_onehot.t())
            # compute t
            cls_score_t_norm = cls_score_t/torch.sqrt((cls_score_t**2).sum(1,keepdim=True))
            t = cls_score_t_norm.mm(cls_score_t_norm.t()).detach()
            t = mask * t
            gw_loss = gw(feat_s,feat_t,t)

            shape = base_feat_s.size()
            gw_global_loss = gw(torch.t(base_feat_s.view(shape[0]*shape[1],shape[2]*shape[3])),torch.t(base_feat_t.view(shape[0]*shape[1],shape[2]*shape[3])),0)

            cons_loss = 0.0 * cons_loss # do not add cons_loss
            gw_loss = args.gw_ins_weight * gw_loss
            gw_global_loss = args.gw_global_weight * gw_global_loss
            label_loss = args.label_weight * label_loss

            loss = cons_loss + gw_loss + gw_global_loss + label_loss
            
            loss_temp_fasterRCNN += loss.item()
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()
            for param_s, param_t in zip(fasterRCNN.parameters(), fasterRCNN_T.parameters()):
                param_t.data = args.alpha * param_t.data.detach() + (1 - args.alpha) * \
                    param_s.data.detach()

            if step % args.disp_interval == 0:

                end = time.time()
                loss_temp_fasterRCNN /= args.disp_interval

                if args.mGPUs:
                    pass
                else:
                    pass

                print("[epoch %2d][iter %4d/%4d] time cost: %f, lr_faster_rcnn: %.2e"
                      % (epoch, step, iters_per_epoch, end-start, lr_faster_rcnn))
                print("\t\t\tfastRCNN loss:%.4f " % (loss_temp_fasterRCNN))
                print("\t\t\tcons_loss: %.4f,gw_loss:%.4f,gw_global_loss:%.4f,label_loss:%.4f"%(
                    cons_loss.mean().item(),gw_loss.mean().item(),gw_global_loss.mean().item(),label_loss.mean().item()))
                print("\t\t\tcons_weight: %.2f,gw_ins_weight:%.2f,gw_global_weight:%.2f,label_weight:%.2f"
                    %(0,args.gw_ins_weight,args.gw_global_weight,args.label_weight))

                if args.use_tfboard:
                    info = {
                        'loss_fasterRCNN': loss_temp_fasterRCNN,
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp_fasterRCNN = 0
                start = time.time()

            if step % 100 == 0:
                save_name = os.path.join(output_dir, 'target',
                                         'faster_rcnn_target_{}_{}_{}.pth'.format(args.session, epoch, step))
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': fasterRCNN_T.module.state_dict() if args.mGPUs else fasterRCNN_T.state_dict(),
                    'model_s': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
                print('save temporary target model: {}'.format(save_name))
                trained_models.append(save_name)

        save_name = os.path.join(output_dir, 'target',
                                 'faster_rcnn_target_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN_T.module.state_dict() if args.mGPUs else fasterRCNN_T.state_dict(),
            'model_s': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save temporary target model: {}'.format(save_name))
        trained_models.append(save_name)
    
    models.append('end')

    if args.use_tfboard:
        logger.close()


def runTest(args,models,accs,gpu_id):
    for i in models:
        accs[i]=test(args,i,gpu_id)
    for i in range(len(models)):
        models.pop()

def splitModel(args,models,accs,gpu_id):
    import time
    import pickle
    import os
    import shutil
    while True:
        if len(models)==0:
            time.sleep(3)
            continue
        if 'end' in models:
            models.pop()
            end=True
        else:
            end=False
        temp = models[:]
        a1 = temp[::2]
        a2 = temp[1::2]
        for i in range(len(a1)):
            models.pop(0)
        for i in range(len(a2)):
            models.pop(0)
        t1 = Thread(target=runTest, args=(args,a1,accs,gpu_id))
        t1.start()
        time.sleep(30)
        t2 = Thread(target=runTest, args=(args,a2,accs,1-gpu_id))
        t2.start()
        while True:
            if len(a1)==0 and len(a2)==0:
                break
            else:
                time.sleep(2)
        if end:
            print('The test is completed, start to delete redundant models, save the three models with the highest accuracy...')
            remain=sorted(accs.items(),key = lambda x:x[1],reverse = True)[:3]
            for i in remain:
                path,temp = os.path.split(i[0])
                file_name,extension = os.path.splitext(temp)
                os.system('cp %s %s'%(i[0],os.path.join(path,"..",file_name+"_"+str(round(i[1][0], 2))+extension)))
            with open(os.path.join(path,'..','accs.pkl'), 'wb') as file:
                pickle.dump(accs, file, pickle.HIGHEST_PROTOCOL)
                file.close()
            os.system('rm -rf %s'%(path))
            output_dir= os.path.split(path)[0]
            os.system('mv %s %s'%(output_dir,os.path.join(output_dir,'..',str(round(remain[0][1][0], 2)))))
            print('Save is complete!')
            break

if __name__=='__main__':
    from threading import Thread
    args = parse_args()
    # end if 'end' is detected
    models = []
    accs = {}
    t = Thread(target=splitModel, args=(args,models,accs,1-args.gpu_default_id))
    t.start()
    train(args,models,args.gpu_default_id)