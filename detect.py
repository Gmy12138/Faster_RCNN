# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
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

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import matplotlib.patches as patches
from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from lib.model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.net_utils import save_net, load_net, vis_detections
from lib.model.utils.blob import im_list_to_blob
from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.faster_rcnn.resnet import resnet
import pdb


# try:
#     xrange          # Python 2
# except NameError:
#     xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',default='pascal_voc', type=str,help='training dataset')
    parser.add_argument('--cfg', dest='cfg_file',default='cfgs/vgg16.yml', type=str,help='optional config file')
    parser.add_argument('--net', dest='net', default='vgg16', type=str,help='vgg16, res50, res101, res152')
    parser.add_argument('--set', dest='set_cfgs',default=None,help='set config keys', nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',default="models",help='directory to load models')
    parser.add_argument('--image_dir', dest='image_dir',default="data/samples",help='directory to load images for demo')
    parser.add_argument('--output_dir', dest='output_dir', default="output_image", help='directory to load images for demo')
    parser.add_argument('--cuda', dest='cuda',default=True,action='store_true',help='whether use CUDA')
    parser.add_argument('--mGPUs', dest='mGPUs',action='store_true',help='whether use multiple GPUs')
    parser.add_argument('--cag', dest='class_agnostic',action='store_true',help='whether perform class_agnostic bbox regression')
    parser.add_argument('--parallel_type', dest='parallel_type',default=0, type=int,help='which part of model to parallel, 0: all, 1: model before roi pooling')
    parser.add_argument('--checksession', dest='checksession',default=1, type=int,help='checksession to load model')
    parser.add_argument('--checkepoch', dest='checkepoch', default=1, type=int,help='checkepoch to load network')
    parser.add_argument('--checkpoint', dest='checkpoint',default=10021, type=int,help='checkpoint to load network')
    parser.add_argument('--bs', dest='batch_size',default=1, type=int,help='batch_size')
    parser.add_argument('--vis', dest='vis',action='store_true',help='visualization mode')

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    # print('Using config:')
    print(cfg['POOLING_MODE'])
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)

    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_best.pth'.format(cfg['POOLING_MODE']))

    classes = np.asarray(['__background__',  # always index 0
                         'crazing', 'inclusion', 'patches',
                         'pitted_surface', 'rolled-in_scale', 'scratches'])

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # pdb.set_trace()

    # print("load checkpoint %s" % (load_name))

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda :
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()


    if args.cuda :
        cfg.CUDA = True

    if args.cuda :
        fasterRCNN.cuda()

    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.05
    # vis = True
    vis = False


    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)
    # print(imglist)

    print('Loaded Photo: {} images.'.format(num_images))

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 7)]
    # print(colors)

    for img in imglist:

        total_tic = time.time()

        im_file = os.path.join(args.image_dir, img)
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb -> bgr
        im = im_in[:, :, ::-1]

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()

        # pdb.set_trace()
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
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        # print('detect_time:{}'.format(detect_time))
        misc_tic = time.time()
        if vis:
            im2show = np.copy(im)
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(im2show)
        for j in range(1, len(classes)):
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
                # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]

                if vis:
                    dets = cls_dets.cpu().numpy()
                    for i in range(np.minimum(10, dets.shape[0])):

                        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
                        score = dets[i, -1]
                        if score > 0.5:
                            x1=bbox[0] if bbox[0]>0 else 0
                            y1=bbox[1] if bbox[1]>0 else 0
                            x2=bbox[2]
                            y2=bbox[3]
                            box_w = x2 - x1
                            box_h = y2 - y1

                            # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                            color = colors[j]
                            # Create a Rectangle patch
                            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                            # Add the bbox to the plot
                            ax.add_patch(bbox)
                            # Add label
                            plt.text(
                                x1,
                                y1,
                                s=classes[j],
                                color="white",
                                verticalalignment="top",
                                bbox={"color": color, "pad": 0},
                            )

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
        total_time = detect_time + nms_time
        # print('nms_time:{}'.format(nms_time))
        print('total_time:{}'.format(total_time))


        if vis :
            outdir = args.output_dir + '/' + cfg['POOLING_MODE']
            os.makedirs(outdir, exist_ok=True)
            result_path = os.path.join(outdir, img)
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())

            plt.savefig(result_path, bbox_inches="tight", pad_inches=0.0)
            plt.close()

        # else:
        #     im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
        #     cv2.imshow("frame", im2showRGB)
        #     total_toc = time.time()
        #     total_time = total_toc - total_tic
        #     frame_rate = 1 / total_time
        #     print('Frame rate:', frame_rate)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

