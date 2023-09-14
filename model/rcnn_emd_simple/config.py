import os
import sys

import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = 'D://'
add_path(os.path.join(root_dir))
add_path(os.path.join(root_dir, 'lib'))

class Crowd_human:
    #class_names = ['background', 'fbox', 'hbox', 'vbox']
    #num_classes = len(class_names)
    # root_folder = os.path.join(root_dir, 'datasets/crowdhuman')
    # image_folder = os.path.join(root_dir, 'datasets/crowdhuman/Images')
    # test_folder = os.path.join(root_dir, 'datasets/crowdhuman/images_test')
    # train_source = os.path.join(root_dir, 'datasets/crowdhuman/annotation_train.odgt')
    # test_source = os.path.join(root_dir, 'datasets/crowdhuman/test_list.txt')
    # eval_source = os.path.join(root_dir, 'datasets/crowdhuman/annotation_val.odgt')

    # root_folder = os.path.join(root_dir, '/data/CrowdHuman')
    # image_folder = os.path.join(root_dir, 'tools/outputs/Images-val')
    # test_folder = os.path.join(root_dir, 'tools/outputs/test_dump')
    # #train_source = os.path.join(root_dir, 'datasets/crowdhuman/annotation_train.odgt')
    # #test_source = os.path.join(root_dir, 'datasets/crowdhuman/test_list.txt')
    # eval_source = os.path.join('./lib/data/CrowdHuman/annotation_val.odgt')
    # training = False


    #class_names = ['background', 'person']
    class_names = ['background', 'fbox', 'hbox', 'vbox']
    num_classes = len(class_names)
    root_folder = './lib/data/CrowdHuman'
    image_folder = './lib/data/CrowdHuman/Images-train'
    #image_folder = '../tools/outputs/Images-val'
    train_source = os.path.join('./lib/data/CrowdHuman/annotation_train.odgt')
    eval_source = os.path.join('./lib/data/CrowdHuman/annotation_val.odgt')
    training = False
    test_folder = os.path.join("./tools/outputs")

class Config:
    output_dir = os.path.join("./", 'outputs')
    model_dir = os.path.join(output_dir, 'model_dump')
    eval_dir = os.path.join(output_dir, 'eval_dump')
    test_dir = os.path.join(output_dir, 'test_dump')
    #init_weights = os.path.join(root_dir, 'datasets/crowdhuman/resnet50_fbaug.pth')
    init_weights = '/home/rtx2080ti/data/model/resnet50_fbaug.pth'

    # ----------data config---------- #
    image_mean = np.array([103.530, 116.280, 123.675])
    image_std = np.array([57.375, 57.120, 58.395])
    train_image_short_size = 800
    train_image_max_size = 1400
    eval_resize = True
    eval_image_short_size = 800
    eval_image_max_size = 1400
    seed_dataprovider = 3
    train_source = Crowd_human.train_source
    #test_source = Crowd_human.test_source
    eval_source = Crowd_human.eval_source
    image_folder = Crowd_human.image_folder
    #test_folder = Crowd_human.test_folder

    class_names = Crowd_human.class_names
    num_classes = Crowd_human.num_classes
    class_names2id = dict(list(zip(class_names, list(range(num_classes)))))

    # gt_boxes_name = 'fbox'  # hbox  [1] modify

    # ----------train config---------- #
    backbone_freeze_at = 2
    rpn_channel = 256
    
    train_batch_per_gpu = 2
    momentum = 0.9
    weight_decay = 1e-4
    base_lr = 1e-3 * 1.25

    warm_iter = 800
    max_epoch = 40
    lr_decay = [20, 26]
    # nr_images_epoch = 15000
    nr_images_epoch = 30000
    log_dump_interval = 20

    # ----------test config---------- #
    test_nms = 0.5
    test_nms_method = 'set_nms'
    visulize_threshold = 0.3
    pred_cls_threshold = 0.01

    # ----------model config---------- #
    batch_filter_box_size = 0
    nr_box_dim = 5
    ignore_label = -1
    max_boxes_of_image = 500

    # ----------rois generator config---------- #
    anchor_base_size = 32
    anchor_base_scale = [1]
    # anchor_aspect_ratios = [1, 2, 3]
    anchor_aspect_ratios = [0.4, 0.6, 1.5]
    num_cell_anchors = len(anchor_aspect_ratios)
    anchor_within_border = False

    rpn_min_box_size = 2
    rpn_nms_threshold = 0.7
    train_prev_nms_top_n = 12000
    train_post_nms_top_n = 2000
    test_prev_nms_top_n = 6000
    test_post_nms_top_n = 1000

    # ----------binding&training config---------- #
    rpn_smooth_l1_beta = 1
    rcnn_smooth_l1_beta = 1

    num_sample_anchors = 256
    positive_anchor_ratio = 0.5
    rpn_positive_overlap = 0.7
    rpn_negative_overlap = 0.3
    rpn_bbox_normalize_targets = False

    num_rois = 512
    fg_ratio = 0.5
    fg_threshold = 0.5
    bg_threshold_high = 0.5
    bg_threshold_low = 0.0
    rcnn_bbox_normalize_targets = True
    bbox_normalize_means = np.array([0, 0, 0, 0])
    bbox_normalize_stds = np.array([0.1, 0.1, 0.2, 0.2])

config = Config()

