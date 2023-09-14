import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# from config import config
# from backbone.resnet50 import ResNet50
# from backbone.fpn import FPN
# from module.rpn import RPN
# from layers.pooler import roi_pooler
# from det_oprs.bbox_opr import bbox_transform_inv_opr
# from det_oprs.fpn_roi_target import fpn_roi_target
# from det_oprs.loss_opr import emd_loss_softmax
# from det_oprs.utils import get_padded_tensor
# from det_oprs.bbox_opr import box_overlap_opr_2boxes

from lib.backbone.resnet50 import ResNet50
from lib.backbone.fpn import FPN
from lib.module.rpn import RPN
from lib.layers.pooler import roi_pooler
from lib.det_oprs.bbox_opr import bbox_transform_inv_opr
from lib.det_oprs.fpn_roi_target import fpn_roi_target
from lib.det_oprs.loss_opr import softmax_loss, smooth_l1_loss
from lib.det_oprs.utils import get_padded_tensor

from model.rcnn_emd_simple.config import config
from lib.det_oprs.bbox_opr import box_overlap_opr_2boxes
from lib.det_oprs.loss_opr import emd_loss_softmax


def torch_nanmean(x):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
    return value / num

def merge_dict(x, y):
    for k, v in x.items():
        if k in y.keys():
            y[k] += v
        # else:
        #     y[k] = v

def smooth_l1_loss_modified(pred, target):
    if pred.shape != target.shape:
        return 0
    else:
        denominator = (pred * target).sum(1)
        molecule = (pred * pred).sum(1).sqrt() * (target * target).sum(1).sqrt()
        cosineSimilarity = denominator / molecule
        difference = 1.0 - cosineSimilarity
        difference = difference.mean()
        difference = torch.where(torch.isnan(difference), torch.full_like(difference, 0), difference)
        return difference

def match_loss(batchszie, boxes1, boxes2):
    loss_box_match = dict()

    for i in range(batchszie):

        iou1 = box_overlap_opr_2boxes(boxes1[i][:, 1:], boxes2[i][:, 1:])
        iou2 = box_overlap_opr_2boxes(boxes2[i][:, 1:], boxes1[i][:, 1:])

        overlaps1, overlaps_indices1 = iou1.sort(descending=True, dim=1)
        overlaps2, overlaps_indices2 = iou2.sort(descending=True, dim=1)

        # VBox of head2 is assigned to VBox of head1
        max_overlaps_indices1 = overlaps_indices1[:, :1].flatten()
        match_boxes1 = boxes2[i][max_overlaps_indices1]


        # VBox of head1 is assigned to VBox of head2
        max_overlaps_indices2 = overlaps_indices2[:, :1].flatten()
        match_boxes2 = boxes1[i][max_overlaps_indices2]

        concat_loss1 = smooth_l1_loss_modified(boxes1[i][:, 1:], match_boxes1[:, 1:])
        concat_loss2 = smooth_l1_loss_modified(boxes2[i][:, 1:], match_boxes2[:, 1:])
        concat_loss = concat_loss1 + concat_loss2
        if i == 0:
            loss_box_match["loss_box_match"] = concat_loss
        else:
            loss_box_match["loss_box_match"] += concat_loss
    return loss_box_match

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Share
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 2, 6)
        # Independent
        self.RPN = RPN(config.rpn_channel)
        self.RCNN = RCNN()

        self.RPN2 = RPN(config.rpn_channel)
        self.RCNN2 = RCNN()

        self.RPN3 = RPN(config.rpn_channel)
        self.RCNN3 = RCNN()

        self.RPN4 = RPN(config.rpn_channel)
        self.RCNN4 = RCNN()

        self.RPN5 = RPN(config.rpn_channel)
        self.RCNN5 = RCNN()

        self.RPN6 = RPN(config.rpn_channel)
        self.RCNN6 = RCNN()

    def forward(self, image, im_info, gt_boxes=None):
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
                torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        image = get_padded_tensor(image, 64)
        if self.training:
            return self._forward_train(image, im_info, gt_boxes)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        # -----------------shujuji fenlei-------------------
        # gt with fbox and vbox
        gt_boxes_1 = list()
        # gt with hbox and vbox
        gt_boxes_2 = list()
        # Loop to get the gt of each image
        for gt in gt_boxes:
            gt_1 = gt[gt[:, 4] != 2]      #fbox+vbox
            pad_len_1 = 500 - gt_1.shape[0]
            gt_1 = torch.cat((gt_1, torch.zeros((pad_len_1, 5)).cuda().float()), dim=0)

            gt_2 = gt[gt[:, 4] != 1]      #vbox+hbox
            pad_len_2 = 500 - gt_2.shape[0]
            gt_2 = torch.cat((gt_2, torch.zeros((pad_len_2, 5)).cuda().float()), dim=0)

            gt_boxes_1.append(gt_1.unsqueeze(0))
            gt_boxes_2.append(gt_2.unsqueeze(0))

        gt_boxes_1 = torch.cat(gt_boxes_1, dim=0)     #list----->tensor
        gt_boxes_2 = torch.cat(gt_boxes_2, dim=0)
        # -------------------------------------------------------------------------------------

        # -----------------shujuji  fenlei-------------------
        # gt with fbox
        gt_boxes_3 = list()
        # gt with hbox
        gt_boxes_4 = list()
        # Loop to get the gt of each image
        for gt in gt_boxes:
            gt_3 = gt[gt[:, 4] != 2.0]
            gt_3 = gt_3[gt_3[:, 4] != 3.0]
            pad_len_3 = 500 - gt_3.shape[0]
            gt_3 = torch.cat((gt_3, torch.zeros((pad_len_3, 5)).cuda().float()), dim=0)

            gt_4 = gt[gt[:, 4] != 1.0]
            gt_4 = gt_4[gt_4[:, 4] != 3.0]
            pad_len_4 = 500 - gt_4.shape[0]
            gt_4 = torch.cat((gt_4, torch.zeros((pad_len_4, 5)).cuda().float()), dim=0)

            gt_boxes_3.append(gt_3.unsqueeze(0))
            gt_boxes_4.append(gt_4.unsqueeze(0))

        gt_boxes_3 = torch.cat(gt_boxes_3, dim=0)
        gt_boxes_4 = torch.cat(gt_boxes_4, dim=0)
        # -------------------------------------------------------------------------------------
        # # -----------------shujuji fenlei-------------------
        # gt with vbox
        gt_boxes_5 = list()
        # # gt with hbox + fbox
        #gt_boxes_6 = list()
        # # xunhuan get the gt of each image
        for gt in gt_boxes:
            gt_5 = gt[gt[:, 4] != 2.0]
            gt_5 = gt_5[gt_5[:, 4] != 1.0]
            pad_len_5 = 500 - gt_5.shape[0]
            gt_5 = torch.cat((gt_5, torch.zeros((pad_len_5, 5)).cuda().float()), dim=0)

            # gt_6 = gt[gt[:, 4] != 3.0]
            # pad_len_6 = 500 - gt_6.shape[0]
            # gt_6 = torch.cat((gt_6, torch.zeros((pad_len_6, 5)).cuda().float()), dim=0)
        #
            gt_boxes_5.append(gt_5.unsqueeze(0))
            #gt_boxes_6.append(gt_6.unsqueeze(0))


        gt_boxes_5 = torch.cat(gt_boxes_5, dim=0)
        #gt_boxes_6 = torch.cat(gt_boxes_6, dim=0)
        # -------------------------------------------------------------------------------------

        # Share
        batchszie = image.shape[0]
        fpn_fms = self.FPN(image)

        # Independent -------------------------------------------------------------------------------------
        # fbox and vbox
        loss_dict = {}
        rpn_rois, loss_dict_rpn = self.RPN(fpn_fms, im_info, gt_boxes_1)  # gt_boxes -> gt_boxes_1
        rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(rpn_rois, im_info, gt_boxes_1, top_k=2)  # gt_boxes -> gt_boxes_1
        # Separation vbox and fbox
        rcnn_rois_tmp = rcnn_rois
        rcnn_labels_tmp = rcnn_labels[:, 0]
        v_rcnn_rois = rcnn_rois_tmp[rcnn_labels_tmp == 3.0] #vbox
        f_rcnn_rois = rcnn_rois_tmp[rcnn_labels_tmp == 1.0]
        # v_box_match and f_box_match
        image_vrois_list = list()
        v_batchid = v_rcnn_rois[:, 0]
        image_frois_list = list()
        f_batchid = f_rcnn_rois[:, 0]
        for i in range(batchszie):
            image_vrois_list.append(v_rcnn_rois[v_batchid == float(i), :])
            image_frois_list.append(f_rcnn_rois[f_batchid == float(i), :])

        loss_dict_rcnn = self.RCNN(fpn_fms, rcnn_rois, rcnn_labels, rcnn_bbox_targets)

        # Independent -------------------------------------------------------------------------------------
        # hbox and vbox
        loss_dict2 = {}
        rpn_rois2, loss_dict_rpn2 = self.RPN2(fpn_fms, im_info, gt_boxes_2)  # gt_boxes -> gt_boxes_2
        rcnn_rois2, rcnn_labels2, rcnn_bbox_targets2 = fpn_roi_target(rpn_rois2, im_info, gt_boxes_2, top_k=2)  # gt_boxes -> gt_boxes_2
        # Separation of rois of vbox from fbox and vbox
        rcnn_rois_tmp2 = rcnn_rois2
        rcnn_labels_tmp2 = rcnn_labels2[:, 0]
        v_rcnn_rois2 = rcnn_rois_tmp2[rcnn_labels_tmp2 == 3.0]
        h_rcnn_rois2 = rcnn_rois_tmp2[rcnn_labels_tmp2 == 2.0]
        # v_box_match and h_box_match
        image_vrois_list2 = list()
        image_hrois_list2 = list()
        v_batchid2 = v_rcnn_rois2[:, 0]
        h_batchid2 = h_rcnn_rois2[:, 0]
        for i in range(batchszie):
            image_vrois_list2.append(v_rcnn_rois2[v_batchid2 == float(i), :])
            image_hrois_list2.append(h_rcnn_rois2[h_batchid2 == float(i), :])

        loss_dict_rcnn2 = self.RCNN2(fpn_fms, rcnn_rois2, rcnn_labels2, rcnn_bbox_targets2)

        # Independent -------------------------------------------------------------------------------------
        # fbox
        loss_dict3 = {}
        rpn_rois3, loss_dict_rpn3 = self.RPN3(fpn_fms, im_info, gt_boxes_3)  # gt_boxes -> gt_boxes_3
        rcnn_rois3, rcnn_labels3, rcnn_bbox_targets3 = fpn_roi_target(rpn_rois3, im_info, gt_boxes_3,
                                                                      top_k=2)  # gt_boxes -> gt_boxes_3
        rcnn_rois_tmp3 = rcnn_rois3
        rcnn_labels_tmp3 = rcnn_labels3[:, 0]
        f_rcnn_rois3 = rcnn_rois_tmp3[rcnn_labels_tmp3 == 1.0]
        # f_box_match
        image_frois_list3 = list()
        batchid3 = f_rcnn_rois3[:, 0]
        for i in range(batchszie):
            image_frois_list3.append(f_rcnn_rois3[batchid3 == float(i), :])

        loss_dict_rcnn3 = self.RCNN3(fpn_fms, rcnn_rois3, rcnn_labels3, rcnn_bbox_targets3)

        # Independent -------------------------------------------------------------------------------------
        # hbox
        loss_dict4 = {}
        rpn_rois4, loss_dict_rpn4 = self.RPN4(fpn_fms, im_info, gt_boxes_4)  # gt_boxes -> gt_boxes_4
        rcnn_rois4, rcnn_labels4, rcnn_bbox_targets4 = fpn_roi_target(rpn_rois4, im_info, gt_boxes_4,
                                                                      top_k=2)  # gt_boxes -> gt_boxes_4
        rcnn_rois_tmp4 = rcnn_rois4
        rcnn_labels_tmp4 = rcnn_labels4[:, 0]
        h_rcnn_rois4 = rcnn_rois_tmp4[rcnn_labels_tmp4 == 2.0]
        # h_box_match
        image_hrois_list4 = list()
        batchid4 = h_rcnn_rois4[:, 0]
        for i in range(batchszie):
            image_hrois_list4.append(h_rcnn_rois4[batchid4 == float(i), :])

        loss_dict_rcnn4 = self.RCNN4(fpn_fms, rcnn_rois4, rcnn_labels4, rcnn_bbox_targets4)
        #-------------------------------------------------------------------------------------------------------

        # Independent -------------------------------------------------------------------------------------
        # gt_5: vbox
        loss_dict5 = {}
        rpn_rois5, loss_dict_rpn5 = self.RPN5(fpn_fms, im_info, gt_boxes_5)
        rcnn_rois5, rcnn_labels5, rcnn_bbox_targets5 = fpn_roi_target(rpn_rois5, im_info, gt_boxes_5,
                                                                      top_k=2)
        rcnn_rois_tmp5 = rcnn_rois5
        rcnn_labels_tmp5 = rcnn_labels5[:, 0]
        v_rcnn_rois5 = rcnn_rois_tmp5[rcnn_labels_tmp5 == 3.0]
        # v_box_match
        image_vrois_list5 = list()
        batchid5 = v_rcnn_rois5[:, 0]
        for i in range(batchszie):
            image_vrois_list5.append(v_rcnn_rois5[batchid5 == float(i), :])

        loss_dict_rcnn5 = self.RCNN5(fpn_fms, rcnn_rois5, rcnn_labels5, rcnn_bbox_targets5)
        # -------------------------------------------------------------------------------------------------------
        #
        # Independent -------------------------------------------------------------------------------------
        # # gt_6: fbox + hbox
        # loss_dict6 = {}
        # rpn_rois6, loss_dict_rpn6 = self.RPN6(fpn_fms, im_info, gt_boxes_6)  # gt_boxes -> gt_boxes_6
        # rcnn_rois6, rcnn_labels6, rcnn_bbox_targets6 = fpn_roi_target(rpn_rois6, im_info, gt_boxes_6,
        #                                                               top_k=2)  # gt_boxes -> gt_boxes_6
        # rcnn_rois_tmp6 = rcnn_rois6
        # rcnn_labels_tmp6 = rcnn_labels6[:, 0]
        # f_rcnn_rois6 = rcnn_rois_tmp6[rcnn_labels_tmp6 == 1.0]
        # h_rcnn_rois6 = rcnn_rois_tmp6[rcnn_labels_tmp6 == 2.0]
        # # v_box_match
        # image_frois_list6 = list()
        # image_hrois_list6 = list()
        # f_batchid6 = f_rcnn_rois6[:, 0]
        # h_batchid6 = h_rcnn_rois6[:, 0]
        # for i in range(batchszie):
        #     image_frois_list6.append(f_rcnn_rois6[f_batchid6 == float(i), :])
        #     image_hrois_list6.append(h_rcnn_rois6[h_batchid6 == float(i), :])
        #
        # loss_dict_rcnn6 = self.RCNN6(fpn_fms, rcnn_rois6, rcnn_labels6, rcnn_bbox_targets6)
        # -------------------------------------------------------------------------------------------------------

        # update loss
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)

        loss_dict2.update(loss_dict_rpn2)
        loss_dict2.update(loss_dict_rcnn2)

        loss_dict3.update(loss_dict_rpn3)
        loss_dict3.update(loss_dict_rcnn3)

        loss_dict4.update(loss_dict_rpn4)
        loss_dict4.update(loss_dict_rcnn4)


        loss_dict5.update(loss_dict_rpn5)
        loss_dict5.update(loss_dict_rcnn5)
        #
        # loss_dict6.update(loss_dict_rpn6)
        # loss_dict6.update(loss_dict_rcnn6)

        # merge loss
        merge_dict(loss_dict2, loss_dict)
        merge_dict(loss_dict3, loss_dict)
        merge_dict(loss_dict4, loss_dict)
        merge_dict(loss_dict5, loss_dict)
        #merge_dict(loss_dict6, loss_dict)

        # add match_loss
        loss_box_match = match_loss(batchszie, image_vrois_list, image_vrois_list2)
        loss_fbox_match1 = match_loss(batchszie, image_frois_list3, image_frois_list)
        merge_dict(loss_fbox_match1, loss_box_match)
        # loss_fbox_match2 = match_loss(batchszie, image_frois_list3, image_frois_list6)
        # merge_dict(loss_fbox_match2, loss_box_match)
        # loss_hbox_match1 = match_loss(batchszie, image_hrois_list4, image_hrois_list6)
        # merge_dict(loss_hbox_match1, loss_box_match)
        loss_hbox_match2 = match_loss(batchszie, image_hrois_list2, image_hrois_list4)
        merge_dict(loss_hbox_match2, loss_box_match)
        # loss_vbox_match1 = match_loss(batchszie, image_vrois_list2, image_vrois_list5)
        # merge_dict(loss_vbox_match1, loss_box_match)
        loss_vbox_match2 = match_loss(batchszie, image_vrois_list, image_vrois_list5)
        merge_dict(loss_vbox_match2, loss_box_match)

        loss_box_match['loss_box_match'] = loss_box_match['loss_box_match'] * batchszie * 4

        loss_dict.update(loss_box_match)
        return loss_dict

    def _forward_test(self, image, im_info):
        # Only fbox and vbox output is used in the test
        fpn_fms = self.FPN(image)
        rpn_rois = self.RPN3(fpn_fms, im_info)
        pred_bbox = self.RCNN3(fpn_fms, rpn_rois)
        return pred_bbox.cpu().detach()

class RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)


        self.emd_pred_cls_0 = nn.Linear(1024, config.num_classes)
        self.emd_pred_delta_0 = nn.Linear(1024, config.num_classes * 4)
        self.emd_pred_cls_1 = nn.Linear(1024, config.num_classes)
        self.emd_pred_delta_1 = nn.Linear(1024, config.num_classes * 4)
        for l in [self.emd_pred_cls_0, self.emd_pred_cls_1]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        for l in [self.emd_pred_delta_0, self.emd_pred_delta_1]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None):

        fpn_fms = fpn_fms[1:][::-1]
        stride = [4, 8, 16, 32]

        pool_features = roi_pooler(fpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")
        flatten_feature = torch.flatten(pool_features, start_dim=1)
        flatten_feature = F.relu_(self.fc1(flatten_feature))
        flatten_feature = F.relu_(self.fc2(flatten_feature))

        pred_emd_cls_0 = self.emd_pred_cls_0(flatten_feature)
        pred_emd_delta_0 = self.emd_pred_delta_0(flatten_feature)
        pred_emd_cls_1 = self.emd_pred_cls_1(flatten_feature)
        pred_emd_delta_1 = self.emd_pred_delta_1(flatten_feature)
        if self.training:
            loss0 = emd_loss_softmax(
                        pred_emd_delta_0, pred_emd_cls_0,
                        pred_emd_delta_1, pred_emd_cls_1,
                        bbox_targets, labels)
            loss1 = emd_loss_softmax(
                        pred_emd_delta_1, pred_emd_cls_1,
                        pred_emd_delta_0, pred_emd_cls_0,
                        bbox_targets, labels)
            loss = torch.cat([loss0, loss1], axis=1)

            _, min_indices = loss.min(axis=1)
            loss_emd = loss[torch.arange(loss.shape[0]), min_indices]
            loss_emd = loss_emd.mean()
            loss_dict = {}
            loss_dict['loss_rcnn_emd'] = loss_emd
            return loss_dict
        else:
            class_num = pred_emd_cls_0.shape[-1] - 1
            tag = torch.arange(class_num).type_as(pred_emd_cls_0)+1
            tag = tag.repeat(pred_emd_cls_0.shape[0], 1).reshape(-1,1)
            pred_scores_0 = F.softmax(pred_emd_cls_0, dim=-1)[:, 1:].reshape(-1, 1)
            pred_scores_1 = F.softmax(pred_emd_cls_1, dim=-1)[:, 1:].reshape(-1, 1)
            pred_delta_0 = pred_emd_delta_0[:, 4:].reshape(-1, 4)
            pred_delta_1 = pred_emd_delta_1[:, 4:].reshape(-1, 4)
            base_rois = rcnn_rois[:, 1:5].repeat(1, class_num).reshape(-1, 4)
            pred_bbox_0 = restore_bbox(base_rois, pred_delta_0, True)
            pred_bbox_1 = restore_bbox(base_rois, pred_delta_1, True)
            pred_bbox_0 = torch.cat([pred_bbox_0, pred_scores_0, tag], axis=1)
            pred_bbox_1 = torch.cat([pred_bbox_1, pred_scores_1, tag], axis=1)
            pred_bbox = torch.cat((pred_bbox_0, pred_bbox_1), axis=1)
            return pred_bbox

def restore_bbox(rois, deltas, unnormalize=True):
    if unnormalize:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox


