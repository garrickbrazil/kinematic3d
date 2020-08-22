"""
This file is meant to contain functions which are
specific to region proposal networks.
"""

import matplotlib.pyplot as plt
import subprocess
import torch
import math
import re
import gc

from lib.util import *
from lib.core import *
from lib.math_3d import *
from lib.augmentations import *
from lib.nms.gpu_nms import gpu_nms
import torch.nn.functional as F

from copy import deepcopy


def generate_anchors(conf, imdb, cache_folder):
    """
    Generates the anchors according to the configuration and
    (optionally) based on the imdb properties.
    """

    decomp_alpha = 'decomp_alpha' in conf and conf.decomp_alpha
    has_vel = 'has_vel' in conf and conf.has_vel

    # use cache?
    if (cache_folder is not None) and os.path.exists(os.path.join(cache_folder, 'anchors.pkl')):

        anchors = pickle_read(os.path.join(cache_folder, 'anchors.pkl'))

    # generate anchors
    else:

        anchors = np.zeros([len(conf.anchor_scales)*len(conf.anchor_ratios), 4], dtype=np.float32)

        aind = 0

        # compute simple anchors based on scale/ratios
        for scale in conf.anchor_scales:

            for ratio in conf.anchor_ratios:

                h = scale
                w = scale*ratio

                anchors[aind, 0:4] = anchor_center(w, h, conf.feat_stride)
                aind += 1

        # has 3d? then need to compute stats for each new dimension
        # presuming that anchors are initialized in "2d"
        if conf.has_3d:

            # compute the default stats for each anchor
            normalized_gts = None

            # check all images
            for imind, imobj in enumerate(imdb):

                # has ground truths?
                if len(imobj.gts) > 0:

                    scale = imobj.scale * conf.test_scale / imobj.imH

                    # determine ignores
                    igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls, conf.min_gt_vis,
                                                   conf.min_gt_h, np.inf, scale)

                    # accumulate boxes
                    gts_all = bbXYWH2Coords(np.array([gt.bbox_full * scale for gt in imobj.gts]))
                    gts_val = gts_all[(rmvs == False) & (igns == False), :]

                    gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                    gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

                    if gts_val.shape[0] > 0:

                        # center all 2D ground truths
                        for gtind in range(0, gts_val.shape[0]):
                            w = gts_val[gtind, 2] - gts_val[gtind, 0] + 1
                            h = gts_val[gtind, 3] - gts_val[gtind, 1] + 1

                            gts_val[gtind, 0:4] = anchor_center(w, h, conf.feat_stride)

                    if gts_val.shape[0] > 0:
                        gt_info = np.ones([gts_val.shape[0], 100])*-1
                        gt_info[:, :(gts_val.shape[1] + gts_3d.shape[1])] = np.concatenate((gts_val, gts_3d), axis=1)
                        normalized_gts = gt_info if normalized_gts is None else np.vstack((normalized_gts, gt_info))

            # expand dimensions
            anchors = np.concatenate((anchors, np.zeros([anchors.shape[0], 5])), axis=1)

            if decomp_alpha:
                anchors = np.concatenate((anchors, np.zeros([anchors.shape[0], 2])), axis=1)

            if has_vel:
                anchors = np.concatenate((anchors, np.zeros([anchors.shape[0], 1])), axis=1)

            # bbox_3d order --> [cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY]
            anchors_x3d = [[] for x in range(anchors.shape[0])]
            anchors_y3d = [[] for x in range(anchors.shape[0])]
            anchors_z3d = [[] for x in range(anchors.shape[0])]
            anchors_w3d = [[] for x in range(anchors.shape[0])]
            anchors_h3d = [[] for x in range(anchors.shape[0])]
            anchors_l3d = [[] for x in range(anchors.shape[0])]
            anchors_rotY = [[] for x in range(anchors.shape[0])]
            anchors_elv = [[] for x in range(anchors.shape[0])]
            anchors_sin = [[] for x in range(anchors.shape[0])]
            anchors_cos = [[] for x in range(anchors.shape[0])]
            anchors_vel = [[] for x in range(anchors.shape[0])]

            # find best matches for each ground truth
            ols = iou(anchors[:, 0:4], normalized_gts[:, 0:4])
            gt_target_ols = np.amax(ols, axis=0)
            gt_target_anchor = np.argmax(ols, axis=0)

            # assign each box to an anchor
            for gtind, gt in enumerate(normalized_gts):

                anum = gt_target_anchor[gtind]

                if gt_target_ols[gtind] > 0.2:
                    anchors_x3d[anum].append(gt[4])
                    anchors_y3d[anum].append(gt[5])
                    anchors_z3d[anum].append(gt[6])
                    anchors_w3d[anum].append(gt[7])
                    anchors_h3d[anum].append(gt[8])
                    anchors_l3d[anum].append(gt[9])
                    anchors_rotY[anum].append(gt[10])
                    anchors_elv[anum].append(gt[15])
                    anchors_sin[anum].append(gt[16])
                    anchors_cos[anum].append(gt[17])

                    if gt[20] >= 0:
                        anchors_vel[anum].append(gt[20])

            # compute global means
            anchors_x3d_gl = np.empty(0)
            anchors_y3d_gl = np.empty(0)
            anchors_z3d_gl = np.empty(0)
            anchors_w3d_gl = np.empty(0)
            anchors_h3d_gl = np.empty(0)
            anchors_l3d_gl = np.empty(0)
            anchors_rotY_gl = np.empty(0)
            anchors_elv_gl = np.empty(0)
            anchors_sin_gl = np.empty(0)
            anchors_cos_gl = np.empty(0)
            anchors_vel_gl = np.empty(0)

            # update anchors
            for aind in range(0, anchors.shape[0]):

                if len(np.array(anchors_z3d[aind])) > 0:

                    if conf.has_3d:

                        anchors_x3d_gl = np.hstack((anchors_x3d_gl, np.array(anchors_x3d[aind])))
                        anchors_y3d_gl = np.hstack((anchors_y3d_gl, np.array(anchors_y3d[aind])))
                        anchors_z3d_gl = np.hstack((anchors_z3d_gl, np.array(anchors_z3d[aind])))
                        anchors_w3d_gl = np.hstack((anchors_w3d_gl, np.array(anchors_w3d[aind])))
                        anchors_h3d_gl = np.hstack((anchors_h3d_gl, np.array(anchors_h3d[aind])))
                        anchors_l3d_gl = np.hstack((anchors_l3d_gl, np.array(anchors_l3d[aind])))
                        anchors_rotY_gl = np.hstack((anchors_rotY_gl, np.array(anchors_rotY[aind])))
                        anchors_elv_gl = np.hstack((anchors_elv_gl, np.array(anchors_elv[aind])))
                        anchors_sin_gl = np.hstack((anchors_sin_gl, np.array(anchors_sin[aind])))
                        anchors_cos_gl = np.hstack((anchors_cos_gl, np.array(anchors_cos[aind])))
                        anchors_vel_gl = np.hstack((anchors_vel_gl, np.array(anchors_vel[aind])))

                        anchors[aind, 4] = np.mean(np.array(anchors_z3d[aind]))
                        anchors[aind, 5] = np.mean(np.array(anchors_w3d[aind]))
                        anchors[aind, 6] = np.mean(np.array(anchors_h3d[aind]))
                        anchors[aind, 7] = np.mean(np.array(anchors_l3d[aind]))
                        anchors[aind, 8] = np.mean(np.array(anchors_rotY[aind]))

                        if decomp_alpha:
                            anchors[aind, 9] = np.mean(np.array(anchors_sin[aind]))
                            anchors[aind, 10] = np.mean(np.array(anchors_cos[aind]))

                        if has_vel:
                            anchors[aind, 11] = np.mean(np.array(anchors_vel[aind]))

                else:
                    logging.info('WARNING: Non-used anchor #{} found. Removing this anchor.'.format(aind))
                    anchors[aind, :] = -1

        # remove non-used
        anchors = anchors[np.all(anchors == -1, axis=1) ==  False, :]

        logging.info('Anchor info')

        for aind, anchor in enumerate(anchors):
            w = anchor[2] - anchor[0] + 1
            h = anchor[3] - anchor[1] + 1
            ar = w / h
            line = 'anchor {:2} w: {:6.2f}, h: {:6.2f}, ar: {:.2f}, z: {:5.2f}, w3d: {:.2f}, h3d: {:.2f}, l3d: {:.2f}, rot: {:5.2f}'.format(
                aind, w, h, ar, anchor[4], anchor[5], anchor[6], anchor[7], anchor[8]
            )

            if decomp_alpha:
                line += ', sin: {:6.2f}, cos: {:6.2f}'.format(anchor[9], anchor[10])

            if has_vel:
                line += ', vel: {:6.2f}'.format(anchor[11])

            logging.info(line)

        if (cache_folder is not None):
            pickle_write(os.path.join(cache_folder, 'anchors.pkl'), anchors)

    conf.anchors = anchors


def anchor_center(w, h, stride):
    """
    Centers an anchor based on a stride and the anchor shape (w, h).

    center ground truths with steps of half stride
    hence box 0 is centered at (7.5, 7.5) rather than (0, 0)
    for a feature stride of 16 px.
    """

    anchor = np.zeros([4], dtype=np.float32)

    anchor[0] = -w / 2 + (stride - 1) / 2
    anchor[1] = -h / 2 + (stride - 1) / 2
    anchor[2] = w / 2 + (stride - 1) / 2
    anchor[3] = h / 2 + (stride - 1) / 2

    return anchor


def cluster_anchors(feat_stride, anchors, test_scale, imdb, lbls, ilbls, anchor_ratios, min_gt_vis=0.99,
                    min_gt_h=0, max_gt_h=10e10, even_anchor_distribution=False, expand_anchors=False,
                    expand_stop_dt=0.0025):
    """
    Clusters the anchors based on the imdb boxes (in 2D and/or 3D).

    Generally, this method does a custom k-means clustering using 2D IoU
    as a distance metric.
    """

    normalized_gts = []

    # keep track if using 3d
    has_3d = False

    # check all images
    for imind, imobj in enumerate(imdb):

        # has ground truths?
        if len(imobj.gts) > 0:

            scale = imobj.scale * test_scale / imobj.imH

            # determine ignores
            igns, rmvs = determine_ignores(imobj.gts, lbls, ilbls, min_gt_vis, min_gt_h, np.inf, scale, use_trunc=True)

            # check for 3d box
            has_3d = 'bbox_3d' in imobj.gts[0]

            # accumulate boxes
            gts_all = bbXYWH2Coords(np.array([gt.bbox_full * scale for gt in imobj.gts]))
            gts_val = gts_all[(rmvs == False) & (igns == False), :]

            if has_3d:
                gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

            if gts_val.shape[0] > 0:

                # center all 2D ground truths
                for gtind in range(0, gts_val.shape[0]):

                    w = gts_val[gtind, 2] - gts_val[gtind, 0] + 1
                    h = gts_val[gtind, 3] - gts_val[gtind, 1] + 1

                    gts_val[gtind, 0:4] = anchor_center(w, h, feat_stride)

            if gts_val.shape[0] > 0:

                # add normalized gts given 3d or 2d boxes
                if has_3d: normalized_gts += np.concatenate((gts_val, gts_3d), axis=1).tolist()
                else: normalized_gts += gts_val.tolist()

    # convert to np
    normalized_gts = np.array(normalized_gts)

    logging.info('starting clustering with {} ground truths'.format(normalized_gts.shape[0]))

    # sort by height
    sorted_inds = np.argsort((normalized_gts[:, 3] - normalized_gts[:, 1] + 1))
    normalized_gts = normalized_gts[sorted_inds, :]

    # init expand
    best_anchors = anchors
    expand_last_iou = 0
    expand_dif = 1
    best_met = 0
    best_cov = 0

    # init cluster
    max_rounds = 50
    round = 0
    last_iou = 0
    dif = 1

    while round < max_rounds and dif > -1000.0:

        # make empty arrays for each anchor
        anchors_h = [[] for x in range(anchors.shape[0])]
        anchors_w = [[] for x in range(anchors.shape[0])]

        if has_3d:

            # bbox_3d order --> [cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY]
            anchors_z3d = [[] for x in range(anchors.shape[0])]
            anchors_w3d = [[] for x in range(anchors.shape[0])]
            anchors_h3d = [[] for x in range(anchors.shape[0])]
            anchors_l3d = [[] for x in range(anchors.shape[0])]
            anchors_rotY = [[] for x in range(anchors.shape[0])]

        round_ious = []
        round_zers = []
        round_mets = []

        # find best matches for each ground truth
        zers = np.abs(anchors[:, 4, np.newaxis] - normalized_gts[np.newaxis, :, 6])
        ols = iou(anchors[:, 0:4], normalized_gts[:, 0:4])

        metric = ols # - zers*0.01
        gt_target_anchor_learned = np.argmax(metric, axis=0)
        gt_target_anchor_matched = np.argmax(ols, axis=0)

        # to index manually metric[gt_target_anchor, range(metric.shape[1])]
        gt_mets = metric[gt_target_anchor_learned, range(metric.shape[1])]
        gt_ols = ols[gt_target_anchor_matched, range(metric.shape[1])]
        gt_zers = zers[gt_target_anchor_matched, range(metric.shape[1])]

        # assign each box to an anchor
        for gtind, gt in enumerate(normalized_gts):

            anum = gt_target_anchor_learned[gtind]

            w = gt[2] - gt[0] + 1
            h = gt[3] - gt[1] + 1

            anchors_h[anum].append(h)
            anchors_w[anum].append(w)

            if has_3d:
                anchors_z3d[anum].append(gt[6])
                anchors_w3d[anum].append(gt[7])
                anchors_h3d[anum].append(gt[8])
                anchors_l3d[anum].append(gt[9])
                anchors_rotY[anum].append(gt[10])

            # compute error by IoU matching
            round_ious.append(gt_ols[gtind])
            round_zers.append(gt_zers[gtind])
            round_mets.append(gt_mets[gtind])

        # compute errors
        cur_iou = np.mean(np.array(round_ious))
        cur_zer = np.mean(np.array(round_zers))
        cur_met = np.mean(np.array(round_mets))

        # update anchors
        for aind in range(0, anchors.shape[0]):

            # compute mean h/w
            if len(np.array(anchors_h[aind])) > 0:

                mean_h = np.mean(np.array(anchors_h[aind]))
                mean_w = np.mean(np.array(anchors_w[aind]))

                anchors[aind, 0:4] = anchor_center(mean_w, mean_h, feat_stride)

                if has_3d:
                    anchors[aind, 4] = np.mean(np.array(anchors_z3d[aind]))
                    anchors[aind, 5] = np.mean(np.array(anchors_w3d[aind]))
                    anchors[aind, 6] = np.mean(np.array(anchors_h3d[aind]))
                    anchors[aind, 7] = np.mean(np.array(anchors_l3d[aind]))
                    anchors[aind, 8] = np.mean(np.array(anchors_rotY[aind]))

            else:
                raise ValueError('Non-used anchor #{} found'.format(aind))

        # store best configuration
        if cur_met > best_met:
            best_met = cur_iou
            best_anchors = anchors
            iou_cov = np.mean(np.array(round_ious) >= 0.5)
            zer_cov = np.mean(np.array(round_zers) <= 0.5)

            logging.info('clustering before update round {}, iou={:.4f}, z={:.4f}, met={:.4f}, iou_cov={:.2f}, z_cov={:.4f}'.format(round, cur_iou, cur_zer, cur_met, iou_cov, zer_cov))

        dif = cur_iou - last_iou
        last_iou = cur_iou

        round += 1

    return best_anchors


def compute_targets(gts_val, gts_ign, box_lbls, rois, fg_thresh, ign_thresh, bg_thresh_lo, bg_thresh_hi, best_thresh,
                    gts_3d=None, anchors=[], tracker=[], rois_3d=None, rois_3d_cen=None):
    """
    Computes the bbox targets of a set of rois and a set
    of ground truth boxes, provided various ignore
    settings in configuration
    """

    ols = None
    has_3d = gts_3d is not None
    decomp_alpha = anchors.shape[1] >= 11
    has_vel = anchors.shape[1] == 12

    # init transforms which respectively hold [dx, dy, dw, dh, label]
    # for labels bg=-1, ign=0, fg>=1
    transforms = np.zeros([len(rois), 5], dtype=np.float32)
    raw_gt = np.zeros([len(rois), 5], dtype=np.float32)

    # if 3d, then init other terms after
    if has_3d:
        transforms = np.pad(transforms, [(0, 0), (0, gts_3d.shape[1] + decomp_alpha*2 + has_vel)], 'constant')
        raw_gt = np.pad(raw_gt, [(0, 0), (0, gts_3d.shape[1])], 'constant')

    if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

        if gts_ign.shape[0] > 0:

            # compute overlaps ign
            ols_ign = iou_ign(rois, gts_ign)
            ols_ign_max = np.amax(ols_ign, axis=1)

        else:
            ols_ign_max = np.zeros([rois.shape[0]], dtype=np.float32)

        if gts_val.shape[0] > 0:

            # compute overlaps valid
            ols = iou(rois, gts_val)
            ols_max = np.amax(ols, axis=1)
            targets = np.argmax(ols, axis=1)

            # find best matches for each ground truth
            gt_best_rois = np.argmax(ols, axis=0)
            gt_best_ols = np.amax(ols, axis=0)

            gt_best_rois = gt_best_rois[gt_best_ols >= best_thresh]
            gt_best_ols = gt_best_ols[gt_best_ols >= best_thresh]

            fg_inds = np.flatnonzero(ols_max >= fg_thresh)
            fg_inds = np.concatenate((fg_inds, gt_best_rois))
            fg_inds = np.unique(fg_inds)

            target_rois = gts_val[targets[fg_inds], :]
            src_rois = rois[fg_inds, :]

            if len(fg_inds) > 0:

                # compute 2d transform
                transforms[fg_inds, 0:4] = bbox_transform(src_rois, target_rois)

                raw_gt[fg_inds, 0:4] = target_rois

                if has_3d:

                    tracker = tracker.astype(np.int64)
                    if rois_3d is None:
                        src_3d = anchors[tracker[fg_inds], 4:]
                    else:
                        src_3d = rois_3d[fg_inds, 4:]
                    target_3d = gts_3d[targets[fg_inds]]

                    raw_gt[fg_inds, 5:] = target_3d

                    if rois_3d_cen is None:

                        # compute 3d transform
                        transforms[fg_inds, 5:] = bbox_transform_3d(src_rois, src_3d, target_3d, decomp_alpha=decomp_alpha, has_vel=has_vel)

                    else:
                        transforms[fg_inds, 5:] = bbox_transform_3d(src_rois, src_3d, target_3d, decomp_alpha=decomp_alpha, has_vel=has_vel, rois_3d_cen=rois_3d_cen[fg_inds])


                # store labels
                transforms[fg_inds, 4] = [box_lbls[x] for x in targets[fg_inds]]
                assert (all(transforms[fg_inds, 4] >= 1))

        else:

            ols_max = np.zeros(rois.shape[0], dtype=int)
            fg_inds = np.empty(shape=[0])
            gt_best_rois = np.empty(shape=[0])

        # determine ignores
        ign_inds = np.flatnonzero(ols_ign_max >= ign_thresh)

        # determine background
        bg_inds = np.flatnonzero((ols_max >= bg_thresh_lo) & (ols_max < bg_thresh_hi))

        # subtract fg and igns from background
        bg_inds = np.setdiff1d(bg_inds, ign_inds)
        bg_inds = np.setdiff1d(bg_inds, fg_inds)
        bg_inds = np.setdiff1d(bg_inds, gt_best_rois)

        # mark background
        transforms[bg_inds, 4] = -1

    else:

        # all background
        transforms[:, 4] = -1


    return transforms, ols, raw_gt


def clsInd2Name(lbls, ind):
    """
    Converts a cls ind to string name
    """

    if ind>=0 and ind<len(lbls):
        return lbls[ind]
    else:
        raise ValueError('unknown class')


def clsName2Ind(lbls, cls):
    """
    Converts a cls name to an ind
    """
    if cls in lbls:
        return lbls.index(cls) + 1
    else:
        raise ValueError('unknown class')


def compute_bbox_stats(conf, imdb, cache_folder=''):
    """
    Computes the mean and standard deviation for each regression
    parameter (usually pertaining to [dx, dy, sw, sh] but sometimes
    for 3d parameters too).

    Once these stats are known we normalize the regression targets
    to have 0 mean and 1 variance, to hypothetically ease training.
    """

    if (cache_folder is not None) and os.path.exists(os.path.join(cache_folder, 'bbox_means.pkl')) \
            and os.path.exists(os.path.join(cache_folder, 'bbox_stds.pkl')):

        means = pickle_read(os.path.join(cache_folder, 'bbox_means.pkl'))
        stds = pickle_read(os.path.join(cache_folder, 'bbox_stds.pkl'))

    else:


        if ('has_vel' in conf) and conf.has_vel:
            squared_sums = np.zeros([1, 14], dtype=np.float128)
            sums = np.zeros([1, 14], dtype=np.float128)
        elif ('decomp_alpha' in conf) and conf.decomp_alpha:
            squared_sums = np.zeros([1, 13], dtype=np.float128)
            sums = np.zeros([1, 13], dtype=np.float128)
        elif conf.has_3d:
            squared_sums = np.zeros([1, 11], dtype=np.float128)
            sums = np.zeros([1, 11], dtype=np.float128)
        else:
            squared_sums = np.zeros([1, 4], dtype=np.float128)
            sums = np.zeros([1, 4], dtype=np.float128)

        class_counts = np.zeros([1], dtype=np.float128) + 1e-10
        class_counts_vel = np.zeros([1], dtype=np.float128) + 1e-10

        # compute the mean first
        logging.info('Computing bbox regression mean..')

        for imind, imobj in enumerate(imdb):

            if len(imobj.gts) > 0:

                scale_factor = imobj.scale * conf.test_scale / imobj.imH
                feat_size = calc_output_size(np.array([imobj.imH, imobj.imW]) * scale_factor, conf.feat_stride)
                rois = locate_anchors(conf.anchors, feat_size, conf.feat_stride)

                # determine ignores
                igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls, conf.min_gt_vis,
                                               conf.min_gt_h, np.inf, scale_factor, use_trunc=True)

                # accumulate boxes
                gts_all = bbXYWH2Coords(np.array([gt.bbox_full * scale_factor for gt in imobj.gts]))

                # filter out irrelevant cls, and ignore cls
                gts_val = gts_all[(rmvs == False) & (igns == False), :]
                gts_ign = gts_all[(rmvs == False) & (igns == True), :]

                # accumulate labels
                box_lbls = np.array([gt.cls for gt in imobj.gts])
                box_lbls = box_lbls[(rmvs == False) & (igns == False)]
                box_lbls = np.array([clsName2Ind(conf.lbls, cls) for cls in box_lbls])

                if conf.has_3d:

                    # accumulate 3d boxes
                    gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                    gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

                    # rescale centers (in 2d)
                    for gtind, gt in enumerate(gts_3d):
                        gts_3d[gtind, 0:2] *= scale_factor

                    # compute transforms for all 3d
                    transforms, _, _= compute_targets(gts_val, gts_ign, box_lbls, rois, conf.fg_thresh, conf.ign_thresh,
                                                    conf.bg_thresh_lo, conf.bg_thresh_hi, conf.best_thresh, gts_3d=gts_3d,
                                                    anchors=conf.anchors, tracker=rois[:, 4])
                else:

                    # compute transforms for 2d
                    transforms, _, _ = compute_targets(gts_val, gts_ign, box_lbls, rois, conf.fg_thresh, conf.ign_thresh,
                                                    conf.bg_thresh_lo, conf.bg_thresh_hi, conf.best_thresh)

                gt_inds = np.flatnonzero(transforms[:, 4] > 0)

                if len(gt_inds) > 0:

                    if ('has_vel' in conf) and conf.has_vel:
                        sums[:, 0:4] += np.sum(transforms[gt_inds, 0:4], axis=0)
                        sums[:, 4:13] += np.sum(transforms[gt_inds, 5:14], axis=0)

                        valid_vel = transforms[gt_inds, 14] > (-np.inf)
                        sums[:, 13] += transforms[gt_inds, 14][valid_vel].sum()
                        class_counts_vel += valid_vel.sum()

                    elif ('decomp_alpha' in conf) and conf.decomp_alpha:
                        sums[:, 0:4] += np.sum(transforms[gt_inds, 0:4], axis=0)
                        sums[:, 4:] += np.sum(transforms[gt_inds, 5:14], axis=0)
                    elif conf.has_3d:
                        sums[:, 0:4] += np.sum(transforms[gt_inds, 0:4], axis=0)
                        sums[:, 4:] += np.sum(transforms[gt_inds, 5:12], axis=0)
                    else:
                        sums += np.sum(transforms[gt_inds, 0:4], axis=0)

                    class_counts += len(gt_inds)

        means = sums/class_counts

        if ('has_vel' in conf) and conf.has_vel:
            means[:, 13] = sums[:, 13] / class_counts_vel

        logging.info('Computing bbox regression stds..')

        for imobj in imdb:

            if len(imobj.gts) > 0:

                scale_factor = imobj.scale * conf.test_scale / imobj.imH
                feat_size = calc_output_size(np.array([imobj.imH, imobj.imW]) * scale_factor, conf.feat_stride)
                rois = locate_anchors(conf.anchors, feat_size, conf.feat_stride)

                # determine ignores
                igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls, conf.min_gt_vis, conf.min_gt_h, np.inf, scale_factor)

                # accumulate boxes
                gts_all = bbXYWH2Coords(np.array([gt.bbox_full * scale_factor for gt in imobj.gts]))

                # filter out irrelevant cls, and ignore cls
                gts_val = gts_all[(rmvs == False) & (igns == False), :]
                gts_ign = gts_all[(rmvs == False) & (igns == True), :]

                # accumulate labels
                box_lbls = np.array([gt.cls for gt in imobj.gts])
                box_lbls = box_lbls[(rmvs == False) & (igns == False)]
                box_lbls = np.array([clsName2Ind(conf.lbls, cls) for cls in box_lbls])

                if conf.has_3d:

                    # accumulate 3d boxes
                    gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                    gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

                    # rescale centers (in 2d)
                    for gtind, gt in enumerate(gts_3d):
                        gts_3d[gtind, 0:2] *= scale_factor

                    # compute transforms for all 3d
                    transforms, _, _ = compute_targets(gts_val, gts_ign, box_lbls, rois, conf.fg_thresh, conf.ign_thresh,
                                                    conf.bg_thresh_lo, conf.bg_thresh_hi, conf.best_thresh, gts_3d=gts_3d,
                                                    anchors=conf.anchors, tracker=rois[:, 4])
                else:

                    # compute transforms for 2d
                    transforms, _, _ = compute_targets(gts_val, gts_ign, box_lbls, rois, conf.fg_thresh, conf.ign_thresh,
                                                    conf.bg_thresh_lo, conf.bg_thresh_hi, conf.best_thresh)

                gt_inds = np.flatnonzero(transforms[:, 4] > 0)

                if len(gt_inds) > 0:

                    if ('has_vel' in conf) and conf.has_vel:
                        squared_sums[:, 0:4] += np.sum(np.power(transforms[gt_inds, 0:4] - means[:, 0:4], 2), axis=0)
                        squared_sums[:, 4:13] += np.sum(np.power(transforms[gt_inds, 5:14] - means[:, 4:13], 2), axis=0)
                        valid_vel = transforms[gt_inds, 14] > (-np.inf)
                        squared_sums[:, 13] += np.power(transforms[gt_inds, 14][valid_vel] - means[:, 13], 2).sum()
                    elif ('decomp_alpha' in conf) and conf.decomp_alpha:
                        squared_sums[:, 0:4] += np.sum(np.power(transforms[gt_inds, 0:4] - means[:, 0:4], 2), axis=0)
                        squared_sums[:, 4:] += np.sum(np.power(transforms[gt_inds, 5:14] - means[:, 4:], 2), axis=0)
                    elif conf.has_3d:
                        squared_sums[:, 0:4] += np.sum(np.power(transforms[gt_inds, 0:4] - means[:, 0:4], 2), axis=0)
                        squared_sums[:, 4:] += np.sum(np.power(transforms[gt_inds, 5:12] - means[:, 4:], 2), axis=0)

                    else:
                        squared_sums += np.sum(np.power(transforms[gt_inds, 0:4] - means, 2), axis=0)

        stds = np.sqrt((squared_sums/class_counts))

        if ('has_vel' in conf) and conf.has_vel:
            stds[:, 13] = np.sqrt((squared_sums[:, 13]/class_counts_vel))

        means = means.astype(float)
        stds = stds.astype(float)

        logging.info('used {:d} boxes with avg std {:.4f}'.format(int(class_counts[0]), np.mean(stds)))

        if (cache_folder is not None):
            pickle_write(os.path.join(cache_folder, 'bbox_means.pkl'), means)
            pickle_write(os.path.join(cache_folder, 'bbox_stds.pkl'), stds)

    conf.bbox_means = means
    conf.bbox_stds = stds


def flatten_tensor(input):
    """
    Flattens and permutes a tensor from size
    [B x C x W x H] --> [B x (W x H) x C]
    """

    bsize = input.shape[0]
    csize = input.shape[1]

    return input.permute(0, 2, 3, 1).contiguous().view(bsize, -1, csize)


def unflatten_tensor(input, feat_size, anchors):
    """
    Un-flattens and un-permutes a tensor from size
    [B x (W x H) x C] --> [B x C x W x H]
    """

    bsize = input.shape[0]

    if len(input.shape) >= 3: csize = input.shape[2]
    else: csize = 1

    input = input.view(bsize, feat_size[0] * anchors.shape[0], feat_size[1], csize)
    input = input.permute(0, 3, 1, 2).contiguous()

    return input


def bbCoords2XYWH(box):
    """
    Convert from [x1, y1, x2, y2] to [x,y,w,h]
    """

    if box.shape[0] == 0: return np.empty([0, 4], dtype=float)

    box[:, 2] -= box[:, 0] + 1
    box[:, 3] -= box[:, 1] + 1

    return box


def bbXYWH2Coords(box):
    """
    Convert from [x,y,w,h] to [x1, y1, x2, y2]
    """

    if box.shape[0] == 0: return np.empty([0,4], dtype=float)

    box[:, 2] += box[:, 0] - 1
    box[:, 3] += box[:, 1] - 1

    return box


def bbox_transform_3d(ex_rois_2d, ex_rois_3d, gt_rois, decomp_alpha=False, has_vel=False, rois_3d_cen=None):
    """
    Compute the bbox target transforms in 3D.

    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    ex_widths = ex_rois_2d[:, 2] - ex_rois_2d[:, 0] + 1.0
    ex_heights = ex_rois_2d[:, 3] - ex_rois_2d[:, 1] + 1.0

    if rois_3d_cen is None:
        ex_ctr_x = ex_rois_2d[:, 0] + 0.5 * (ex_widths)
        ex_ctr_y = ex_rois_2d[:, 1] + 0.5 * (ex_heights)
    else:
        ex_ctr_x = rois_3d_cen[:, 0]
        ex_ctr_y = rois_3d_cen[:, 1]

    gt_ctr_x = gt_rois[:, 0]
    gt_ctr_y = gt_rois[:, 1]

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights

    delta_z = gt_rois[:, 2] - ex_rois_3d[:, 0]
    scale_w = np.log(gt_rois[:, 3] / ex_rois_3d[:, 1])
    scale_h = np.log(gt_rois[:, 4] / ex_rois_3d[:, 2])
    scale_l = np.log(gt_rois[:, 5] / ex_rois_3d[:, 3])
    deltaRotY = gt_rois[:, 6] - ex_rois_3d[:, 4]

    if decomp_alpha:
        delta_sin = gt_rois[:, 12] - ex_rois_3d[:, 5] #np.sin(gt_rois[:, 12] - ex_rois_3d[:, 5])
        delta_cos = gt_rois[:, 13] - ex_rois_3d[:, 6] #np.cos(gt_rois[:, 13] - ex_rois_3d[:, 6] + math.pi/2)

        if has_vel:
            delta_vel = np.ones(delta_sin.shape)*(-np.inf) if gt_rois.shape[1] != 17 else gt_rois[:, 16] - ex_rois_3d[:, 7]
            targets = np.vstack((targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY, delta_sin, delta_cos, delta_vel)).transpose()
        else:
            targets = np.vstack((targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY, delta_sin, delta_cos)).transpose()
    else:
        targets = np.vstack((targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY)).transpose()

    targets = np.hstack((targets, gt_rois[:, 7:]))


    return targets


def bbox_transform(ex_rois, gt_rois):
    """
    Compute the bbox target transforms in 2D.

    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights)

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets


def bbox_transform_inv(boxes, deltas, means=None, stds=None):
    """
    Compute the bbox target transforms in 3D.

    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    # boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths)
    ctr_y = boxes[:, 1] + 0.5 * (heights)

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    if stds is not None:
        dx *= stds[0]
        dy *= stds[1]
        dw *= stds[2]
        dh *= stds[3]

    if means is not None:
        dx += means[0]
        dy += means[1]
        dw += means[2]
        dh += means[3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros(deltas.shape)

    # x1, y1, x2, y2
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * (pred_w)
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * (pred_h)
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * (pred_w) - 1
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * (pred_h) - 1

    return pred_boxes


def determine_ignores(gts, lbls, ilbls, min_gt_vis=0.99, min_gt_h=0, max_gt_h=10e10, scale_factor=1, use_trunc=False):
    """
    Given various configuration settings, determine which ground truths
    are ignored and which are relevant.
    """

    igns = np.zeros([len(gts)], dtype=bool)
    rmvs = np.zeros([len(gts)], dtype=bool)

    for gtind, gt in enumerate(gts):

        ign = gt.ign
        ign |= gt.visibility < min_gt_vis
        ign |= gt.bbox_full[3] * scale_factor < min_gt_h
        ign |= gt.bbox_full[3] * scale_factor > max_gt_h
        ign |= gt.cls in ilbls

        if use_trunc:
            ign |= gt.trunc > max(1 - min_gt_vis, 0)

        rmv = not gt.cls in (lbls + ilbls)

        igns[gtind] = ign
        rmvs[gtind] = rmv

    return igns, rmvs


def locate_anchors(anchors, feat_size, stride, convert_tensor=False):
    """
    Spreads each anchor shape across a feature map of size feat_size spaced by a known stride.

    Args:
        anchors (ndarray): N x 4 array describing [x1, y1, x2, y2] displacements for N anchors
        feat_size (ndarray): the downsampled resolution W x H to spread anchors across
        stride (int): stride of a network
        convert_tensor (bool, optional): whether to return a torch tensor, otherwise ndarray [default=False]

    Returns:
         ndarray: 2D array = [(W x H) x 5] array consisting of [x1, y1, x2, y2, anchor_index]
    """

    # compute rois
    shift_x = np.array(range(0, feat_size[1], 1)) * float(stride)
    shift_y = np.array(range(0, feat_size[0], 1)) * float(stride)
    [shift_x, shift_y] = np.meshgrid(shift_x, shift_y)

    rois = np.expand_dims(anchors[:, 0:4], axis=1)
    shift_x = np.expand_dims(shift_x, axis=0)
    shift_y = np.expand_dims(shift_y, axis=0)

    shift_x1 = shift_x + np.expand_dims(rois[:, :, 0], axis=2)
    shift_y1 = shift_y + np.expand_dims(rois[:, :, 1], axis=2)
    shift_x2 = shift_x + np.expand_dims(rois[:, :, 2], axis=2)
    shift_y2 = shift_y + np.expand_dims(rois[:, :, 3], axis=2)

    # compute anchor tracker
    anchor_tracker = np.zeros(shift_x1.shape, dtype=float)
    for aind in range(0, rois.shape[0]): anchor_tracker[aind, :, :] = aind

    stack_size = feat_size[0] * anchors.shape[0]

    # torch and numpy MAY have different calls for reshaping, although
    # it is not very important which is used as long as it is CONSISTENT
    if convert_tensor:

        # important to unroll according to pytorch
        shift_x1 = torch.from_numpy(shift_x1).view(1, stack_size, feat_size[1])
        shift_y1 = torch.from_numpy(shift_y1).view(1, stack_size, feat_size[1])
        shift_x2 = torch.from_numpy(shift_x2).view(1, stack_size, feat_size[1])
        shift_y2 = torch.from_numpy(shift_y2).view(1, stack_size, feat_size[1])
        anchor_tracker = torch.from_numpy(anchor_tracker).view(1, stack_size, feat_size[1])

        shift_x1.requires_grad = False
        shift_y1.requires_grad = False
        shift_x2.requires_grad = False
        shift_y2.requires_grad = False
        anchor_tracker.requires_grad = False

        shift_x1 = shift_x1.permute(1, 2, 0).contiguous().view(-1, 1)
        shift_y1 = shift_y1.permute(1, 2, 0).contiguous().view(-1, 1)
        shift_x2 = shift_x2.permute(1, 2, 0).contiguous().view(-1, 1)
        shift_y2 = shift_y2.permute(1, 2, 0).contiguous().view(-1, 1)
        anchor_tracker = anchor_tracker.permute(1, 2, 0).contiguous().view(-1, 1)

        rois = torch.cat((shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker), 1)

    else:

        shift_x1 = shift_x1.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_y1 = shift_y1.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_x2 = shift_x2.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_y2 = shift_y2.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        anchor_tracker = anchor_tracker.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)

        rois = np.concatenate((shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker), 1)

    return rois


def calc_output_size(res, stride):
    """
    Approximate the output size of a network

    Args:
        res (ndarray): input resolution
        stride (int): stride of a network

    Returns:
         ndarray: output resolution
    """

    return np.ceil(np.array(res)/stride).astype(int)


def im_detect_3d(im, net, rpn_conf, preprocess, p2, gpu=0, synced=False, return_base=False):
    """
    Object detection in 3D
    """

    imH_orig = im.shape[0]
    imW_orig = im.shape[1]

    p2_inv = np.linalg.inv(p2)

    im = preprocess(im)

    # move to GPU
    im = torch.from_numpy(im[np.newaxis, :, :, :]).cuda()

    imH = im.shape[2]
    imW = im.shape[3]

    scale_factor = imH / imH_orig

    if return_base:
        cls, prob, bbox_2d, bbox_3d, feat_size, rois, base = net(im, return_base=return_base)
    else:
        cls, prob, bbox_2d, bbox_3d, feat_size, rois = net(im)


    if not ('infer_2d_from_3d' in rpn_conf) or not (rpn_conf.infer_2d_from_3d):
        bbox_x = bbox_2d[:, :, 0]
        bbox_y = bbox_2d[:, :, 1]
        bbox_w = bbox_2d[:, :, 2]
        bbox_h = bbox_2d[:, :, 3]

    bbox_x3d = bbox_3d[:, :, 0]
    bbox_y3d = bbox_3d[:, :, 1]
    bbox_z3d = bbox_3d[:, :, 2]
    bbox_w3d = bbox_3d[:, :, 3]
    bbox_h3d = bbox_3d[:, :, 4]
    bbox_l3d = bbox_3d[:, :, 5]

    if ('orientation_bins' in rpn_conf) and rpn_conf.orientation_bins > 0:

        bbox_alpha = bbox_3d[:, :, 6]
        bbox_alpha_bins = bbox_3d[:, :, 7:]

    elif ('decomp_alpha' in rpn_conf) and rpn_conf.decomp_alpha:

        bbox_rsin = bbox_3d[:, :, 6]
        bbox_rcos = bbox_3d[:, :, 7]
        bbox_axis = bbox_3d[:, :, 8]
        bbox_head = bbox_3d[:, :, 9]
        bbox_rsin = bbox_rsin * rpn_conf.bbox_stds[:, 11][0] + rpn_conf.bbox_means[:, 11][0]
        bbox_rcos = bbox_rcos * rpn_conf.bbox_stds[:, 12][0] + rpn_conf.bbox_means[:, 12][0]

        # gb check
        if ('has_un' in rpn_conf) and rpn_conf.has_un:
            bbox_un = bbox_3d[:, :, 10]
    else:
        bbox_ry3d = bbox_3d[:, :, 6] * rpn_conf.bbox_stds[:, 10][0] + rpn_conf.bbox_means[:, 10][0]

    # detransform 3d
    bbox_x3d = bbox_x3d * rpn_conf.bbox_stds[:, 4][0] + rpn_conf.bbox_means[:, 4][0]
    bbox_y3d = bbox_y3d * rpn_conf.bbox_stds[:, 5][0] + rpn_conf.bbox_means[:, 5][0]
    bbox_z3d = bbox_z3d * rpn_conf.bbox_stds[:, 6][0] + rpn_conf.bbox_means[:, 6][0]
    bbox_w3d = bbox_w3d * rpn_conf.bbox_stds[:, 7][0] + rpn_conf.bbox_means[:, 7][0]
    bbox_h3d = bbox_h3d * rpn_conf.bbox_stds[:, 8][0] + rpn_conf.bbox_means[:, 8][0]
    bbox_l3d = bbox_l3d * rpn_conf.bbox_stds[:, 9][0] + rpn_conf.bbox_means[:, 9][0]

    # find 3d source
    tracker = rois[:, 4].cpu().detach().numpy().astype(np.int64)
    src_3d = torch.from_numpy(rpn_conf.anchors[tracker, 4:]).cuda().type(torch.cuda.FloatTensor)

    # compute 3d transform
    widths = rois[:, 2] - rois[:, 0] + 1.0
    heights = rois[:, 3] - rois[:, 1] + 1.0
    ctr_x = rois[:, 0] + 0.5 * widths
    ctr_y = rois[:, 1] + 0.5 * heights

    bbox_x3d = bbox_x3d[0, :] * widths + ctr_x
    bbox_y3d = bbox_y3d[0, :] * heights + ctr_y

    bbox_z3d = src_3d[:, 0] + bbox_z3d[0, :]
    bbox_w3d = torch.exp(bbox_w3d[0, :]) * src_3d[:, 1]
    bbox_h3d = torch.exp(bbox_h3d[0, :]) * src_3d[:, 2]
    bbox_l3d = torch.exp(bbox_l3d[0, :]) * src_3d[:, 3]

    if ('orientation_bins' in rpn_conf) and rpn_conf.orientation_bins > 0:

        bins = torch.arange(-math.pi, math.pi, 2 * math.pi / rpn_conf.orientation_bins)

        bbox_ry3d = (bins[bbox_alpha_bins[0].argmax(dim=1)] + bbox_alpha[0])

        a = 1

    elif ('decomp_alpha' in rpn_conf) and rpn_conf.decomp_alpha:

        bbox_rsin = src_3d[:, 5] + bbox_rsin[0, :]
        bbox_rcos = src_3d[:, 6] + bbox_rcos[0, :]
        bbox_axis_sin_mask = bbox_axis[0, :] >= 0.5
        bbox_head_pos_mask = bbox_head[0, :] >= 0.5

        bbox_ry3d = bbox_rcos
        bbox_ry3d[bbox_axis_sin_mask] = bbox_rsin[bbox_axis_sin_mask]
        bbox_ry3d[bbox_head_pos_mask] = bbox_ry3d[bbox_head_pos_mask] + math.pi
    else:
        bbox_ry3d = bbox_ry3d[0, :] + src_3d[:, 4]

    # bundle
    coords_3d = torch.stack((bbox_x3d, bbox_y3d, bbox_z3d[:bbox_x3d.shape[0]], bbox_w3d[:bbox_x3d.shape[0]], bbox_h3d[:bbox_x3d.shape[0]], bbox_l3d[:bbox_x3d.shape[0]], bbox_ry3d[:bbox_x3d.shape[0]]), dim=1)

    if ('use_el_z' in rpn_conf) and rpn_conf.use_el_z:
        coords_3d = torch.cat((coords_3d, bbox_un.t()), dim=1)
    elif ('has_un' in rpn_conf) and rpn_conf.has_un:
        coords_3d = torch.cat((coords_3d, bbox_un.t(), bbox_3d[0, :, 11:]), dim=1)


    if not ('infer_2d_from_3d' in rpn_conf) or not (rpn_conf.infer_2d_from_3d):

        # compile deltas pred
        deltas_2d = torch.cat((bbox_x[0, :, np.newaxis], bbox_y[0, :, np.newaxis], bbox_w[0, :, np.newaxis], bbox_h[0, :, np.newaxis]), dim=1)
        coords_2d = bbox_transform_inv(rois, deltas_2d, means=rpn_conf.bbox_means[0, :], stds=rpn_conf.bbox_stds[0, :])

        # detach onto cpu
        coords_2d = coords_2d.cpu().detach().numpy()
        coords_3d = coords_3d.cpu().detach().numpy()
        prob = prob[0, :, :].cpu().detach().numpy()

        # scale coords
        coords_2d[:, 0:4] /= scale_factor
        coords_3d[:, 0:2] /= scale_factor

        cls_pred = np.argmax(prob[:, 1:], axis=1) + 1
        scores = np.amax(prob[:, 1:], axis=1)

    else:
        coords_3d = coords_3d.cpu().detach().numpy()
        prob = prob[0, :, :].cpu().detach().numpy()

        # scale coords
        coords_3d[:, 0:2] /= scale_factor

        cls_pred = np.argmax(prob[:, 1:], axis=1) + 1
        scores = np.amax(prob[:, 1:], axis=1)

        fg_mask = scores > rpn_conf.score_thres
        fg_inds = np.flatnonzero(fg_mask)

        coords_3d = coords_3d[fg_inds, :]
        prob = prob[fg_inds, :]
        scores = scores[fg_inds]
        cls_pred = cls_pred[fg_inds]

        # get info
        x2d = coords_3d[:, 0]
        y2d = coords_3d[:, 1]
        z2d = coords_3d[:, 2]
        w3d = coords_3d[:, 3]
        h3d = coords_3d[:, 4]
        l3d = coords_3d[:, 5]
        alp = coords_3d[:, 6]

        coords_3d_proj = p2_inv.dot(np.vstack((x2d*z2d, y2d*z2d, z2d, np.ones(x2d.shape))))
        x3d = coords_3d_proj[0]
        y3d = coords_3d_proj[1]
        z3d = coords_3d_proj[2]

        ry3d = convertAlpha2Rot(alp, z3d, x3d)

        coords_2d, ign = get_2D_from_3D(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d)

    aboxes = np.hstack((coords_2d, scores[:, np.newaxis]))

    sorted_inds = (-aboxes[:, 4]).argsort()
    original_inds = (sorted_inds).argsort()
    aboxes = aboxes[sorted_inds, :]
    coords_3d = coords_3d[sorted_inds, :]
    cls_pred = cls_pred[sorted_inds]
    tracker = tracker[sorted_inds]

    if synced and aboxes.shape[0] > 0:

        # nms
        keep_inds = gpu_nms(aboxes[:, 0:5].astype(np.float32), rpn_conf.nms_thres, device_id=gpu)

        # convert to bool
        keep = np.zeros([aboxes.shape[0], 1], dtype=bool)
        keep[keep_inds, :] = True

        # stack the keep array,
        # sync to the original order
        aboxes = np.hstack((aboxes, keep))
        aboxes[original_inds, :]

    elif aboxes.shape[0] > 0:

        # pre-nms
        cls_pred = cls_pred[0:min(rpn_conf.nms_topN_pre, cls_pred.shape[0])]
        tracker = tracker[0:min(rpn_conf.nms_topN_pre, tracker.shape[0])]
        aboxes = aboxes[0:min(rpn_conf.nms_topN_pre, aboxes.shape[0]), :]
        coords_3d = coords_3d[0:min(rpn_conf.nms_topN_pre, coords_3d.shape[0])]

        # nms
        keep_inds = gpu_nms(aboxes[:, 0:5].astype(np.float32), rpn_conf.nms_thres, device_id=gpu)

        # stack cls prediction
        aboxes = np.hstack((aboxes, cls_pred[:, np.newaxis], coords_3d, tracker[:, np.newaxis]))

        # suppress boxes
        aboxes = aboxes[keep_inds, :]

    # clip boxes
    if rpn_conf.clip_boxes:
        aboxes[:, 0] = np.clip(aboxes[:, 0], 0, imW_orig - 1)
        aboxes[:, 1] = np.clip(aboxes[:, 1], 0, imH_orig - 1)
        aboxes[:, 2] = np.clip(aboxes[:, 2], 0, imW_orig - 1)
        aboxes[:, 3] = np.clip(aboxes[:, 3], 0, imH_orig - 1)

    if return_base:
        return aboxes, base
    else:
        return aboxes


def test_kitti_3d(dataset_test, net, rpn_conf, results_path, test_path, has_rcnn=False):
    """
    Test the KITTI framework for object detection in 3D
    """

    # import read_kitti_cal
    from lib.imdb_util import read_kitti_cal
    from lib.imdb_util import read_kitti_label

    imlist = list_files(os.path.join(test_path, dataset_test, 'validation', 'image_2', ''), '*'+rpn_conf.datasets_train[0]['im_ext'])

    preprocess = Preprocess([rpn_conf.test_scale], rpn_conf.image_means, rpn_conf.image_stds)

    # fix paths slightly
    _, test_iter, _ = file_parts(results_path.replace('/data', ''))
    test_iter = test_iter.replace('results_', '')

    # init
    test_start = time()

    for imind, impath in enumerate(imlist):

        im = cv2.imread(impath)

        base_path, name, ext = file_parts(impath)

        # read in calib
        p2 = read_kitti_cal(os.path.join(test_path, dataset_test, 'validation', 'calib', name + '.txt'))
        p2_inv = np.linalg.inv(p2)

        # forward test batch
        aboxes = im_detect_3d(im, net, rpn_conf, preprocess, p2)

        base_path, name, ext = file_parts(impath)

        file = open(os.path.join(results_path, name + '.txt'), 'w')
        text_to_write = ''

        for boxind in range(0, min(rpn_conf.nms_topN_post, aboxes.shape[0])):

            box = aboxes[boxind, :]
            score = box[4]
            cls = rpn_conf.lbls[int(box[5] - 1)]

            if ('has_un' in rpn_conf) and rpn_conf.has_un:
                un = score * box[13]

            if ('has_un' in rpn_conf) and rpn_conf.has_un and ('use_un_for_score' in rpn_conf) and rpn_conf.use_un_for_score:
                score = un

            if score > rpn_conf.score_thres and cls == 'Car':

                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                width = (x2 - x1 + 1)
                height = (y2 - y1 + 1)

                # plot 3D box
                x3d = box[6]
                y3d = box[7]
                z3d = box[8]
                w3d = box[9]
                h3d = box[10]
                l3d = box[11]
                ry3d = box[12]

                # convert alpha into ry3d
                coord3d = np.linalg.inv(p2).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
                ry3d = convertAlpha2Rot(ry3d, coord3d[2], coord3d[0])

                while ry3d > math.pi: ry3d -= math.pi * 2
                while ry3d <= (-math.pi): ry3d += math.pi * 2

                # predict a more accurate projection
                coord3d = np.linalg.inv(p2).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))

                x3d = coord3d[0]
                y3d = coord3d[1]
                z3d = coord3d[2]

                alpha = convertRot2Alpha(ry3d, z3d, x3d)

                y3d += h3d / 2

                if ('has_un' in rpn_conf) and rpn_conf.has_un and ('use_un_for_score' in rpn_conf) and rpn_conf.use_un_for_score:
                    score = un

                text_to_write += ('{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                                      + '{:.6f} {:.6f}\n').format(cls, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score)

        file.write(text_to_write)
        file.close()

        # display stats
        if (imind + 1) % 500 == 0:
            time_str, dt = compute_eta(test_start, imind + 1, len(imlist))
            logging.info('testing {}/{}, dt: {:0.3f}, eta: {}'.format(imind + 1, len(imlist), dt, time_str))


    evaluate_kitti_results_verbose(test_path, dataset_test, results_path, test_iter, rpn_conf)


def test_kitti_3d_kalman_boxes(dataset_test, net, rpn_conf, bbox_measure_val, p2s, p2_invs, scales, results_path, test_path, dataset_type='validation', fast=False, report_stats=False, eval_type='evaluate_object'):
    """
    Test the KITTI framework for object detection in 3D
    """

    from lib.imdb_util import read_kitti_label

    imlist = list_files(os.path.join(test_path, dataset_test, dataset_type, 'image_2', ''), '*.png')

    si_shots, te_shots, poses = net.forward_boxes(bbox_measure_val, p2s, p2_invs, scales)

    # init
    test_start = time()

    if report_stats:
        scores = []
        stats_iou = []
        stats_deg = []
        stats_mat = []
        stats_z = []

    visualize = 0

    if visualize:
        write_folder = '/home/garrick/Desktop/kitti_vis'
        mkdir_if_missing(write_folder, delete_if_exist=True)

    for imind, impath in enumerate(imlist):

        base_path, name, ext = file_parts(impath)

        p2 = p2s[imind].detach().cpu().numpy()

        if visualize:
            im = cv2.imread(impath)

        if report_stats:
            gts = read_kitti_label(os.path.join('data', dataset_test, dataset_type, 'label_2', name + '.txt'), p2)

            igns, rmvs = determine_ignores(gts, ['Car'], rpn_conf.ilbls, rpn_conf.min_gt_vis, rpn_conf.min_gt_h)

            gts_full = bbXYWH2Coords(np.array([gt['bbox_full'] for gt in gts]))
            gts_3d = np.array([gt['bbox_3d'] for gt in gts])
            gts_cen = np.array([gt['center_3d'] for gt in gts])
            gts_cls = np.array([gt['cls'] for gt in gts])

            gts_full = gts_full[(rmvs == False) & (igns == False), :]
            gts_3d = gts_3d[(rmvs == False) & (igns == False), :]
            gts_cen = gts_cen[(rmvs == False) & (igns == False), :]
            gts_cls = gts_cls[(rmvs == False) & (igns == False)]

        # first write normal_predictions
        file = open(os.path.join(results_path, name + '.txt'), 'w')
        text_to_write = ''

        if te_shots is None or te_shots[-1] is None:
            aboxes = []
        else:

            if te_shots[-1][imind] is None or len(te_shots[-1][imind].Xs) == 0:
                aboxes = []

            else:

                bbox_2d = te_shots[-1][imind].box2ds.detach().cpu().numpy()
                Xs = te_shots[-1][imind].Xs.detach().cpu().numpy()
                bbox_un = te_shots[-1][imind].bbox_un.unsqueeze(1).detach().cpu().numpy()

                # apply head
                Xs[Xs[:, 7] >= 0.5, 6] += math.pi
                Cs = te_shots[-1][imind].Cs.detach().cpu().numpy()
                Cs = np.array([a.diagonal() for a in Cs])

                # recall that Xs is:
                # [x, y, z, w, h, l, theta, head, vel]
                aboxes = np.concatenate((bbox_2d, Xs, bbox_un, Cs), axis=1)


        for boxind in range(0, min(rpn_conf.nms_topN_post, len(aboxes))):

            box = aboxes[boxind, :]
            score = box[4]
            cls = rpn_conf.lbls[int(box[5] - 1)]

            if ('has_un' in rpn_conf) and rpn_conf.has_un:
                un = box[15]
                cs = 1/box[16:].sum()

            if un > rpn_conf.score_thres and cls=='Car':

                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                width = (x2 - x1 + 1)
                height = (y2 - y1 + 1)

                # for comparison only
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # plot 3D box
                x3d = box[6]
                y3d = box[7]
                z3d = box[8]
                w3d = box[9]
                h3d = box[10]
                l3d = box[11]
                ry3d = box[12]

                alpha = convertRot2Alpha(ry3d, z3d, x3d)

                if report_stats:

                    cur_2d = np.array([[x1, y1, x2, y2]])
                    verts_cur, corners_3d_cur = project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=True)

                    # match with gt
                    ols_gts = iou(cur_2d, gts_full)[0, :]
                    if ols_gts.shape[0] > 0:
                        gtind = np.argmax(ols_gts)
                        ol_gt = np.amax(ols_gts)
                    else:
                        ol_gt = 0

                    # found gt?
                    if ol_gt > 0.50:

                        # get gt values
                        gt_x3d = gts_cen[gtind, 0]
                        gt_y3d = gts_cen[gtind, 1]
                        gt_z3d = gts_cen[gtind, 2]
                        gt_w3d = gts_3d[gtind, 3]
                        gt_h3d = gts_3d[gtind, 4]
                        gt_l3d = gts_3d[gtind, 5]
                        gt_alpha = gts_3d[gtind, 6]
                        gt_rotY = gts_3d[gtind, 10]
                        gt_el = gts_3d[gtind, 11]

                        verts_cur, corners_3d_cur = project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=True)
                        verts_gts, corners_3d_gt = project_3d(p2, gt_x3d, gt_y3d, gt_z3d, gt_w3d, gt_h3d, gt_l3d, gt_rotY, return_3d=True)

                        # compute 3D IoU
                        iou_bev_cur, iou_3d_cur = iou3d(corners_3d_cur, corners_3d_gt, gt_h3d * gt_l3d * gt_w3d + h3d * l3d * w3d)

                        # ry3d_tmp = ry3d + math.pi * np.round((gt_rotY - ry3d) / math.pi)

                        stats_iou.append(iou_3d_cur)
                        stats_z.append(np.abs(gt_z3d - z3d))
                        stats_deg.append(np.rad2deg(np.abs(snap_to_pi(gt_alpha) - snap_to_pi(alpha))))

                        iou_3d_cur = np.round(iou_3d_cur, 2)

                        if cls == 'Cyclist' and iou_3d_cur >= 0.5:
                            success = True
                        elif cls == 'Pedesrian' and iou_3d_cur >= 0.5:
                            success = True
                        elif cls == 'Car' and iou_3d_cur >= 0.7:
                            success = True
                        else:
                            success = False

                        stats_mat.append(success)


                if visualize:
                    verts_cur, corners_3d_cur = project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=True)
                    draw_3d_box(im, verts_cur, thickness=2)

                y3d += h3d / 2
                alpha = convertRot2Alpha(ry3d, z3d, x3d)

                if ('has_un' in rpn_conf) and rpn_conf.has_un and ('use_un_for_score' in rpn_conf) and rpn_conf.use_un_for_score:
                    score = un

                text_to_write += ('{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                                  + '{:.6f} {:.6f}\n').format(cls, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score)

        file.write(text_to_write)
        file.close()

        if visualize:
            imwrite(im, os.path.join(write_folder, name+'.jpg'))

        # display stats
        if (imind + 1) % 100 == 0:
            time_str, dt = compute_eta(test_start, imind + 1, len(imlist))

            if report_stats:
                logging.info('testing {}/{}, dt: {:0.3f}, eta: {}, match: {:.4f}, iou: {:.4f}, z: {:.1f}, rot: {:.1f}'
                    .format(imind + 1, len(imlist), dt, time_str, np.array(stats_mat).mean(),
                            np.array(stats_iou).mean(), np.mean(stats_z), np.array(stats_deg).mean()))
            else:
                logging.info('testing {}/{}, dt: {:0.3f}, eta: {}'.format(imind + 1, len(imlist), dt, time_str))

    if report_stats:

        logging.info(
            'testing {}/{}, dt: {:0.3f}, eta: {}, match: {:.4f}, iou: {:.4f}, z: {:.1f}, rot: {:.1f}'
            .format(imind + 1, len(imlist), dt, time_str, np.array(stats_mat).mean(),
                    np.array(stats_iou).mean(), np.mean(stats_z), np.array(stats_deg).mean()))

    _, test_iter, _ = file_parts(results_path.replace('/data', ''))
    test_iter = test_iter.replace('results_', '')

    evaluate_kitti_results_verbose(test_path, dataset_test, results_path, test_iter, rpn_conf, fast=fast, default_eval=eval_type)


def extract_boxes(dataset_test, net, rpn_conf, test_path, data_folder):
    """
    Test the KITTI framework for object detection in 3D
    """

    # import read_kitti_cal
    from lib.imdb_util import read_kitti_cal
    from lib.imdb_util import read_kitti_label
    from lib.imdb_util import get_kitti_raw_ids

    if rpn_conf.video_count > 4:
        raw_ids = get_kitti_raw_ids(test_path, dataset_test)

    imlist = list_files(os.path.join(test_path, dataset_test, data_folder, 'image_2', ''), '*.png')

    preprocess = Preprocess(rpn_conf.crop_size, rpn_conf.image_means, rpn_conf.image_stds)

    # init
    test_start = time()

    bbox_measures = []

    for imind, impath in enumerate(imlist):

        aboxes_all = []

        base_path, name, ext = file_parts(impath)

        ims = cv2.imread(impath)

        # read in calib
        p2 = read_kitti_cal(os.path.join(test_path, dataset_test, data_folder, 'calib', name + '.txt'))
        p2_inv = np.linalg.inv(p2)

        aboxes = im_detect_3d(ims, net, rpn_conf, preprocess, p2)
        aboxes_all.append(aboxes)


        if rpn_conf.video_det:

            for vid_index in range(rpn_conf.video_count-1):

                if vid_index == 0:
                    impath_prev = os.path.join(test_path, dataset_test, data_folder, 'prev_2', name + '_01' + ext)
                elif vid_index == 1:
                    impath_prev = os.path.join(test_path, dataset_test, data_folder, 'prev_2', name + '_02' + ext)
                elif vid_index == 2:
                    impath_prev = os.path.join(test_path, dataset_test, data_folder, 'prev_2', name + '_03' + ext)
                else:
                    impath_prev = os.path.join(test_path, 'kitti', 'raw', raw_ids[imind][0][0:10], raw_ids[imind][0], 'image_02', 'data', '{:010d}'.format(max(int(raw_ids[imind][1]) - vid_index -1, 0)) + ext)

                im_tmp = cv2.imread(impath_prev)

                if im_tmp is not None:
                    ims = im_tmp

                aboxes = im_detect_3d(ims, net, rpn_conf, preprocess, p2)
                aboxes_all.append(aboxes)

        bbox_measures.append(aboxes_all)

        # display stats
        if (imind + 1) % 10 == 0:
            time_str, dt = compute_eta(test_start, imind + 1, len(imlist))
            print('testing {}/{}, dt: {:0.3f}, eta: {}'.format(imind + 1, len(imlist), dt, time_str))

    return bbox_measures

def extract_kalman_boxes(dataset_test, net, rpn_conf, test_path, data_folder):
    """
    Test the KITTI framework for object detection in 3D
    """

    # import read_kitti_cal
    from lib.imdb_util import read_kitti_cal
    from lib.imdb_util import read_kitti_label
    from lib.imdb_util import get_kitti_raw_ids

    if rpn_conf.video_count > 4:
        raw_ids = get_kitti_raw_ids(test_path, dataset_test)

    imlist = list_files(os.path.join(test_path, dataset_test, data_folder, 'image_2', ''), '*.png')

    preprocess = Preprocess(rpn_conf.crop_size, rpn_conf.image_means, rpn_conf.image_stds)

    # init
    test_start = time()

    bbox_measures = []

    for imind, impath in enumerate(imlist):

        base_path, name, ext = file_parts(impath)

        ims = cv2.imread(impath)

        # read in calib
        p2 = read_kitti_cal(os.path.join(test_path, dataset_test, data_folder, 'calib', name + '.txt'))
        p2_inv = np.linalg.inv(p2)

        h_before = ims.shape[0]
        ims = preprocess(ims)
        ims = torch.from_numpy(ims).cuda()

        if rpn_conf.video_det:

            '''
            im0 = imread(impath)
            im1 = imread(os.path.join(test_path, dataset_test, data_folder, 'prev_2', name + '_01' + ext))
            im2 = imread(os.path.join(test_path, dataset_test, data_folder, 'prev_2', name + '_02' + ext))
            im3 = imread(os.path.join(test_path, dataset_test, data_folder, 'prev_2', name + '_03' + ext))


            im4 = imread(os.path.join(test_path, 'kitti', 'raw', raw_ids[imind][0][0:10], raw_ids[imind][0], 'image_02', 'data', '{:010d}'.format(max(int(raw_ids[imind][1]) - 4, 0)) + ext))
            im5 = imread(os.path.join(test_path, 'kitti', 'raw', raw_ids[imind][0][0:10], raw_ids[imind][0], 'image_02', 'data', '{:010d}'.format(max(int(raw_ids[imind][1]) - 5, 0)) + ext))
            im6 = imread(os.path.join(test_path, 'kitti', 'raw', raw_ids[imind][0][0:10], raw_ids[imind][0], 'image_02', 'data', '{:010d}'.format(max(int(raw_ids[imind][1]) - 6, 0)) + ext))
            im7 = imread(os.path.join(test_path, 'kitti', 'raw', raw_ids[imind][0][0:10], raw_ids[imind][0], 'image_02', 'data', '{:010d}'.format(max(int(raw_ids[imind][1]) - 7, 0)) + ext))
            im8 = imread(os.path.join(test_path, 'kitti', 'raw', raw_ids[imind][0][0:10], raw_ids[imind][0], 'image_02', 'data', '{:010d}'.format(max(int(raw_ids[imind][1]) - 8, 0)) + ext))
            ims_tmp = np.concatenate((im7, im6, im5, im4, im3, im2, im1, im0), axis=1)

            imshow(ims_tmp)
            '''

            for vid_index in range(rpn_conf.video_count-1):

                if vid_index == 0:
                    impath_prev = os.path.join(test_path, dataset_test, data_folder, 'prev_2', name + '_01' + ext)
                elif vid_index == 1:
                    impath_prev = os.path.join(test_path, dataset_test, data_folder, 'prev_2', name + '_02' + ext)
                elif vid_index == 2:
                    impath_prev = os.path.join(test_path, dataset_test, data_folder, 'prev_2', name + '_03' + ext)
                else:
                    impath_prev = os.path.join(test_path, 'kitti', 'raw', raw_ids[imind][0][0:10], raw_ids[imind][0], 'image_02', 'data', '{:010d}'.format(max(int(raw_ids[imind][1]) - vid_index -1, 0)) + ext)

                im = cv2.imread(impath_prev)

                if im is None:
                    ims = torch.cat((ims, ims[-3:].clone()), dim=0)
                else:
                    im = preprocess(im)
                    im = torch.from_numpy(im).cuda()
                    ims = torch.cat((ims, im), dim=0)

        scale_factor = ims.shape[1] / h_before

        # forward test batch
        p2_inv = np.linalg.inv(p2)[np.newaxis, :, :]
        p2 = p2[np.newaxis, :, :]

        si_shots, tr_shots, poses = net(ims[np.newaxis, :, :, :], p2, p2_inv, [scale_factor])

        objs = []
        for vid_index in range(len(si_shots)):

            measure = si_shots[vid_index][0]
            if measure is not None:
                measure = measure.detach().cpu().numpy()

            pose = None if vid_index==0 else poses[vid_index-1].detach().cpu().numpy()

            objs.append((measure, pose))

        bbox_measures.append(objs)

        # display stats
        if (imind + 1) % 10 == 0:
            time_str, dt = compute_eta(test_start, imind + 1, len(imlist))
            logging.info('testing {}/{}, dt: {:0.3f}, eta: {}'.format(imind + 1, len(imlist), dt, time_str))

    return bbox_measures


def parse_kitti_result(respath, use_40=False):

    text_file = open(respath, 'r')

    acc = np.zeros([3, 41], dtype=float)

    lind = 0
    for line in text_file:

        parsed = re.findall('([\d]+\.?[\d]*)', line)

        for i, num in enumerate(parsed):
            acc[lind, i] = float(num)

        lind += 1

    text_file.close()

    if use_40:
        easy = np.mean(acc[0, 1:41:1])
        mod = np.mean(acc[1, 1:41:1])
        hard = np.mean(acc[2, 1:41:1])
    else:
        easy = np.mean(acc[0, 0:41:4])
        mod = np.mean(acc[1, 0:41:4])
        hard = np.mean(acc[2, 0:41:4])

    return easy, mod, hard


def run_kitti_eval_script(script_path, results_data, lbls, use_40=True):

    # evaluate primary experiment
    with open(os.devnull, 'w') as devnull:
        _ = subprocess.check_output([script_path, results_data], stderr=devnull)

    results_obj = edict()

    for lbl in lbls:

        lbl = lbl.lower()

        respath_2d = os.path.join(results_data, 'stats_{}_detection.txt'.format(lbl))
        respath_or = os.path.join(results_data, 'stats_{}_orientation.txt'.format(lbl))
        respath_gr = os.path.join(results_data, 'stats_{}_detection_ground.txt'.format(lbl))
        respath_3d = os.path.join(results_data, 'stats_{}_detection_3d.txt'.format(lbl))

        if os.path.exists(respath_2d):
            easy, mod, hard = parse_kitti_result(respath_2d, use_40=use_40)
            results_obj['det_2d_' + lbl] = [easy, mod, hard]
        if os.path.exists(respath_or):
            easy, mod, hard = parse_kitti_result(respath_or, use_40=use_40)
            results_obj['or_' + lbl] = [easy, mod, hard]
        if os.path.exists(respath_gr):
            easy, mod, hard = parse_kitti_result(respath_gr, use_40=use_40)
            results_obj['gr_' + lbl] = [easy, mod, hard]
        if os.path.exists(respath_3d):
            easy, mod, hard = parse_kitti_result(respath_3d, use_40=use_40)
            results_obj['det_3d_' + lbl] = [easy, mod, hard]

    return results_obj


def evaluate_kitti_results_verbose(test_path, dataset_test, results_path, test_iter, rpn_conf, use_logging=True, fast=False, default_eval='evaluate_object'):

    # evaluate primary experiment
    script = os.path.join(test_path, dataset_test, 'devkit', 'cpp', default_eval)

    results_obj = edict()

    task_keys = ['det_2d', 'or', 'gr', 'det_3d']

    # main experiment results
    results_obj.main = run_kitti_eval_script(script, results_path.replace('/data', ''), rpn_conf.lbls, use_40=True)

    # print main experimental results for each class, and each task
    for lbl in rpn_conf.lbls:

        lbl = lbl.lower()

        for task in task_keys:

            task_lbl = task + '_' + lbl

            if task_lbl in results_obj.main:

                easy = results_obj.main[task_lbl][0]
                mod = results_obj.main[task_lbl][1]
                hard = results_obj.main[task_lbl][2]

                print_str = 'test_iter {} {} {} --> easy: {:0.4f}, mod: {:0.4f}, hard: {:0.4f}'\
                    .format(test_iter, lbl, task.replace('det_', ''), easy, mod, hard)

                if use_logging:
                    logging.info(print_str)
                else:
                    print(print_str)

    if fast or ('fast_eval' in rpn_conf and rpn_conf.fast_eval): return

    iou_keys = ['0_1', '0_2', '0_3', '0_4', '0_5', '0_6', '0_7']
    dis_keys = ['15', '30', '45', '60']

    print('working on distance=', end='')

    for dis_key in dis_keys:

        print(dis_key, end='', flush=True)

        for iou_key in iou_keys:

            eval_key = 'evaluate_object_{}m_{}'.format(dis_key, iou_key)
            save_key = 'res_{}m_{}'.format(dis_key, iou_key)

            print('.', end='', flush=True)

            script = os.path.join(test_path, dataset_test, 'devkit', 'cpp', eval_key)
            tmp_obj = run_kitti_eval_script(script, results_path.replace('/data', ''), rpn_conf.lbls)
            results_obj[save_key] = tmp_obj

    print('')

    pickle_write(results_path.replace('/data', '/') + 'results_obj', results_obj)

    try:
        backend = plt.get_backend()
        plt.switch_backend('agg')
        save_kitti_ROC(results_obj, results_path.replace('/data', '/'), rpn_conf.lbls)
        print_kitti_ROC(results_obj, test_iter, rpn_conf.lbls, use_logging=use_logging)
        plt.switch_backend(backend)
    except:
        pass


def print_kitti_ROC(results_obj, test_iter, lbls, use_logging=True):


    iou_keys = ['0_1', '0_2', '0_3', '0_4', '0_5', '0_6', '0_7']
    dis_keys = ['15', '30', '45', '60']

    for lbl in lbls:

        lbl = lbl.lower()

        for dis in dis_keys:

            roc = [results_obj['res_{}m_{}'.format(dis, iou)]['det_3d_' + lbl][1] for iou in iou_keys]
            val = np.array(roc).mean()
            if lbl == 'car': legend = '{}m (av={:.4f}, 0.7={:.4f})'.format(dis, val, roc[6])
            else: legend = '{}m (av={:.4f}, 0.5={:.4f})'.format(dis, val, roc[4])

            if use_logging:
                logging.info(test_iter + ' ' + lbl + ' ' + legend)
            else:
                print(test_iter + ' ' + lbl + ' ' + legend)


def save_kitti_ROC(results_obj, folder, lbls):

    iou_keys = ['0_1', '0_2', '0_3', '0_4', '0_5', '0_6', '0_7']
    iou_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dis_keys = ['15', '30', '45', '60']

    for lbl in lbls:

        lbl = lbl.lower()

        for dis in dis_keys:

            roc = [results_obj['res_{}m_{}'.format(dis, iou)]['det_3d_' + lbl][1] for iou in iou_keys]
            val = np.array(roc).mean()
            if lbl == 'car': legend = '{}m ({:.4f}, @5 {:.4f}, @7 {:.4f})'.format(dis, val, roc[5], roc[6])
            else: legend = '{}m (av={:.4f}, 0.5={:.4f})'.format(dis, val, roc[4])

            plt.plot(iou_vals, roc, label=legend)

        plt.xlabel('3D IoU Criteria')
        plt.ylabel('AP')
        plt.legend()

        roc_path = os.path.join(folder, 'roc_' + lbl + '.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.clf()


def hill_climb(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d, step_z_init=0, step_r_init=0, z_lim=0, r_lim=0, min_ol_dif=0.0, alpha=False):

    step_z = step_z_init
    step_r = step_r_init

    ol_best, verts_best, _, invalid = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d)

    if invalid: return z2d, ry3d, verts_best

    # attempt to fit z/rot more properly
    while (step_z > z_lim or step_r > r_lim):

        if step_z > z_lim:

            ol_neg, verts_neg, _, invalid_neg = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d - step_z, w3d, h3d, l3d, ry3d)
            ol_pos, verts_pos, _, invalid_pos = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d + step_z, w3d, h3d, l3d, ry3d)

            invalid = ((ol_pos - ol_best) <= min_ol_dif) and ((ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_z = step_z * 0.5

            elif (ol_pos - ol_best) > min_ol_dif and ol_pos > ol_neg and not invalid_pos:
                z2d += step_z
                ol_best = ol_pos
                verts_best = verts_pos
            elif (ol_neg - ol_best) > min_ol_dif and not invalid_neg:
                z2d -= step_z
                ol_best = ol_neg
                verts_best = verts_neg
            else:
                step_z = step_z * 0.5

        if step_r > r_lim:

            ol_neg, verts_neg, _, invalid_neg = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d - step_r)
            ol_pos, verts_pos, _, invalid_pos = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d + step_r)

            invalid = ((ol_pos - ol_best) <= min_ol_dif) and ((ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_r = step_r * 0.5

            elif (ol_pos - ol_best) > min_ol_dif and ol_pos > ol_neg and not invalid_pos:
                ry3d += step_r
                ol_best = ol_pos
                verts_best = verts_pos
            elif (ol_neg - ol_best) > min_ol_dif and not invalid_neg:
                ry3d -= step_r
                ol_best = ol_neg
                verts_best = verts_neg
            else:
                step_r = step_r * 0.5

    while ry3d > math.pi: ry3d -= math.pi * 2
    while ry3d < (-math.pi): ry3d += math.pi * 2

    return z2d, ry3d, verts_best


def test_projection(p2, p2_inv, box_2d, cx, cy, z, w3d, h3d, l3d, rotY):
    """
    Tests the consistency of a 3D projection compared to a 2D box
    """

    x = box_2d[0]
    y = box_2d[1]
    x2 = x + box_2d[2] - 1
    y2 = y + box_2d[3] - 1

    coord3d = p2_inv.dot(np.array([cx * z, cy * z, z, 1]))

    cx3d = coord3d[0]
    cy3d = coord3d[1]
    cz3d = coord3d[2]

    # put back on ground first
    #cy3d += h3d/2

    # re-compute the 2D box using 3D (finally, avoids clipped boxes)
    verts3d, corners_3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

    invalid = np.any(corners_3d[2, :] <= 0)

    x_new = min(verts3d[:, 0])
    y_new = min(verts3d[:, 1])
    x2_new = max(verts3d[:, 0])
    y2_new = max(verts3d[:, 1])

    b1 = np.array([x, y, x2, y2])[np.newaxis, :]
    b2 = np.array([x_new, y_new, x2_new, y2_new])[np.newaxis, :]

    ol = iou(b1, b2)[0][0]
    #ol = -(np.abs(x - x_new) + np.abs(y - y_new) + np.abs(x2 - x2_new) + np.abs(y2 - y2_new))

    return ol, verts3d, b2, invalid