import torch.nn as nn
from torchvision import models
from lib.rpn_util import *
import torch
from lib.nms.gpu_nms import gpu_nms
from collections import OrderedDict
import torch.utils.checkpoint as cp
from torch import Tensor

def dilate_layer(layer, val):

    layer.dilation = val
    layer.padding = val


class RPN(nn.Module):


    def __init__(self, phase, base, conf):
        super(RPN, self).__init__()

        self.tracks = None

        self.base = base

        del self.base.transition3.pool

        # dilate
        dilate_layer(self.base.denseblock4.denselayer1.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer2.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer3.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer4.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer5.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer6.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer7.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer8.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer9.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer10.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer11.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer12.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer13.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer14.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer15.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer16.conv2, 2)

        # settings
        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]
        self.progressive = conf.progressive
        self.video_count = conf.video_count

        self.prop_feats = nn.Sequential(
            nn.Conv2d(self.base[-1].num_features, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pose_feats = nn.Sequential(
            nn.Conv2d(1024 * 2, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.lambda_o = 0.2
        self.k_p = 0.75
        self.k_m = 0.05

        self.Q_cov = nn.Parameter(1 * torch.ones([9]).type(torch.cuda.FloatTensor))
        self.R_cov = nn.Parameter(0.2 * torch.ones([8]).type(torch.cuda.FloatTensor))

        # pose predict
        self.pose = nn.Conv2d(self.pose_feats[-2].out_channels, 6, 1)

        # pixel-wise conf
        self.conf = nn.Conv2d(self.pose_feats[-2].out_channels, 1, 1)

        # outputs
        self.cls = nn.Conv2d(self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1, )

        # bbox 2d
        self.bbox_x = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.bbox_alpha = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_axis = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_head = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.bbox_un = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.feat_stride = conf.feat_stride
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride, convert_tensor=True)
        self.rois = self.rois.type(torch.cuda.FloatTensor)
        self.anchors = conf.anchors

        self.feat_size = [0, 0]

        self.bbox_means = conf.bbox_means
        self.bbox_stds = conf.bbox_stds

        self.pose_means = conf.pose_means
        self.pose_stds = conf.pose_stds

        self.image_means = conf.image_means
        self.image_stds = conf.image_stds

        self.nms_thres = conf.nms_thres
        self.best_thresh = conf.best_thresh

        self.score_thres = conf.score_thres
        self.forecast = 0 if 'forecast' not in conf else conf.forecast

        self.H = self.make_H_matrix()

        self.torch_bool = hasattr(torch, 'bool')


    def pose_forward(self, base):

        # pose features
        pose_feats = self.pose_feats(base)

        # weightedly aggregate pose
        pose_h = pose_feats.size(2)
        pose_w = pose_feats.size(3)

        # pose predictions and confidences
        pose = self.pose(pose_feats)
        conf = self.conf(pose_feats)

        # make confidences sum to one
        conf = conf.view([base.shape[0], -1, 1, 1]).contiguous()
        conf = self.softmax(conf)
        conf = conf.view([base.shape[0], 1, pose_h, pose_w]).contiguous()

        # weightedly fuse spatial pose predictions
        pose = pose * conf
        pose = (pose).sum(dim=2).sum(dim=2)

        return pose


    def clean_and_denorm(self, out, p2s, p2_invs, scales):

        cls = out[0].clone()
        prob = out[1].clone()
        bbox_2d = out[2].clone()
        bbox_3d = out[3].clone()

        batch_size = cls.shape[0]

        # denorm the 2D boxes
        bbox_2d[:, :, 0] = bbox_2d[:, :, 0] * self.bbox_stds[:, 0][0] + self.bbox_means[:, 0][0]
        bbox_2d[:, :, 1] = bbox_2d[:, :, 1] * self.bbox_stds[:, 1][0] + self.bbox_means[:, 1][0]
        bbox_2d[:, :, 2] = bbox_2d[:, :, 2] * self.bbox_stds[:, 2][0] + self.bbox_means[:, 2][0]
        bbox_2d[:, :, 3] = bbox_2d[:, :, 3] * self.bbox_stds[:, 3][0] + self.bbox_means[:, 3][0]

        # denorm the 3D boxes
        bbox_x2d_raw = bbox_3d[:, :, 0] * self.bbox_stds[:, 4][0] + self.bbox_means[:, 4][0]
        bbox_y2d_raw = bbox_3d[:, :, 1] * self.bbox_stds[:, 5][0] + self.bbox_means[:, 5][0]
        bbox_z2d_raw = bbox_3d[:, :, 2] * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
        bbox_w3d_raw = bbox_3d[:, :, 3] * self.bbox_stds[:, 7][0] + self.bbox_means[:, 7][0]
        bbox_h3d_raw = bbox_3d[:, :, 4] * self.bbox_stds[:, 8][0] + self.bbox_means[:, 8][0]
        bbox_l3d_raw = bbox_3d[:, :, 5] * self.bbox_stds[:, 9][0] + self.bbox_means[:, 9][0]

        bbox_rsin_raw = bbox_3d[:, :, 6] * self.bbox_stds[:, 11][0] + self.bbox_means[:, 11][0]
        bbox_rcos_raw = bbox_3d[:, :, 7] * self.bbox_stds[:, 12][0] + self.bbox_means[:, 12][0]

        bbox_axis_raw = bbox_3d[:, :, 8]
        bbox_head_raw = bbox_3d[:, :, 9]

        bbox_un = bbox_3d[:, :, 10]

        # anchor equations 2D
        pred_ctr_x = bbox_2d[:, :, 0] * self.rois_widths + self.rois_ctr_x
        pred_ctr_y = bbox_2d[:, :, 1] * self.rois_heights + self.rois_ctr_y
        pred_w = torch.exp(bbox_2d[:, :, 2]) * self.rois_widths
        pred_h = torch.exp(bbox_2d[:, :, 3]) * self.rois_heights

        # x1, y1, x2, y2
        bbox_2d[:, :, 0] = pred_ctr_x - 0.5 * pred_w
        bbox_2d[:, :, 1] = pred_ctr_y - 0.5 * pred_h
        bbox_2d[:, :, 2] = pred_ctr_x + 0.5 * pred_w
        bbox_2d[:, :, 3] = pred_ctr_y + 0.5 * pred_h

        # anchor equations 3D
        bbox_x2d_raw = bbox_x2d_raw * self.rois_widths + self.rois_ctr_x
        bbox_y2d_raw = bbox_y2d_raw * self.rois_heights + self.rois_ctr_y
        bbox_z2d_raw = self.rois_3d[:, 4] + bbox_z2d_raw
        bbox_w3d_raw = torch.exp(bbox_w3d_raw) * self.rois_3d[:, 5]
        bbox_h3d_raw = torch.exp(bbox_h3d_raw) * self.rois_3d[:, 6]
        bbox_l3d_raw = torch.exp(bbox_l3d_raw) * self.rois_3d[:, 7]

        has_vel = bbox_3d.shape[2] == 20

        if has_vel:
            bbox_vel = bbox_3d[:, :, 19] * self.bbox_stds[:, 13][0] + self.bbox_means[:, 13][0]
            bbox_vel = self.rois_3d[:, 11] + bbox_vel
            bbox_vel = bbox_vel.clamp(min=0)

        bbox_rsin_raw = self.rois_3d[:, 9] + bbox_rsin_raw
        bbox_rcos_raw = self.rois_3d[:, 10] + bbox_rcos_raw

        bbox_axis_sin_mask = bbox_axis_raw >= 0.5
        #bbox_head_pos_mask = bbox_head_raw >= 0.5

        bbox_alp_raw = bbox_rcos_raw.clone()

        bbox_alp_raw[bbox_axis_sin_mask] = bbox_rsin_raw[bbox_axis_sin_mask]
        #bbox_alp_raw[bbox_head_pos_mask] = bbox_alp_raw[bbox_head_pos_mask] + math.pi

        boxes_batch = []
        cls_batch = []

        for bind in range(batch_size):

            p2_inv = torch.from_numpy(p2_invs[bind]).type(torch.cuda.FloatTensor)

            boxes = None
            cls_feat = None

            p2_a = p2s[bind][0, 0].item()
            p2_b = p2s[bind][0, 2].item()
            p2_c = p2s[bind][0, 3].item()
            p2_d = p2s[bind][1, 1].item()
            p2_e = p2s[bind][1, 2].item()
            p2_f = p2s[bind][1, 3].item()
            p2_h = p2s[bind][2, 3].item()

            thresh_s = self.score_thres

            fg_scores, fg_cls = prob[bind, :, 1:].max(dim=1)
            fg_cls = fg_cls + 1

            fg_mask = fg_scores >= thresh_s
            fg_inds = torch.nonzero(fg_mask)

            if fg_inds.shape[0] > 0:

                fg_inds = fg_inds.squeeze(1)

                # scale down 2D boxes
                bbox_2d.data[bind, fg_inds] = bbox_2d.data[bind, fg_inds] / scales[bind]
                #bbox_2d[bind, fg_inds] = bbox_2d[bind, fg_inds].clone().detach() / scales[bind]

                # setup 2D boxes and scores
                bbox_2d_np = bbox_2d[bind, fg_inds].detach().cpu().numpy()
                aboxes = np.hstack((bbox_2d_np, fg_scores[fg_inds].detach().cpu().numpy()[:, np.newaxis]))

                # perform NMS in non-forecasted space
                keep_inds = gpu_nms(aboxes.astype(np.float32), self.nms_thres, device_id=0)
                keep_inds = torch.from_numpy(np.array(keep_inds))

                # update mask
                fg_inds = fg_inds[keep_inds]
                fg_mask[...] = 0
                fg_mask[fg_inds] = 1

                cls_feat = cls[bind, fg_inds, :]

                bbox_2d_fg = bbox_2d[bind, fg_inds]
                scores = fg_scores[fg_inds]
                cls_fg = fg_cls[fg_inds]

                bbox_x2d_dn_fg = bbox_x2d_raw[bind, fg_inds]
                bbox_y2d_dn_fg = bbox_y2d_raw[bind, fg_inds]
                bbox_z2d_dn_fg = bbox_z2d_raw[bind, fg_inds]
                bbox_w3d_dn_fg = bbox_w3d_raw[bind, fg_inds]
                bbox_h3d_dn_fg = bbox_h3d_raw[bind, fg_inds]
                bbox_l3d_dn_fg = bbox_l3d_raw[bind, fg_inds]
                bbox_alp_dn_fg = bbox_alp_raw[bind, fg_inds]
                bbox_head_dn_fg = bbox_head_raw[bind, fg_inds]

                bbox_un_fg = bbox_un[bind, fg_inds]

                # scale x2d and y2d back down
                bbox_x2d_dn_fg = bbox_x2d_dn_fg / scales[bind]
                bbox_y2d_dn_fg = bbox_y2d_dn_fg / scales[bind]

                # project back to 3D
                z3d = bbox_z2d_dn_fg - p2_h
                x3d = ((z3d + p2_h) * bbox_x2d_dn_fg - p2_b * (z3d) - p2_c) / p2_a
                y3d = ((z3d + p2_h) * bbox_y2d_dn_fg - p2_e * (z3d) - p2_f) / p2_d

                # gather 2D coords
                #coords_2d = torch.cat(
                #    (bbox_x2d_dn_fg[np.newaxis, :] * bbox_z2d_dn_fg[np.newaxis, :],
                #     bbox_y2d_dn_fg[np.newaxis, :] * bbox_z2d_dn_fg[np.newaxis, :],
                #     bbox_z2d_dn_fg[np.newaxis, :]), dim=0)

                # pad ones for a 4x1
                #coords_2d = torch.cat((coords_2d, torch.ones([1, coords_2d.shape[1]])), dim=0)
                #coords_3d = torch.mm(p2_inv, coords_2d)

                #x3d = coords_3d[0, :]
                #y3d = coords_3d[1, :]
                #z3d = coords_3d[2, :]

                ry3d = convertAlpha2Rot(bbox_alp_dn_fg, z3d, x3d)

                # [x y, x2, y2, score, cls, x, y, z, w, h, l, theta, head, vars, vel]
                boxes = torch.cat((bbox_2d_fg, scores.unsqueeze(1), cls_fg.unsqueeze(1).type(torch.cuda.FloatTensor),
                                   x3d.unsqueeze(1), y3d.unsqueeze(1), z3d.unsqueeze(1), bbox_w3d_dn_fg.unsqueeze(1),
                                   bbox_h3d_dn_fg.unsqueeze(1), bbox_l3d_dn_fg.unsqueeze(1),
                                   ry3d.unsqueeze(1), bbox_head_dn_fg.unsqueeze(1), bbox_un_fg.unsqueeze(1)), dim=1)

                if has_vel:
                    boxes[:, 23] = bbox_vel[bind, fg_inds]


            cls_batch.append(cls_feat)
            boxes_batch.append(boxes)

        return boxes_batch, cls_batch


    def initialize_tracks(self, bbox_measure):

        tracks_batch = []

        for bind in range(len(bbox_measure)):

            boxes = bbox_measure[bind]

            if boxes is None:
                tracks_batch.append(None)
                continue

            # tracks are defined having
            # where N = # boxes
            # where V = # variables
            # A ~ N x V x V         (state transition matrix)
            # C ~ N x V x V         (covariance matrix)
            # X ~ N x V             (state variables)

            tracks = edict()
            tracks.ids = list(range(boxes.shape[0]))
            tracks.seen = boxes.shape[0]
            tracks.box2ds = boxes[:, 0:6]
            tracks.bbox_un = boxes[:, 14].detach() * boxes[:, 4].detach()
            tracks.Xs = F.pad(boxes[:, 6:13+1], pad=[0, 1])
            tracks.As = self.make_transition_matrix(boxes[:, 12], boxes[:, 13])
            tracks.Cs = self.make_covariance_matrix((self.Q_cov.clamp(min=0)[:-1]*0 + 0*self.R_cov.clamp(min=0)).unsqueeze(0).repeat([tracks.Xs.shape[0], 1]) + 1*(self.lambda_o)*(1 - tracks.bbox_un.unsqueeze(1)))

            tracks_batch.append(tracks)

        return tracks_batch


    def make_covariance_matrix(self, vars_bbox):

        num_variables = 9
        num_available = min(vars_bbox.shape[1], num_variables)
        num_boxes = vars_bbox.shape[0]

        eye_mask = torch.eye(num_available) == 1

        covariances = torch.eye(num_variables).unsqueeze(0).repeat([num_boxes, 1, 1])
        covariances[:, :num_available, :num_available][:, eye_mask] = vars_bbox[:, :num_variables]

        return covariances

    def make_transition_matrix(self, ry3d, head):

        num_variables = 9
        bbox_head_pos_mask = head >= 0.5

        ry3d_with_head = ry3d.clone()
        ry3d_with_head[bbox_head_pos_mask] = ry3d_with_head[bbox_head_pos_mask] + math.pi

        # assumes representation is the same as with Xs
        # [x, y, z, w, h, l, theta, head, vel]
        As = torch.eye(num_variables).unsqueeze(0).repeat([ry3d_with_head.shape[0], 1, 1])

        As[:, 0, 8] = torch.cos(ry3d_with_head)     # velocity transition in X
        As[:, 2, 8] = -torch.sin(ry3d_with_head)    # velocity transition in Z

        return As.detach()


    def make_H_matrix(self):

        num_variables = 9

        # assumes representation is the same as with Xs
        # [x, y, z, w, h, l, theta, head, vel]

        # we ignore vel to convert state to measurements
        H = torch.eye(num_variables).unsqueeze(0)[:, :8, :]

        return H.detach()


    def project_ego(self, tracks_batch, pose_batch, p2s):

        pose_means = torch.from_numpy(self.pose_means).type(torch.cuda.FloatTensor)
        pose_stds = torch.from_numpy(self.pose_stds).type(torch.cuda.FloatTensor)

        for bind in range(len(tracks_batch)):

            tracks = tracks_batch[bind]
            pose_dn = pose_batch[bind].clone().detach() * pose_stds[0] + pose_means[0]

            if len(pose_dn.shape) > 1: pose_dn = pose_dn[0]

            if tracks is None: continue

            pose_full = torch.eye(4)
            pose_r = euler2mat(pose_dn[3], pose_dn[4], pose_dn[5])
            pose_full[:3, :3] = torch.from_numpy(pose_r).type(torch.cuda.FloatTensor)
            pose_full[:3, 3] = pose_dn[:3]

            # project x, y, z using pose
            tracks.Xs[:, :3] = pose_full.mm(F.pad(tracks.Xs[:, :3].clone(), [0, 1], mode='constant', value=1).t()).t()[:, :3]

            # project rotation
            tracks.Xs[:, 6] = tracks.Xs[:, 6].clone() + pose_dn[4].item()

            if 'history' in tracks:

                for idind, id in enumerate(tracks.ids):

                    # already has history?
                    if str(id) in tracks.history:

                        # move history with the camera!
                        tracks.history[str(id)][:, :3] = pose_full.mm(F.pad(tracks.history[str(id)][:, :3], [0, 1], mode='constant', value=1).t()).t()[:, :3]

                        # move rotation as well
                        tracks.history[str(id)][:, 6] = tracks.history[str(id)][:, 6] + pose_dn[4].item()

            if tracks.Xs.shape[0] > 0:
                # update 2D box projection
                tracks.box2ds[:, :4], ign = get_2D_from_3D(p2s[bind], tracks.Xs[:, 0].detach(),
                                                           tracks.Xs[:, 1].detach(), tracks.Xs[:, 2].detach(),
                                                           tracks.Xs[:, 3].detach(), tracks.Xs[:, 4].detach(),
                                                           tracks.Xs[:, 5].detach(), tracks.Xs[:, 6].detach())

        return tracks_batch


    def associate_tracks(self, tracks_batch, bbox_measure_batch):

        match_thres = 0.5

        batch_size = len(tracks_batch)

        associate_tr_batch = []
        associate_me_batch = []

        for bind in range(batch_size):

            tracks_found = []
            detect_found = []

            tracks = tracks_batch[bind]
            bbox_measure = bbox_measure_batch[bind]

            if tracks is None or bbox_measure is None or bbox_measure.shape[0] == 0:
                associate_tr_batch.append(tracks_found)
                associate_me_batch.append(detect_found)
                continue

            boxes_tr = tracks.box2ds[:, :4].detach().cpu().numpy()
            boxes_me = bbox_measure[:, :4].detach().cpu().numpy()

            ols = iou(boxes_tr, boxes_me)
            dist = np.sqrt(((bbox_measure[np.newaxis, :, 6:6 + 3] - tracks.Xs[:, np.newaxis, :3]).detach().cpu().numpy() ** 2).sum(axis=2))

            # first match by distance
            while (dist <= match_thres).any():

                # per track
                # best_ss_val = ols.max(1)
                # best_ss_ind = ols.argmax(1)

                best_ss_val = dist.min(1)
                best_ss_ind = dist.argmin(1)

                # per detection
                # best_tr_val = best_ss_val.max()
                # best_tr_ind = best_ss_val.argmax()

                best_tr_val = best_ss_val.min()
                best_tr_ind = best_ss_val.argmin()

                # if best_tr_val >= match_thres:
                if best_tr_val <= match_thres:
                    # now "clear" both the track and detection
                    ols[best_tr_ind, :] = 0
                    ols[:, best_ss_ind[best_tr_ind]] = 0

                    dist[best_tr_ind, :] = np.inf
                    dist[:, best_ss_ind[best_tr_ind]] = np.inf

                    tracks_found.append(best_tr_ind)
                    detect_found.append(best_ss_ind[best_tr_ind])

            # now match by IoU
            match_thres = self.best_thresh
            while (ols >= match_thres).any():

                # per track
                best_ss_val = ols.max(1)
                best_ss_ind = ols.argmax(1)

                # per detection
                best_tr_val = best_ss_val.max()
                best_tr_ind = best_ss_val.argmax()

                if best_tr_val >= match_thres:
                    # now "clear" both the track and detection
                    ols[best_tr_ind, :] = 0
                    ols[:, best_ss_ind[best_tr_ind]] = 0

                    dist[best_tr_ind, :] = np.inf
                    dist[:, best_ss_ind[best_tr_ind]] = np.inf

                    tracks_found.append(best_tr_ind)
                    detect_found.append(best_ss_ind[best_tr_ind])

            associate_tr_batch.append(tracks_found)
            associate_me_batch.append(detect_found)

        return associate_tr_batch, associate_me_batch


    def forecast_tracks(self, tracks_batch, p2s):

        batch_size = len(tracks_batch)

        for bind in range(batch_size):

            tracks = tracks_batch[bind]

            if tracks is None:
                continue

            tracks.Xs_pre_pre = tracks.Xs.clone()

            if not ('history' in tracks):

                tracks.history = dict()

                for idind, id in enumerate(tracks.ids):
                    new_instance = torch.cat((
                        tracks.Xs[idind].clone(),
                        tracks.bbox_un[idind].clone().unsqueeze(0),
                        tracks.box2ds[idind].clone())).unsqueeze(0)

                    tracks.history[str(id)] = new_instance

            else:

                for idind, id in enumerate(tracks.ids):

                    new_instance = torch.cat((
                        tracks.Xs[idind].clone(),
                        tracks.bbox_un[idind].clone().unsqueeze(0),
                        tracks.box2ds[idind].clone())).unsqueeze(0)

                    # already has history?
                    if str(id) in tracks.history:

                        tracks.history[str(id)] = torch.cat((tracks.history[str(id)], new_instance), dim=0)

                    # needs init history?
                    else:

                        tracks.history[str(id)] = new_instance

            # Kalman forecast state
            tracks.Xs = tracks.As.bmm(tracks.Xs.unsqueeze(2)).squeeze(2)

            # Kalman forecast covariance
            tracks.Cs = tracks.As.bmm(tracks.Cs).bmm(tracks.As.transpose(1, 2))

            if tracks.Xs.shape[0] > 0:

                # update 2D box projection
                tracks.box2ds[:, :4], ign = get_2D_from_3D(p2s[bind], tracks.Xs[:, 0], tracks.Xs[:, 1], tracks.Xs[:, 2], tracks.Xs[:, 3], tracks.Xs[:, 4], tracks.Xs[:, 5], tracks.Xs[:, 6])

        return tracks_batch


    def update_tracks(self, tracks_batch, bbox_measure_batch, associate_tr_batch, associate_me_batch):

        num_variables = 9
        batch_size = len(tracks_batch)

        for bind in range(batch_size):

            tracks = tracks_batch[bind]
            associate_tr = associate_tr_batch[bind]
            associate_me = associate_me_batch[bind]
            bbox_measure = bbox_measure_batch[bind]

            if tracks is None or len(tracks) == 0 or len(associate_tr) == 0 or bbox_measure is None:
                tracks_batch[bind] = None
                continue

            associate_tr = torch.tensor(associate_tr).type(torch.cuda.LongTensor)
            associate_me = torch.tensor(associate_me).type(torch.cuda.LongTensor)

            # covariance predicted and measured
            C_pre = tracks.Cs[associate_tr, :]

            bbox_un = bbox_measure[associate_me, 14] * bbox_measure[associate_me, 4]
            bbox_un_pre = tracks.bbox_un[associate_tr].clone()

            R_tmp = 0*self.R_cov.clamp(min=0).unsqueeze(0).repeat([C_pre.shape[0], 1]) + 1*(self.lambda_o)*(1 - bbox_un.unsqueeze(1))
            C_mea = self.make_covariance_matrix(R_tmp)[:, :8, :8]

            X_pre = tracks.Xs[associate_tr, :]
            X_mea = bbox_measure[associate_me, 6:13 + 1]

            H = self.H.repeat([C_pre.shape[0], 1, 1])

            C_pre += self.make_covariance_matrix(1*(1 - bbox_un_pre.unsqueeze(1)) + 0*self.Q_cov.clamp(min=0).unsqueeze(0).repeat([X_pre.shape[0], 1]))

            # kalman gain
            K = C_pre.bmm(H.transpose(1, 2).bmm((H.bmm(C_pre.bmm(H.transpose(1, 2))) + C_mea).inverse()))

            # final state
            X_final = X_pre + K.bmm((X_mea.unsqueeze(2) - H.bmm(X_pre.unsqueeze(2)))).squeeze(2)

            tracks.bbox_un[associate_tr] = bbox_un * 0.5 + bbox_un_pre * 0.5

            # reduce the uncertainty
            C_final = (torch.eye(num_variables).unsqueeze(0) - K.bmm(H)).bmm(C_pre)

            tracks.Xs[associate_tr, :] = X_final
            tracks.Cs[associate_tr, :] = C_final

            # reminder that format is
            # [x, y, z, w, h, l, theta, head, vel]

            # must update
            tracks.box2ds[associate_tr, :] = bbox_measure[associate_me, 0:6]

            # must update state transition matrix
            tracks.As[associate_tr, :] = self.make_transition_matrix(X_final[:, 6], X_final[:, 7])

            # penalty for missing tracks
            missing_tracks = torch.ones(tracks.Xs[:, 2].shape).type(torch.BoolTensor)
            missing_tracks[associate_tr] = False
            tracks.bbox_un[missing_tracks] = tracks.bbox_un[missing_tracks] * self.k_p

            # remove unhealthy tracks
            valid_tracks = (tracks.Xs[:, 2] > 1) & (tracks.bbox_un > self.k_m)

            tracks.bbox_un = tracks.bbox_un[valid_tracks]
            tracks.Xs = tracks.Xs[valid_tracks, :]
            tracks.Cs = tracks.Cs[valid_tracks, :]
            tracks.box2ds = tracks.box2ds[valid_tracks, :]
            tracks.As = tracks.As[valid_tracks, :]
            tracks.ids = np.array(tracks.ids)[valid_tracks.cpu().numpy()].tolist()

        return tracks_batch


    def add_unused_measurements(self, tracks_batch, bbox_measure_batch, associate_me_batch):

        batch_size = len(tracks_batch)

        for bind in range(batch_size):

            tracks = tracks_batch[bind]

            associate_me = associate_me_batch[bind]
            bbox_measure = bbox_measure_batch[bind]

            if bbox_measure is None:
                continue

            new_tracks = []

            for boxind in range(bbox_measure.shape[0]):
                if boxind not in associate_me:
                    new_tracks.append(boxind)

            if len(new_tracks) > 0:

                new_tracks = torch.tensor(new_tracks).type(torch.cuda.LongTensor)

                box2ds = bbox_measure[new_tracks, 0:6]
                Xs = F.pad(bbox_measure[new_tracks, 6:13+1], pad=[0, 1])
                As = self.make_transition_matrix(bbox_measure[new_tracks, 12], bbox_measure[new_tracks, 13])
                bbox_un = bbox_measure[new_tracks, 14] * bbox_measure[new_tracks, 4]
                Cs = self.make_covariance_matrix(1*(self.lambda_o)*(1 - bbox_un.unsqueeze(1)) + 0*self.R_cov.clamp(min=0).unsqueeze(0).repeat([Xs.shape[0], 1]))


                if tracks is None:

                    tracks = edict()
                    tracks.Xs = Xs
                    tracks.As = As
                    tracks.Cs = Cs
                    tracks.box2ds = box2ds
                    tracks_batch[bind] = tracks
                    tracks.bbox_un = bbox_un
                    tracks.ids = list(range(Xs.shape[0]))
                    tracks.seen = Xs.shape[0]

                else:
                    tracks.Xs = torch.cat((tracks.Xs, Xs), dim=0)
                    tracks.As = torch.cat((tracks.As, As), dim=0)
                    tracks.Cs = torch.cat((tracks.Cs, Cs), dim=0)
                    tracks.bbox_un = torch.cat((tracks.bbox_un, bbox_un), dim=0)
                    tracks.box2ds = torch.cat((tracks.box2ds, box2ds), dim=0)
                    tracks.ids += list(range(tracks.seen, tracks.seen+Xs.shape[0]))
                    tracks.seen += Xs.shape[0]



        return tracks_batch

    def clone_tracks(self, tracks_batch):

        tracks_batch_new = []

        batch_size = len(tracks_batch)

        for bind in range(batch_size):

            tracks = tracks_batch[bind]

            if tracks is None or len(tracks) == 0:
                tracks_batch_new.append(None)
                continue

            tracks_new = edict()
            tracks_new.Xs = tracks.Xs.clone()
            tracks_new.As = tracks.As.clone()
            tracks_new.Cs = tracks.Cs.clone()
            tracks_new.box2ds = tracks.box2ds.clone()
            tracks_new.bbox_un = tracks.bbox_un.clone()
            tracks_new.ids = copy.copy(tracks.ids)
            tracks_new.seen = copy.copy(tracks.seen)

            if 'history' in tracks: tracks_new.history = tracks.history
            if 'C_pre' in tracks: tracks_new.C_pre = tracks.C_pre.clone()
            if 'X_pre' in tracks: tracks_new.X_pre = tracks.X_pre.clone()
            if 'X_mea' in tracks: tracks_new.X_mea = tracks.X_mea.clone()
            if 'box2ds_mea' in tracks: tracks_new.box2ds_mea = tracks.box2ds_mea.clone()

            tracks_batch_new.append(tracks_new)

        return tracks_batch_new

    def forward_boxes(self, bbox_measures, p2s, p2_invs, scales):

        batch_size = len(p2s)

        num_images = self.video_count

        measures_all = bbox_measures[:, 0]
        poses_all = bbox_measures[:, 1]

        tracks = None
        tr_shots = []
        si_shots = []
        poses = []

        test_start = time()

        for imind in range(num_images):

            bbox_measure = measures_all[:, imind]

            if tracks is None:
                tracks = self.initialize_tracks(bbox_measure)
            else:

                if imind >= (num_images - self.forecast):
                    pose_tmp = poses_all[:, (num_images - self.forecast) - 1]
                    poses.append(pose_tmp)

                    # project all tracks using ego-motion
                    tracks = self.project_ego(tracks, pose_tmp, p2s)

                    # forecast all available tracks, to align with measurement
                    tracks = self.forecast_tracks(tracks, p2s)

                else:
                    pose_tmp = poses_all[:, imind]
                    poses.append(pose_tmp)

                    # project all tracks using ego-motion
                    tracks = self.project_ego(tracks, pose_tmp, p2s)

                    # forecast all available tracks, to align with measurement
                    tracks = self.forecast_tracks(tracks, p2s)

                    # associate tracks with new single-shots
                    associate_tr, associate_me = self.associate_tracks(tracks, bbox_measure)

                    # KALMAN update single-shots with predicted variances
                    tracks = self.update_tracks(tracks, bbox_measure, associate_tr, associate_me)

                    # initialize any new tracks
                    tracks = self.add_unused_measurements(tracks, bbox_measure, associate_me)

            # store some features
            tr_shots.append(self.clone_tracks(tracks))
            si_shots.append(bbox_measure)

            time_str, dt = compute_eta(test_start, imind + 1, num_images)
            logging.info('processed boxes {}/{}, dt: {:0.3f}, eta: {}'.format(imind + 1, num_images, dt, time_str))

        return si_shots, tr_shots, poses

    def forward_single(self, x, p2s, p2_invs, scales, extract_only=False):

        p2s = torch.from_numpy(p2s).cuda().type(torch.cuda.FloatTensor)

        batch_size = x.size(0)

        num_images = int(x.shape[1] / 3)

        # share most computation
        # stack images in batch dimensions
        # index by x[bind*num_images + imind, 0, 40:45, 100])
        x = x.view(batch_size * num_images, 3, x.shape[2], x.shape[3]).contiguous()

        # backbone features
        base = self.base(x)

        # ---------- 3D Object Detection of Single-Shot  ----------
        prop_feats = self.prop_feats(base)

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)

        bbox_un = self.sigmoid(self.bbox_un(prop_feats))

        bbox_alpha = self.bbox_alpha(prop_feats)
        bbox_axis = self.sigmoid(self.bbox_axis(prop_feats))
        bbox_head = self.sigmoid(self.bbox_head(prop_feats))

        feat_h = cls.size(2)
        feat_w = cls.size(3)

        # update rois
        if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
            self.feat_size = [feat_h, feat_w]
            self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
            self.rois = self.rois.type(torch.cuda.FloatTensor)

            # more computations
            # tracker_raw = self.rois[:, 4].cpu().detach().numpy().astype(np.int64)
            self.rois_3d = self.anchors[self.rois[:, 4].type(torch.LongTensor), :]
            self.rois_3d = torch.tensor(self.rois_3d, requires_grad=False).type(torch.cuda.FloatTensor)

            # compute 3d transform
            self.rois_widths = self.rois[:, 2] - self.rois[:, 0] + 1.0
            self.rois_heights = self.rois[:, 3] - self.rois[:, 1] + 1.0
            self.rois_ctr_x = self.rois[:, 0] + 0.5 * (self.rois_widths)
            self.rois_ctr_y = self.rois[:, 1] + 0.5 * (self.rois_heights)
            self.rois_3d_cen = torch.cat((self.rois_ctr_x.unsqueeze(1), self.rois_ctr_y.unsqueeze(1)), dim=1)

        out_sp = torch.cat((cls, bbox_x, bbox_y, bbox_w, bbox_h, bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d,
                            bbox_l3d, bbox_alpha, bbox_alpha, bbox_axis, bbox_head), dim=1).clone()

        # reshape for cross entropy
        cls = cls.view(batch_size*num_images, self.num_classes, feat_h * self.num_anchors, feat_w)

        # score probabilities
        prob = self.softmax(cls)

        # reshape for consistency
        bbox_x = flatten_tensor(bbox_x.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_y = flatten_tensor(bbox_y.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d = flatten_tensor(bbox_y3d.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))

        bbox_alpha = flatten_tensor(bbox_alpha.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_axis = flatten_tensor(bbox_axis.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_head = flatten_tensor(bbox_head.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))

        bbox_un = flatten_tensor(bbox_un.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))

        # bundle
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)
        bbox_3d = torch.cat((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_alpha, bbox_alpha.clone(),
                             bbox_axis, bbox_head, bbox_un), dim=2)

        feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        tracks = None
        base_prev = None
        tr_shots = []
        si_shots = []
        poses = []

        base_tmp = base[0, :, :, :].contiguous()[np.newaxis, :, :]

        cls_tmp = cls[0, :, :].contiguous()[np.newaxis, :, :]
        prob_tmp = prob[0, :, :].contiguous()[np.newaxis, :, :]
        bbox_2d_tmp = bbox_2d[0, :, :].contiguous()[np.newaxis, :, :]
        bbox_3d_tmp = bbox_3d[0, :, :].contiguous()[np.newaxis, :, :]

        out_tmp = [cls_tmp, prob_tmp, bbox_2d_tmp, bbox_3d_tmp]
        bbox_measure, _ = self.clean_and_denorm(out_tmp, p2s, p2_invs, scales)
        if self.tracks is None:
            self.tracks = self.initialize_tracks(bbox_measure)
        else:

            pose_tmp = self.pose_forward(torch.cat((self.base_prev, base_tmp), dim=1))
            poses.append(pose_tmp)

            # project all tracks using ego-motion
            self.tracks = self.project_ego(self.tracks, pose_tmp, p2s)

            # forecast all available tracks, to align with measurement
            self.tracks = self.forecast_tracks(self.tracks, p2s)

            # associate tracks with new single-shots
            associate_tr, associate_me = self.associate_tracks(self.tracks, bbox_measure)

            # KALMAN update single-shots with predicted variances
            self.tracks = self.update_tracks(self.tracks, bbox_measure, associate_tr, associate_me)

            # initialize any new tracks
            self.tracks = self.add_unused_measurements(self.tracks, bbox_measure, associate_me)

        # store some features
        tr_shots.append(self.clone_tracks(self.tracks))
        si_shots.append(bbox_measure)
        self.base_prev = base_tmp

        return si_shots, tr_shots, poses

    def forward(self, x, p2s, p2_invs, scales, extract_only=False):

        p2s = torch.from_numpy(p2s).cuda().type(torch.cuda.FloatTensor)

        batch_size = x.size(0)

        num_images = int(x.shape[1] / 3)

        # share most computation
        # stack images in batch dimensions
        # index by x[bind*num_images + imind, 0, 40:45, 100])
        x = x.view(batch_size * num_images, 3, x.shape[2], x.shape[3]).contiguous()

        # backbone features
        base = self.base(x)

        # ---------- 3D Object Detection of Single-Shot  ----------
        prop_feats = self.prop_feats(base)

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)

        bbox_un = self.sigmoid(self.bbox_un(prop_feats))

        bbox_alpha = self.bbox_alpha(prop_feats)
        bbox_axis = self.sigmoid(self.bbox_axis(prop_feats))
        bbox_head = self.sigmoid(self.bbox_head(prop_feats))

        feat_h = cls.size(2)
        feat_w = cls.size(3)

        # update rois
        if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
            self.feat_size = [feat_h, feat_w]
            self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
            self.rois = self.rois.type(torch.cuda.FloatTensor)

            # more computations
            # tracker_raw = self.rois[:, 4].cpu().detach().numpy().astype(np.int64)
            self.rois_3d = self.anchors[self.rois[:, 4].type(torch.LongTensor), :]
            self.rois_3d = torch.tensor(self.rois_3d, requires_grad=False).type(torch.cuda.FloatTensor)

            # compute 3d transform
            self.rois_widths = self.rois[:, 2] - self.rois[:, 0] + 1.0
            self.rois_heights = self.rois[:, 3] - self.rois[:, 1] + 1.0
            self.rois_ctr_x = self.rois[:, 0] + 0.5 * (self.rois_widths)
            self.rois_ctr_y = self.rois[:, 1] + 0.5 * (self.rois_heights)
            self.rois_3d_cen = torch.cat((self.rois_ctr_x.unsqueeze(1), self.rois_ctr_y.unsqueeze(1)), dim=1)

        out_sp = torch.cat((cls, bbox_x, bbox_y, bbox_w, bbox_h, bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d,
                            bbox_l3d, bbox_alpha, bbox_alpha, bbox_axis, bbox_head), dim=1).clone()

        # reshape for cross entropy
        cls = cls.view(batch_size*num_images, self.num_classes, feat_h * self.num_anchors, feat_w)

        # score probabilities
        prob = self.softmax(cls)

        # reshape for consistency
        bbox_x = flatten_tensor(bbox_x.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_y = flatten_tensor(bbox_y.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d = flatten_tensor(bbox_y3d.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))

        bbox_alpha = flatten_tensor(bbox_alpha.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_axis = flatten_tensor(bbox_axis.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))
        bbox_head = flatten_tensor(bbox_head.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))

        bbox_un = flatten_tensor(bbox_un.view(batch_size * num_images, 1, feat_h * self.num_anchors, feat_w))

        # bundle
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)
        bbox_3d = torch.cat((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_alpha, bbox_alpha.clone(),
                             bbox_axis, bbox_head, bbox_un), dim=2)

        feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        tracks = None
        base_prev = None
        tr_shots = []
        si_shots = []
        poses = []

        for imind in reversed(range(num_images)):

            base_tmp = base[imind::num_images, :, :, :].contiguous()

            cls_tmp = cls[imind::num_images, :, :].contiguous()
            prob_tmp = prob[imind::num_images, :, :].contiguous()
            bbox_2d_tmp = bbox_2d[imind::num_images, :, :].contiguous()
            bbox_3d_tmp = bbox_3d[imind::num_images, :, :].contiguous()

            out_tmp = [cls_tmp, prob_tmp, bbox_2d_tmp, bbox_3d_tmp]
            bbox_measure, _ = self.clean_and_denorm(out_tmp, p2s, p2_invs, scales)

            if tracks is None:
                tracks = self.initialize_tracks(bbox_measure)
            else:

                pose_tmp = self.pose_forward(torch.cat((base_prev, base_tmp), dim=1))
                poses.append(pose_tmp)

                # project all tracks using ego-motion
                tracks = self.project_ego(tracks, pose_tmp, p2s)

                # forecast all available tracks, to align with measurement
                tracks = self.forecast_tracks(tracks, p2s)

                # associate tracks with new single-shots
                associate_tr, associate_me = self.associate_tracks(tracks, bbox_measure)

                # KALMAN update single-shots with predicted variances
                tracks = self.update_tracks(tracks, bbox_measure, associate_tr, associate_me)

                # initialize any new tracks
                tracks = self.add_unused_measurements(tracks, bbox_measure, associate_me)

            # store some features
            tr_shots.append(self.clone_tracks(tracks))
            si_shots.append(bbox_measure)
            base_prev = base_tmp

        return si_shots, tr_shots, poses


def build(conf, phase):

    train = phase.lower() == 'train'

    densenet121 = models.densenet121(pretrained=train)

    # make network
    rpn_net = RPN(phase, densenet121.features, conf)

    if train: rpn_net.train()
    else: rpn_net.eval()

    return rpn_net

