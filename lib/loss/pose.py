import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import sys

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.rpn_util import *


class Pose_loss(nn.Module):

    def __init__(self, conf, verbose=True):
        super(Pose_loss, self).__init__()

        self.pose_lambda_t = conf.pose_lambda_t
        self.pose_lambda_r = conf.pose_lambda_r
        self.pose_means = conf.pose_means
        self.pose_stds = conf.pose_stds

        self.verbose = verbose


    def forward(self, poses, imobjs, pose_code='ego_10'):

        stats = []
        loss = torch.tensor(0).type(torch.cuda.FloatTensor)

        batch_size = poses.shape[0]

        poses_dn = torch.zeros(poses.shape)
        poses_tar = torch.zeros(poses.shape)
        poses_tar_dn = torch.zeros(poses.shape)

        for bind, imobj in enumerate(imobjs):

            pose_dx, pose_dy, pose_dz, pose_rx, pose_ry, pose_rz = imobj[pose_code]

            poses_tar_dn[bind, 0] = pose_dx
            poses_tar_dn[bind, 1] = pose_dy
            poses_tar_dn[bind, 2] = pose_dz
            poses_tar_dn[bind, 3] = pose_rx
            poses_tar_dn[bind, 4] = pose_ry
            poses_tar_dn[bind, 5] = pose_rz

            pose_dx = (pose_dx - self.pose_means[0, 0]) / self.pose_stds[0, 0]
            pose_dy = (pose_dy - self.pose_means[0, 1]) / self.pose_stds[0, 1]
            pose_dz = (pose_dz - self.pose_means[0, 2]) / self.pose_stds[0, 2]
            pose_rx = (pose_rx - self.pose_means[0, 3]) / self.pose_stds[0, 3]
            pose_ry = (pose_ry - self.pose_means[0, 4]) / self.pose_stds[0, 4]
            pose_rz = (pose_rz - self.pose_means[0, 5]) / self.pose_stds[0, 5]

            poses_tar[bind, 0] = pose_dx
            poses_tar[bind, 1] = pose_dy
            poses_tar[bind, 2] = pose_dz
            poses_tar[bind, 3] = pose_rx
            poses_tar[bind, 4] = pose_ry
            poses_tar[bind, 5] = pose_rz

            poses_dn[bind, 0] = (poses[bind, 0]) * self.pose_stds[0, 0] + self.pose_means[0, 0]
            poses_dn[bind, 1] = (poses[bind, 1]) * self.pose_stds[0, 1] + self.pose_means[0, 1]
            poses_dn[bind, 2] = (poses[bind, 2]) * self.pose_stds[0, 2] + self.pose_means[0, 2]
            poses_dn[bind, 3] = (poses[bind, 3]) * self.pose_stds[0, 3] + self.pose_means[0, 3]
            poses_dn[bind, 4] = (poses[bind, 4]) * self.pose_stds[0, 4] + self.pose_means[0, 4]
            poses_dn[bind, 5] = (poses[bind, 5]) * self.pose_stds[0, 5] + self.pose_means[0, 5]

        #loss_poses_t = F.smooth_l1_loss(poses[:, 0:3], poses_tar[:, 0:3], reduction='none').mean()
        #loss_poses_r = F.smooth_l1_loss(poses[:, 3:6], poses_tar[:, 3:6], reduction='none').mean()
        loss_poses_t = torch.abs(poses_dn[:, 0:3] - poses_tar_dn[:, 0:3]).mean() * self.pose_lambda_t
        loss_poses_r = torch.abs(poses_dn[:, 3:6] - poses_tar_dn[:, 3:6]).mean() * self.pose_lambda_r
        #loss_poses_t = torch.abs(poses[:, 0:3] - poses_tar[:, 0:3]).mean()
        #loss_poses_r = torch.abs(poses[:, 3:6] - poses_tar[:, 3:6]).mean()

        loss += (loss_poses_r + loss_poses_t)

        if self.pose_lambda_r > 0:
            stats.append({'name': 'pose_r', 'val': loss_poses_r, 'format': '{:0.4f}', 'group': 'loss'})
            pose_abs_rx = np.mean(torch.abs(poses_dn[:, 3] - poses_tar_dn[:, 3]).detach().cpu().numpy())
            pose_abs_ry = np.mean(torch.abs(poses_dn[:, 4] - poses_tar_dn[:, 4]).detach().cpu().numpy())
            pose_abs_rz = np.mean(torch.abs(poses_dn[:, 5] - poses_tar_dn[:, 5]).detach().cpu().numpy())
            if self.verbose: stats.append({'name': 'pose_rx', 'val': pose_abs_rx, 'format': '{:0.4f}', 'group': 'misc'})
            stats.append({'name': 'pose_ry', 'val': pose_abs_ry, 'format': '{:0.4f}', 'group': 'misc'})
            if self.verbose: stats.append({'name': 'pose_rz', 'val': pose_abs_rz, 'format': '{:0.4f}', 'group': 'misc'})

        if self.pose_lambda_t > 0:
            stats.append({'name': 'pose_t', 'val': loss_poses_t, 'format': '{:0.4f}', 'group': 'loss'})
            pose_abs_x = np.mean(torch.abs(poses_dn[:, 0] - poses_tar_dn[:, 0]).detach().cpu().numpy())
            pose_abs_y = np.mean(torch.abs(poses_dn[:, 1] - poses_tar_dn[:, 1]).detach().cpu().numpy())
            pose_abs_z = np.mean(torch.abs(poses_dn[:, 2] - poses_tar_dn[:, 2]).detach().cpu().numpy())
            if self.verbose: stats.append({'name': 'pose_x', 'val': pose_abs_x, 'format': '{:0.4f}', 'group': 'misc'})
            if self.verbose: stats.append({'name': 'pose_y', 'val': pose_abs_y, 'format': '{:0.4f}', 'group': 'misc'})
            stats.append({'name': 'pose_z', 'val': pose_abs_z, 'format': '{:0.4f}', 'group': 'misc'})

        return loss, stats
