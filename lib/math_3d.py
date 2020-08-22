"""
This file is meant to contain 3D geometry math functions
"""

import os
import sys
from glob import glob
from time import time
import matplotlib.pyplot as plt
import numpy as np
import importlib
import pickle
import logging
import datetime
import pprint
import shutil
import math
import torch
import copy
import cv2
#from scipy.spatial.transform import Rotation as scipy_R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import random
from itertools import combinations


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def project_3d_point(p2, x3d, y3d, z3d):

    coord2d = p2.dot(np.array([x3d, y3d, z3d, 1]))
    coord2d[:2] /= coord2d[2]

    return coord2d[0], coord2d[1], coord2d[2]


def backproject_3d_point(p2_inv, x2d, y2d, z2d):

    coord3d = p2_inv.dot(np.array([x2d * z2d, y2d * z2d, z2d, 1]))

    return coord3d[0], coord3d[1], coord3d[2]


def norm(vec):
    return np.linalg.norm(vec)


def get_2D_from_3D(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY):

    if type(cx3d) == torch.Tensor:
        cx3d = cx3d.detach()
        cy3d = cy3d.detach()
        cz3d = cz3d.detach()
        w3d = w3d.detach()
        h3d = h3d.detach()
        l3d = l3d.detach()
        rotY = rotY.detach()

    verts3d, corners_3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

    if type(cx3d) == np.ndarray:
        ign = np.any(corners_3d[:, 2, :] <= 0, axis=1)
        x = verts3d[:, 0, :].min(axis=1)
        y = verts3d[:, 1, :].min(axis=1)
        x2 = verts3d[:, 0, :].max(axis=1)
        y2 = verts3d[:, 1, :].max(axis=1)

        return np.vstack((x, y, x2, y2)).T, ign

    if type(cx3d) == torch.Tensor:
        ign = torch.any(corners_3d[:, 2, :] <= 0, dim=1)
        x = verts3d[:, 0, :].min(dim=1)[0]
        y = verts3d[:, 1, :].min(dim=1)[0]
        x2 = verts3d[:, 0, :].max(dim=1)[0]
        y2 = verts3d[:, 1, :].max(dim=1)[0]

        return torch.cat((x.unsqueeze(1), y.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1), ign

    else:
        # any boxes behind camera plane?
        ign = np.any(corners_3d[2, :] <= 0)
        x = min(verts3d[:, 0])
        y = min(verts3d[:, 1])
        x2 = max(verts3d[:, 0])
        y2 = max(verts3d[:, 1])

        return np.array([x, y, x2, y2]), ign


def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    if type(x3d) == np.ndarray:

        p2_batch = np.zeros([x3d.shape[0], 4, 4])
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = np.cos(ry3d)
        ry3d_sin = np.sin(ry3d)

        R = np.zeros([x3d.shape[0], 4, 3])
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = np.zeros([x3d.shape[0], 3, 8])

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = R @ corners_3d

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = p2_batch @ corners_3d

        corners_2d[:, :2, :] /= corners_2d[:, 2, :][:, np.newaxis, :]

        verts3d = corners_2d

    elif type(x3d) == torch.Tensor:

        p2_batch = torch.zeros(x3d.shape[0], 4, 4)
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = torch.cos(ry3d)
        ry3d_sin = torch.sin(ry3d)

        R = torch.zeros(x3d.shape[0], 4, 3)
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = torch.zeros(x3d.shape[0], 3, 8)

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = torch.bmm(R, corners_3d)

        corners_3d = corners_3d.to(x3d.device)
        p2_batch = p2_batch.to(x3d.device)

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = torch.bmm(p2_batch, corners_3d)

        corners_2d[:, :2, :] /= corners_2d[:, 2, :][:, np.newaxis, :]

        verts3d = corners_2d

    else:

        # compute rotational matrix around yaw axis
        R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                      [0, 1, 0],
                      [-math.sin(ry3d), 0, +math.cos(ry3d)]])

        # 3D bounding box corners
        x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
        y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
        z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

        x_corners += -l3d / 2
        y_corners += -h3d / 2
        z_corners += -w3d / 2

        # bounding box in object co-ordinate
        corners_3d = np.array([x_corners, y_corners, z_corners])

        # rotate
        corners_3d = R.dot(corners_3d)

        # translate
        corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

        corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
        corners_2D = p2.dot(corners_3D_1)
        corners_2D = corners_2D / corners_2D[2]

        # corners_2D = np.zeros([3, corners_3d.shape[1]])
        # for i in range(corners_3d.shape[1]):
        #    a, b, c, d = argoverse.utils.calibration.proj_cam_to_uv(corners_3d[:, i][np.newaxis, :], p2)
        #    corners_2D[:2, i] = a
        #    corners_2D[2, i] = corners_3d[2, i]

        bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

        verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d


def project_3d_corners(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
    y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
    z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

    '''
    order of vertices
    0  upper back right
    1  upper front right
    2  bottom front right
    3  bottom front left
    4  upper front left
    5  upper back left
    6  bottom back left
    7  bottom back right

    bot_inds = np.array([2,3,6,7])
    top_inds = np.array([0,1,4,5])
    '''

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]

    return corners_2D, corners_3D_1


def normalize(vec):

    vec /= norm(vec)
    return vec

def snap_to_pi(ry3d):

    if type(ry3d) == torch.Tensor:
        while (ry3d > (math.pi)).any(): ry3d[ry3d > (math.pi)] -= 2 * math.pi
        while (ry3d <= (-math.pi)).any(): ry3d[ry3d <= (-math.pi)] += 2 * math.pi
    elif type(ry3d) == np.ndarray:
        while np.any(ry3d > (math.pi)): ry3d[ry3d > (math.pi)] -= 2 * math.pi
        while np.any(ry3d <= (-math.pi)): ry3d[ry3d <= (-math.pi)] += 2 * math.pi
    else:

        while ry3d > math.pi: ry3d -= math.pi * 2
        while ry3d <= (-math.pi): ry3d += math.pi * 2

    return ry3d

