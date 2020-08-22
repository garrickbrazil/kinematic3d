# -----------------------------------------
# python modules
# -----------------------------------------
from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import sys
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.imdb_util import *

# settings
exp_name = 'val1_uncertainty'
weights_path = '/home/garrick/Desktop/Kinematic3D-Release/val1_uncertainty/model_final'
conf_path = '/home/garrick/Desktop/Kinematic3D-Release/val1_uncertainty/conf.pkl'

out_dir = '/home/garrick/Desktop/tmp/'

dataset_test = 'kitti_split1'
dataset_type = 'validation'

suffix = '_ss'
ignore_cache = 0
write_im = 1
use_gts = 1

print(exp_name)

eval_type = 'evaluate_object'

# paths
write_path = '{}/{}_ims/'.format(out_dir, exp_name + '_' + dataset_type + suffix)
results = '{}/{}'.format(out_dir, exp_name + '_' + dataset_type + suffix)
results_data = os.path.join(results, 'data')
cache_file = os.path.join(results, exp_name + '.pkl')

use_gts |= write_im

# load config
rpn_conf = edict(pickle_read(conf_path))
rpn_conf.pretrained = None

# make directories
mkdir_if_missing(results)
mkdir_if_missing(results_data, delete_if_exist=True)
if write_im:
    mkdir_if_missing(write_path, delete_if_exist=True)

# -----------------------------------------
# torch defaults
# -----------------------------------------

# default tensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# -----------------------------------------
# setup network
# -----------------------------------------

# net
net = import_module('models.' + rpn_conf.model).build(rpn_conf, 'train')
load_weights(net, weights_path, replace_module=True)
net.eval()

# -----------------------------------------
# test kitti
# -----------------------------------------
imlist = list_files(os.path.join('data', dataset_test, dataset_type, 'image_2', ''), '*'+rpn_conf.datasets_train[0]['im_ext'])
preprocess = Preprocess([rpn_conf.test_scale], rpn_conf.image_means, rpn_conf.image_stds)

has_cache = os.path.exists(cache_file)

# load cache?
if has_cache and not ignore_cache:
    cache_boxes = pickle_read(cache_file)
else:
    cache_boxes = []

# init
test_start = time()

# constants
bev_w = 615
bev_scale = 20
bev_c1 = (0, 250, 250)
bev_c2 = (0, 175, 250)
c_succ = (200, 50, 50)
c_fail = (0, 0, 255)
c_gts = (10, 175, 10)

# make color bar
canvas_bev_orig = create_colorbar(50 * 20, bev_w, color_lo=bev_c1, color_hi=bev_c2)

rpn_conf.score_thres = 0.75 if not ('score_thres' in rpn_conf) else rpn_conf.score_thres

scores = []
scores_orig = []
stats_iou = []
stats_deg = []
stats_mat = []
stats_z = []

for imind, impath in enumerate(imlist):

    base_path, name, ext = file_parts(impath)

    im0 = cv2.imread(impath)

    h_before = im0.shape[0]

    scale_factor = rpn_conf.test_scale / h_before

    # read in calib
    p2 = read_kitti_cal(os.path.join('data', dataset_test, dataset_type, 'calib', name + '.txt'))
    p2_inv = np.linalg.inv(p2)

    p2_a = p2[0, 0].item()
    p2_b = p2[0, 2].item()
    p2_c = p2[0, 3].item()
    p2_d = p2[1, 1].item()
    p2_e = p2[1, 2].item()
    p2_f = p2[1, 3].item()
    p2_h = p2[2, 3].item()

    if write_im:

        # copy the canvas
        canvas_bev = deepcopy(canvas_bev_orig)
        im0_orig = deepcopy(im0)

    if use_gts:
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

    if has_cache and not ignore_cache:
        aboxes = cache_boxes[imind][0]

    else:

        # forward test batch
        aboxes = im_detect_3d(im0, net, rpn_conf, preprocess, p2)
        cache_boxes.append(deepcopy((aboxes,)))

    base_path, name, ext = file_parts(impath)

    results_data = results_data

    file = open(os.path.join(results_data, name + '.txt'), 'w')
    text_to_write = ''

    for boxind in range(0, min(rpn_conf.nms_topN_post, aboxes.shape[0])):

        box = aboxes[boxind, :]
        score = box[4]
        cls = rpn_conf.lbls[int(box[5] - 1)]

        if 'has_un' in rpn_conf and rpn_conf.has_un:
            score = score * box[13]

        if cls.lower() =='car':

            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            width = (x2 - x1 + 1)
            height = (y2 - y1 + 1)

            x2d = box[6]
            y2d = box[7]
            z2d = box[8]
            w3d = box[9]
            h3d = box[10]
            l3d = box[11]
            alpha = box[12]

            # convert alpha into ry3d
            coord3d = np.linalg.inv(p2).dot(np.array([x2d * z2d, y2d * z2d, 1 * z2d, 1]))
            ry3d = convertAlpha2Rot(alpha, coord3d[2], coord3d[0])

            step_r = 0.3 * math.pi
            r_lim = 0.01

            box_2d = np.array([x1, y1, width, height])

            while ry3d > math.pi: ry3d -= math.pi * 2
            while ry3d < (-math.pi): ry3d += math.pi * 2

            # predict a more accurate projection
            coord3d = np.linalg.inv(p2).dot(np.array([x2d * z2d, y2d * z2d, 1 * z2d, 1]))

            x3d = coord3d[0]
            y3d = coord3d[1]
            z3d = coord3d[2]

            if use_gts:

                cur_2d = np.array([[x1, y1, x2, y2]])
                verts_cur, corners_3d_cur = project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=True)

                # match with gt
                ols_gts = iou(cur_2d, gts_full)[0, :]
                if ols_gts.shape[0] > 0:
                    gtind = np.argmax(ols_gts)
                    ol_gt = np.amax(ols_gts)
                else: ol_gt = 0

                # found gt?
                if ol_gt > 0.5:

                    # get gt values
                    gt_x3d = gts_cen[gtind, 0]
                    gt_y3d = gts_cen[gtind, 1]
                    gt_z3d = gts_cen[gtind, 2]
                    gt_x2d = gts_cen[gtind, 0]
                    gt_y2d = gts_cen[gtind, 1]
                    gt_z2d = gts_cen[gtind, 2]
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

                    scores_orig.append(box[4])
                    scores.append(score)
                    stats_iou.append(iou_3d_cur)

                    if score > rpn_conf.score_thres:
                        stats_z.append(np.abs(gt_z3d - z3d))
                        stats_deg.append(np.rad2deg(np.abs(snap_to_pi(gt_alpha) - snap_to_pi(alpha))))

                    iou_3d_cur = np.round(iou_3d_cur, 2)

                    if cls == 'Cyclist' and iou_3d_cur >= 0.5: success = True
                    elif cls == 'Pedesrian' and iou_3d_cur >= 0.5: success = True
                    elif cls == 'Car' and iou_3d_cur >= 0.7: success = True
                    else: success = False

                    stats_mat.append(success)

                    c = c_succ if success else c_fail


                    if write_im and score > rpn_conf.score_thres:

                        # draw 3D iou and ground truth
                        draw_text(im0_orig, 'iou={:.2f}'.format(iou_3d_cur), (x1, y1))
                        draw_bev(canvas_bev, gt_z3d, gt_l3d, gt_w3d, gt_x3d, gt_rotY, color=c_gts, scale=bev_scale, thickness=2)

                else: c = c_fail

                if write_im and score > rpn_conf.score_thres:

                    # draw detected box
                    draw_3d_box(im0_orig, verts_cur, c, thickness=2)
                    draw_bev(canvas_bev, z3d, l3d, w3d, x3d, ry3d, color=c, scale=bev_scale, thickness=3)

            y3d += h3d / 2

            alpha = convertRot2Alpha(ry3d, z3d, x3d)

            if score > rpn_conf.score_thres:
                text_to_write += ('{} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                                  + '{:.6f} {:.6f}\n').format(cls, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score)

    file.write(text_to_write)
    file.close()

    if write_im:

        canvas_bev = cv2.flip(canvas_bev, 0)

        # draw tick marks
        ticks = [50, 40, 30, 20, 10, 0]
        draw_tick_marks(canvas_bev, ticks)
        im_concat = imhstack(im0_orig, canvas_bev)
        imwrite(im_concat, write_path + '/{:06d}'.format(imind) + '.jpg')

    # display stats
    if (imind + 1) % 100 == 0:
        time_str, dt = compute_eta(test_start, imind + 1, len(imlist))
        if use_gts:

            cor = np.corrcoef(np.array(scores), np.array(stats_iou))

            print('testing {}/{}, dt: {:0.3f}, eta: {}, match: {:.4f}, iou: {:.4f}, z: {:.1f}, rot: {:.1f}, cor: {:.4f}'
                  .format(imind + 1, len(imlist), dt, time_str, np.array(stats_mat).mean(), np.array(stats_iou).mean(),
                          np.mean(stats_z), np.array(stats_deg).mean(), cor[0, 1]))

        else:
            print('testing {}/{}, dt: {:0.3f}, eta: {}'.format(imind + 1, len(imlist), dt, time_str))


if use_gts:

    cor = np.corrcoef(np.array(scores), np.array(stats_iou))

    print('testing {}/{}, dt: {:0.3f}, eta: {}, match: {:.4f}, iou: {:.4f}, z: {:.1f}, rot: {:.1f}, cor: {:.4f}'
          .format(imind + 1, len(imlist), dt, time_str, np.array(stats_mat).mean(), np.array(stats_iou).mean(),
                  np.mean(stats_z), np.array(stats_deg).mean(), cor[0, 1]))

if not has_cache and not ignore_cache:
    pickle_write(cache_file, cache_boxes)

_, test_iter, _ = file_parts(results)
test_iter = test_iter.replace('results_', '')

if dataset_type == 'validation':
    evaluate_kitti_results_verbose('data', dataset_test, results_data, test_iter, rpn_conf, use_logging=False, fast=True, default_eval=eval_type)
