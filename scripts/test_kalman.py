# -----------------------------------------
# python modules
# -----------------------------------------
from easydict import EasyDict as edict
from getopt import getopt
import sys
import numpy as np
import torch
from importlib import import_module
torch.set_printoptions(sci_mode=False)

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.core import *
from lib.imdb_util import *

# settings
exp_name = 'val1_kinematic'
weights_path = '/home/garrick/Desktop/Kinematic3D-Release/val1_kinematic/model_final'
conf_path = '/home/garrick/Desktop/Kinematic3D-Release/val1_kinematic/conf.pkl'

out_dir = '/home/garrick/Desktop/tmp/'

dataset_type = 'validation'
eval_type = 'evaluate_object'
dataset_test = 'kitti_split1'

suffix = '_video'
ignore_cache = False
write_im = False

write_path = '{}/{}_ims/'.format(out_dir, exp_name + '_' + dataset_type + suffix)
results = '{}/{}'.format(out_dir, exp_name + '_' + dataset_type + suffix)
results_data = os.path.join(results, 'data')
cache_file = os.path.join(results, exp_name + '.pkl')

# -----------------------------------------
# torch defaults
# -----------------------------------------

# default tensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# load config
conf = edict(pickle_read(conf_path))

conf.dataset_test = dataset_test
conf.pretrained = None
conf.progressive = True
conf.video_count = 4
video_count_load = 4
conf.fast_eval = True

paths = init_training_paths(exp_name)

# make directories
mkdir_if_missing(results)
mkdir_if_missing(os.path.join(results, 'log'))
if write_im: mkdir_if_missing(write_path, delete_if_exist=True)
mkdir_if_missing(results_data, delete_if_exist=True)

# -----------------------------------------
# torch defaults
# -----------------------------------------

# default tensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# -----------------------------------------
# setup network
# -----------------------------------------

net_det = import_module('models.' + conf.model).build(conf, 'train')
load_weights(net_det, weights_path, replace_module=True)
freeze_layers(net_det, [], None)
net_det.eval()

# -----------------------------------------
# basic setup
# -----------------------------------------

init_torch(1, conf.cuda_seed)
init_log_file(os.path.join(results, 'log'))

# -----------------------------------------
# data and anchors
# -----------------------------------------

conf.has_un = True
conf.use_un_for_score = True
conf.only_tracks = False

MPH_FACTOR = 10*2.23694

# show configuration
pretty = pretty_print('conf', conf)
logging.info(pretty)

logging.info('Box extraction...')
print('videocount -> ', conf.video_count)

cache_bbox_val = os.path.join(results, 'bbox_extract_val_{}'.format(video_count_load))

if os.path.exists(cache_bbox_val):
    bbox_measure_val = pickle_read(cache_bbox_val)
else:
    bbox_measure_val = extract_kalman_boxes(conf.dataset_test, net_det, conf, paths.data, dataset_type)
    pickle_write(cache_bbox_val, bbox_measure_val)


logging.info('Data setup on GPU...')

imobjs = []

p2s = None
p2_invs = None
scales = []

p2s_val = None
p2_invs_val = None
scales_val = []

imlist = list_files(os.path.join(paths.data, conf.dataset_test, dataset_type, 'image_2', ''), '*.png')

bbox_measure_val_torch = []

imobjs_val = []

for imind, impath in enumerate(imlist):

    base_path, name, ext = file_parts(impath)

    # read in calib
    p2 = read_kitti_cal(os.path.join(paths.data, conf.dataset_test, dataset_type, 'calib', name + '.txt'))
    p2_inv = np.linalg.inv(p2)

    im = Image.open(impath)
    imW, imH = im.size

    scale_factor = conf.test_scale / imH

    if p2s_val is None: p2s_val = p2[np.newaxis, :, :]
    else: p2s_val = np.concatenate((p2s_val, p2[np.newaxis, :, :]), axis=0)

    if p2_invs_val is None: p2_invs_val = p2_inv[np.newaxis, :, :]
    else: p2_invs_val = np.concatenate((p2_invs_val, p2_inv[np.newaxis, :, :]), axis=0)

    scales_val.append(scale_factor)

p2s_val = torch.from_numpy(p2s_val).cuda().type(torch.cuda.FloatTensor)
p2_invs_val = torch.from_numpy(p2_invs_val).cuda().type(torch.cuda.FloatTensor)

logging.info('Starting box test')

bbox_measure_val_torch = []

net_det.video_count = conf.video_count

net_det.tracks = None
net_det.history = dict()

for imind, impath in enumerate(imlist):

    base_path, name, ext = file_parts(impath)

    obj = bbox_measure_val[imind]

    measures_torch = []
    poses_torch = []

    p2 = p2s_val[imind].cpu().numpy()
    count = 0

    for vid_idx in range(video_count_load):

        if vid_idx < (video_count_load - conf.video_count):
            continue
        count += 1

        measure = obj[vid_idx][0]
        pose = obj[vid_idx][1]

        if measure is not None:
            measure = measure[measure[:, 5] == 1]

        if measure is not None:
            measure = measure[measure[:, 5] == 1]
            measure = torch.from_numpy(measure).type(torch.cuda.FloatTensor)

            if measure.shape[0] == 0:
                measure = None

        if pose is not None:
            pose = torch.from_numpy(pose).type(torch.cuda.FloatTensor)

        measures_torch.append(measure)
        poses_torch.append(pose)


    measures_torch = np.array(measures_torch)
    poses_torch = np.array(poses_torch)

    bbox_measure_val_torch.append((measures_torch, poses_torch))

bbox_measure_val_torch = np.array(bbox_measure_val_torch)

net_det.lambda_o = 0.2
net_det.k_p = 0.75
net_det.k_m = 0.05

for varind in range(8):
    net_det.R_cov.data[varind] = net_det.lambda_o

results_path_ego = os.path.join(results, 'results_{}'.format((1)), 'data')
mkdir_if_missing(results_path_ego, delete_if_exist=True)

test_kitti_3d_kalman_boxes(conf.dataset_test, net_det, conf, bbox_measure_val_torch, p2s_val, p2_invs_val, scales_val, results_path_ego, paths.data, dataset_type=dataset_type, report_stats=True, eval_type=eval_type)
