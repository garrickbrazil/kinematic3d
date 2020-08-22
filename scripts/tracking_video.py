# -----------------------------------------
# python modules
# -----------------------------------------
from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import sys
from copy import deepcopy
import numpy as np

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.imdb_util import *

# settings
exp_name = 'val1_kinematic'
weights_path = '/home/garrick/Desktop/Kinematic3D-Release/val1_kinematic/model_final'
conf_path = '/home/garrick/Desktop/Kinematic3D-Release/val1_kinematic/conf.pkl'

tracking_dir = '/home/garrick/Desktop/datasets/kitti/tracking'
out_dir = '/home/garrick/Desktop/tmp/'

suffix = '_tracking'
ignore_cache = True
write_im = 1

write_path = '{}/{}_ims/'.format(out_dir, exp_name + '_' + suffix)
results = '{}/{}'.format(out_dir, exp_name + '_' + suffix)
results_data = os.path.join(results, 'data')
cache_file = os.path.join(results, exp_name + '.pkl')

# load config
conf = edict(pickle_read(conf_path))
conf.pretrained = None

conf.has_un = True
conf.use_un_for_score = True
conf.only_tracks = False
conf.progressive = True
conf.video_count = 4

# make directories
mkdir_if_missing(results)
if write_im: mkdir_if_missing(write_path, delete_if_exist=True)
mkdir_if_missing(results_data, delete_if_exist=True)

MPH_FACTOR = 10*2.23694

# -----------------------------------------
# torch defaults
# -----------------------------------------

# default tensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# -----------------------------------------
# setup network
# -----------------------------------------

# -------------------- det network --------------------
net_det = import_module('models.' + conf.model).build(conf, 'train')
net_det.eval()

load_weights(net_det, weights_path, replace_module=True)
freeze_layers(net_det, [], None)
net_det.eval()

preprocess = Preprocess([conf.test_scale], conf.image_means, conf.image_stds)

has_cache = os.path.exists(cache_file)

# load cache?
if has_cache and not ignore_cache:
    cache_boxes = pickle_read(cache_file)
else:
    cache_boxes = []

# constants
bev_w = 615
bev_scale = 20
bev_c1 = (0, 250, 250)
bev_c2 = (0, 175, 250)
c_succ = (200, 50, 50)
c_fail = (0, 0, 255)
c_gts = (10, 175, 10)
track_thres = -1
match_thres = 0.1
max_pen = 1
vel_thres = np.inf

# make color bar
canvas_bev_orig = create_colorbar(50 * 20, bev_w, color_lo=bev_c1, color_hi=bev_c2)

trackers = []
base_prev = None
unique_objs = 0

stats_iou = []
stats_deg = []
stats_mat = []

def compute_track_velocity(ego, gts1, gts):

    pose_mat = np.eye(4)
    pose_mat[:3, :3] = euler2mat(ego[3], ego[4], ego[5])
    pose_mat[0, 3] = ego[0]
    pose_mat[1, 3] = ego[1]
    pose_mat[2, 3] = ego[2]

    for gtind0, gt0 in enumerate(gts):

        found = False

        for gtind1, gt1 in enumerate(gts1):
            if gt1.track == gt0.track:
                found = True
                break

        if found:

            x1 = gt1.center_3d[0]
            y1 = gt1.center_3d[1]
            z1 = gt1.center_3d[2]

            # compute velocity
            pos = pose_mat.dot(np.array([[x1, y1, z1, 1]]).T)

            x1 = pos[0].item()
            y1 = pos[1].item()
            z1 = pos[2].item()

            dx = gts[gtind0].center_3d[0] - x1
            dz = gts[gtind0].center_3d[2] - z1
            vel = np.sqrt(dx ** 2 + dz ** 2)
        else:
            vel = -np.inf

        gts[gtind0].vel = vel
        gts[gtind0].bbox_3d.append(vel)


tracks = None

def read_kitti_tracking_label(file, num_images=0):
    """
    Reads the kitti label file from disc.

    Args:
        file (str): path to single label file for an image
        p2 (ndarray): projection matrix for the given image
    """

    gts = [[] for i in range(num_images)]

    text_file = open(file, 'r')

    '''
     Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.
    '''

    pattern = re.compile(('(fpat)\s+(fpat)\s+([a-zA-Z\-\?\_]+)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s*((fpat)?)\n').replace('fpat', '[-+]?\d*\.\d+|[-+]?\d+'))


    for line in text_file:

        parsed = pattern.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:

            obj = edict()

            ign = False

            idx = int(parsed.group(1))
            track_id = int(parsed.group(2))
            cls = parsed.group(2+1)
            trunc = float(parsed.group(2+2))
            occ = float(parsed.group(2+3))
            alpha = float(parsed.group(2+4))

            x = float(parsed.group(2+5))
            y = float(parsed.group(2+6))
            x2 = float(parsed.group(2+7))
            y2 = float(parsed.group(2+8))

            width = x2 - x + 1
            height = y2 - y + 1

            h3d = float(parsed.group(2+9))
            w3d = float(parsed.group(2+10))
            l3d = float(parsed.group(2+11))

            cx3d = float(parsed.group(2+12)) # center of car in 3d
            cy3d = float(parsed.group(2+13)) # bottom of car in 3d
            cz3d = float(parsed.group(2+14)) # center of car in 3d
            rotY = float(parsed.group(2+15))

            # store the elevation of car (should be ~ 1.65)
            elevation = cy3d

            # actually center the box
            cy3d -= (h3d / 2)

            # encode occlusion with range estimation
            # 0 = fully visible, 1 = partly occluded
            # 2 = largely occluded, 3 = unknown
            if occ == 0: vis = 1
            elif occ == 1: vis = 0.66
            elif occ == 2: vis = 0.33
            else: vis = 0.0

            while rotY >= math.pi: rotY -= math.pi * 2
            while rotY < (-math.pi): rotY += math.pi * 2

            # recompute alpha
            alpha = convertRot2Alpha(rotY, cz3d, cx3d)

            # snap to [-pi, pi)
            while alpha > math.pi: alpha -= math.pi * 2
            while alpha <= (-math.pi): alpha += math.pi * 2

            alpha_cos = alpha
            alpha_sin = alpha

            # label for axis
            axis_lbl = np.abs(np.sin(alpha)) < np.abs(np.cos(alpha))

            # use sin, more vertical
            while alpha_sin > (math.pi / 2): alpha_sin -= math.pi
            while alpha_sin <= (-math.pi / 2): alpha_sin += math.pi

            # use cos, more horizontal
            while alpha_cos > (0): alpha_cos -= math.pi
            while alpha_cos <= (-math.pi): alpha_cos += math.pi

            # sin
            if axis_lbl == 1:
                head_acc = np.min([np.abs(alpha_sin - alpha), np.abs(snap_to_pi(alpha_sin + math.pi) - alpha)])
                head_lbl = np.argmin([np.abs(alpha_sin - alpha), np.abs(snap_to_pi(alpha_sin + math.pi) - alpha)])
                if not (np.isclose(head_acc, 0)): logging.log('WARNING, error in heading calculation not accurate!')

            # cos
            else:
                head_acc = np.min([np.abs(alpha_cos - alpha), np.abs(snap_to_pi(alpha_cos + math.pi) - alpha)])
                head_lbl = np.argmin([np.abs(alpha_cos - alpha), np.abs(snap_to_pi(alpha_cos + math.pi) - alpha)])
                if not (np.isclose(head_acc, 0)): logging.log('WARNING, error in heading calculation not accurate!')

            obj.elevation = elevation
            obj.cls = cls
            obj.occ = occ > 0
            obj.ign = ign
            obj.visibility = vis
            obj.trunc = trunc
            obj.alpha = alpha
            obj.rotY = rotY

            obj.track_id = track_id

            obj.bbox_full = np.array([x, y, width, height])

            obj.bbox_3d = [-1, -1, -1, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY, elevation, alpha_sin, alpha_cos, axis_lbl, head_lbl]
            obj.center_3d = [cx3d, cy3d, cz3d]

            if cls == 'Car' and not ign and vis > 0.5 and trunc < 0.05:
                gts[idx].append(obj)

    text_file.close()

    # wrap with numpy
    for imind in range(num_images):

        gts[imind] = np.array(gts[imind])

    # wrap with numpy
    gts = np.array(gts)

    return gts

seqs  = [4, 5, 6, 7, 9, 10, 11, 20]
order = [7, 5, 1, 3, 4,  6,  2,  0]

seqs = np.array(seqs)[np.argsort(order)].tolist()
order = np.array(order)[np.argsort(order)].tolist()

processing_times = []

for seqind, seq in enumerate(seqs): #range(0, 20+1):

    print('seq {}/{}'.format(seqind + 1, len(seqs)))

    net_det.tracks = None

    # init
    test_start = time()

    im_path = '{}/training/image_02/{:04d}/'.format(tracking_dir, seq)
    la_path = '{}/training/label_02/{:04d}.txt'.format(tracking_dir, seq)

    imlist = list_files(im_path, '*.png')
    im_gts = read_kitti_tracking_label(la_path, num_images=len(imlist))

    # first write normal_predictions
    file = open(os.path.join(results_data, '{:04d}.txt'.format(seq)), 'w')
    text_to_write = ''

    for imind, impath in enumerate(imlist):

        base_path, name, ext = file_parts(impath)

        name = '{:04d}_{}'.format(seq, name)

        ims = cv2.imread(impath)
        im_orig = deepcopy(ims)

        h_before = ims.shape[0]

        ims = deepcopy(ims)

        p2 = read_kitti_cal(os.path.join('data', 'kitti', 'tracking', 'training', 'calib', '{:04d}.txt'.format(seq)))

        p_start = time()
        ims = preprocess(ims)
        ims = torch.from_numpy(ims).cuda()

        scale_factor = conf.test_scale / h_before

        # read in calib
        p2_inv = np.linalg.inv(p2)

        p2_inv = np.linalg.inv(p2)[np.newaxis, :, :]
        p2 = p2[np.newaxis, :, :]

        if write_im:
            gts = im_gts[imind]
            gts_full = bbXYWH2Coords(np.array([gt['bbox_full'] for gt in gts]))
            gts_3d = np.array([gt['bbox_3d'] for gt in gts])
            gts_cen = np.array([gt['center_3d'] for gt in gts])
            gts_cls = np.array([gt['cls'] for gt in gts])

            # copy the canvas
            canvas_bev = deepcopy(canvas_bev_orig)

        # use cache to get boxes?
        si_shots, tr_shots, poses = net_det.forward_single(ims[np.newaxis, :, :, :], p2, p2_inv, [scale_factor])

        processing_times.append(time() - p_start)

        if tr_shots is None or tr_shots[-1] is None:
            aboxes = []
        else:

            tracks = tr_shots[-1][0]

            if tracks is None or len(tracks.Xs) == 0:
                aboxes = []

            else:

                bbox_2d = tracks.box2ds.detach().cpu().numpy()
                ids = tracks.ids
                Xs = tracks.Xs.detach().cpu().numpy()
                bbox_un = tracks.bbox_un.unsqueeze(1).detach().cpu().numpy()

                # apply head
                Xs[Xs[:, 7] >= 0.5, 6] += math.pi
                Cs = tracks.Cs.detach().cpu().numpy()
                Cs = np.array([a.diagonal() for a in Cs])

                # recall that Xs is:
                # [x, y, z, w, h, l, theta, head, vel]
                aboxes = np.concatenate((bbox_2d, Xs, bbox_un, np.array(ids)[:, np.newaxis]), axis=1)

                if 'history' in tracks:
                    history = tracks.history

                else:
                    history = None


            if seq == 3: continue
            if seq == 4 and (imind < 188 or imind > 249): continue
            if seq == 5 and (imind < 185 or imind > 289): continue
            if seq == 6 and (imind < 47 or imind > 136): continue
            if seq == 7 and (imind < 162 or imind > 255): continue
            if seq == 9 and (imind < 52 or imind > 274): continue
            if seq == 10 and (imind < 70 or imind > 176): continue
            if seq == 11 and (imind < 1 or imind > 360): continue
            if seq == 20 and (imind < 1 or imind > 635): continue

            for boxind in range(0, min(conf.nms_topN_post, len(aboxes))):

                box = aboxes[boxind, :]
                score = box[4]
                cls = conf.lbls[int(box[5] - 1)]

                if ('has_un' in conf) and conf.has_un:
                    un = box[15]

                if un > conf.score_thres and cls == 'Car':

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

                    # id
                    track_id = int(box[16])

                    box_proj, ign = get_2D_from_3D(p2[0], x3d, y3d, z3d, w3d, h3d, l3d, ry3d)

                    width = (x2 - x1 + 1)
                    height = (y2 - y1 + 1)

                    vel = -1 if not (('has_vel' in conf) and conf.has_vel) else np.clip(box[14], a_min=0, a_max=100)


                    box_2d = np.array([x1, y1, width, height])

                    alpha = convertRot2Alpha(ry3d, z3d, x3d)
                    c = get_color(299)
                    if np.abs(vel) > 0.05 and write_im:
                        x3d_fut = x3d + vel * np.cos(ry3d)
                        y3d_fut = y3d
                        z3d_fut = z3d - vel * np.sin(ry3d)

                        x2d, y2d, _ = project_3d_point(p2[0], x3d, y3d, z3d)
                        x2d_fut, y2d_fut, _ = project_3d_point(p2[0], x3d_fut, y3d_fut, z3d_fut)


                        cv2.arrowedLine(im_orig, (int(x2d), int(y2d)), (int(x2d_fut), int(y2d_fut)), (0, 220, 0), 4)

                        c = get_color(track_id)

                        # draw history
                        if history is not None and str(track_id) in history:

                            obj_hist = history[str(track_id)].detach().cpu().numpy()

                            # apply head
                            obj_hist[obj_hist[:, 7] >= 0.5, 6] += math.pi

                            # draw for each point
                            for histind in np.flatnonzero((obj_hist[:, 9] > 0.75) & (np.sqrt((x3d - obj_hist[:, 0])**2 + (z3d - obj_hist[:, 2])**2) > 1.25 )): #range(obj_hist.shape[0]):

                                #if obj_hist[histind, 9] < 0.75: continue

                                x = obj_hist[histind, 0] * bev_scale
                                z = obj_hist[histind, 2] * bev_scale

                                dist = np.sqrt((x3d - obj_hist[histind, 0])**2 + (z3d - obj_hist[histind, 2])**2)
                                dist_scaling = min(dist / 20, 1)

                                faded_c = np.array(c) + (255 - np.array(c)) * dist_scaling
                                faded_c = np.floor(faded_c).astype(int).tolist()

                                w = obj_hist[histind, 5] * bev_scale
                                l = obj_hist[histind, 3] * bev_scale
                                r = obj_hist[histind, 6] * -1

                                corners1 = np.array([
                                    [-w / 2, -l / 2, 1],
                                    [+w / 2, -l / 2, 1],
                                    [+w / 2, +l / 2, 1],
                                    [-w / 2, +l / 2, 1]
                                ])

                                ry = np.array([
                                    [+math.cos(r), -math.sin(r), 0],
                                    [+math.sin(r), math.cos(r), 0],
                                    [0, 0, 1],
                                ])

                                corners2 = ry.dot(corners1.T).T

                                corners2[:, 0] += w / 2 + x + canvas_bev.shape[1] / 2
                                corners2[:, 1] += l / 2 + z

                                x = corners2[:, 0].mean()
                                y = corners2[:, 1].mean()

                                draw_transparent_square(canvas_bev, (x, z), alpha=dist_scaling, color=c, radius=5)

                    if write_im:

                        cur_2d = np.array([[x1, y1, x2, y2]])
                        verts_cur, corners_3d_cur = project_3d(p2[0], x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=True)

                        # match with gt
                        ols_gts = iou(cur_2d, gts_full)[0, :]
                        if ols_gts.shape[0] > 0:
                            gtind = np.argmax(ols_gts)
                            ol_gt = np.amax(ols_gts)
                        else:
                            ol_gt = 0

                        # found gt?
                        if ol_gt > 0.25:

                            # get gt values
                            gt_x3d = gts_cen[gtind, 0]
                            gt_y3d = gts_cen[gtind, 1]
                            gt_z3d = gts_cen[gtind, 2]
                            gt_w3d = gts_3d[gtind, 3]
                            gt_h3d = gts_3d[gtind, 4]
                            gt_l3d = gts_3d[gtind, 5]
                            gt_rotY = gts_3d[gtind, 10]
                            gt_el = gts_3d[gtind, 11]

                            verts_gts, corners_3d_gt = project_3d(p2[0], gt_x3d, gt_y3d, gt_z3d, gt_w3d, gt_h3d, gt_l3d, gt_rotY, return_3d=True)

                            # compute 3D IoU
                            iou_bev_cur, iou_3d_cur = iou3d(corners_3d_cur, corners_3d_gt, gt_h3d * gt_l3d * gt_w3d + h3d * l3d * w3d)

                            iou_3d_cur = np.round(iou_3d_cur, 2)
                            draw_bev(canvas_bev, gt_z3d, gt_l3d, gt_w3d, gt_x3d, gt_rotY, color=c_gts, scale=bev_scale, thickness=6)

                            iou_str = '{:.2f}'.format(iou_3d_cur)
                        else:
                            iou_str = 'NaN'

                        draw_text(im_orig, '{:d}'.format(int(vel * MPH_FACTOR)), (x1, y1), scale=0.6, lineType=2)
                        c = get_color(track_id)

                        # draw detected box
                        verts_cur, corners_3d_cur = project_3d(p2[0], x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=True)
                        draw_3d_box(im_orig, verts_cur, c, thickness=2)
                        draw_bev(canvas_bev, z3d, l3d, w3d, x3d, ry3d, color=c, scale=bev_scale, thickness=6)

                    y3d += h3d / 2

                    text_to_write += ('{} {} {} -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} '
                                  + '{:.6f} {:.6f}\n').format(imind, track_id, cls, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score)

        if write_im:

            canvas_bev = cv2.flip(canvas_bev, 0)

            # draw tick marks
            ticks = [50, 40, 30, 20, 10, 0]
            draw_tick_marks(canvas_bev, ticks)


            im_concat = imhstack(im_orig, canvas_bev)

            if len(poses) > 0:
                pose = poses[0]
                pose = pose.cpu().numpy() * conf.pose_stds + conf.pose_means

                draw_text(im_concat, 'Speed: {:d} MPH'.format(int(max(-pose[0, 2]*MPH_FACTOR,0))), [0, 19*2], scale=0.825*2, lineType=3)
                imwrite(im_concat, write_path + '/' + '{:04d}_{:04d}_{:06d}'.format(order[seqind], seq, imind) + '.jpg')

        # display stats
        if (imind + 1) % 100 == 0:
            time_str, dt = compute_eta(test_start, imind + 1, len(imlist))
            print('testing {}/{}, dt: {:0.3f}, eta: {}, processing_time={:.4f}'.format(imind + 1, len(imlist), dt, time_str, np.mean(processing_times)))


    file.write(text_to_write)
    file.close()


if not has_cache and not ignore_cache:
    pickle_write(cache_file, cache_boxes)
