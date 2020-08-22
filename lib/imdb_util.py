"""
This file contains all image database (imdb) functionality,
such as loading and reading information from a dataset.

Generally, this file is meant to read in a dataset from disk into a
simple custom format for the detetive framework.
"""

# -----------------------------------------
# modules
# -----------------------------------------
import torch
import torch.utils.data as data
import sys
import re
from PIL import Image
from copy import deepcopy
from scipy.io import loadmat

sys.dont_write_bytecode = True

# -----------------------------------------
# custom
# -----------------------------------------
from lib.rpn_util import *
from lib.util import *
from lib.augmentations import *
from lib.core import *
from lib.math_3d import *


class Dataset(torch.utils.data.Dataset):
    """
    A single Dataset class is used for the whole project,
    which implements the __init__ and __get__ functions from PyTorch.
    """

    def compute_track_velocity(self, ego, gts1, gts):

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

    def __init__(self, conf, root, cache_folder=None, data_type='training'):
        """
        This function reads in all datasets to be used in training and stores ANY relevant
        information which may be needed during training as a list of edict()
        (referred to commonly as 'imobj').

        The function also optionally stores the image database (imdb) file into a cache.
        """

        imdb = []

        self.video_det = False if not ('video_det' in conf) else conf.video_det
        self.video_count = 1 if not ('video_count' in conf) else conf.video_count
        self.use_all_tracks = False if not ('use_all_tracks' in conf) else conf.use_all_tracks
        self.use_3d_for_2d = ('use_3d_for_2d' in conf) and conf.use_3d_for_2d

        # use cache?
        if (cache_folder is not None) and data_type == 'training' and os.path.exists(os.path.join(cache_folder, 'imdb.pkl')):
            logging.info('Preloading imdb.')
            imdb = pickle_read(os.path.join(cache_folder, 'imdb.pkl'))

        elif (cache_folder is not None) and data_type == 'validation' and os.path.exists(os.path.join(cache_folder, 'imdb_val.pkl')):
            logging.info('Preloading imdb.')
            imdb = pickle_read(os.path.join(cache_folder, 'imdb_val.pkl'))

        elif (cache_folder is not None) and data_type == 'testing' and os.path.exists(os.path.join(cache_folder, 'imdb_test.pkl')):
            logging.info('Preloading imdb.')
            imdb = pickle_read(os.path.join(cache_folder, 'imdb_test.pkl'))

        else:

            # cycle through each dataset
            for dbind, db in enumerate(conf.datasets_train):

                logging.info('Loading imdb {}'.format(db['name']))

                # single imdb
                imdb_single_db = []

                # kitti formatting
                if db['anno_fmt'].lower() == 'kitti_det':

                    train_folder = os.path.join(root, db['name'], data_type)

                    ann_folder = os.path.join(train_folder, 'label_2', '')
                    cal_folder = os.path.join(train_folder, 'calib', '')
                    im_folder = os.path.join(train_folder, 'image_2', '')

                    # get sorted filepaths
                    annlist = sorted(glob(ann_folder + '*.txt'))

                    imdb_start = time()

                    self.affine_size = None if not ('affine_size' in conf) else conf.affine_size

                    for anind, annpath in enumerate(annlist):

                        # get file parts
                        base = os.path.basename(annpath)
                        id, ext = os.path.splitext(base)

                        calpath = os.path.join(cal_folder, id + '.txt')
                        impath = os.path.join(im_folder, id + db['im_ext'])
                        impath_pre = os.path.join(train_folder, 'prev_2', id + '_01' + db['im_ext'])
                        impath_pre2 = os.path.join(train_folder, 'prev_2', id + '_02' + db['im_ext'])
                        impath_pre3 = os.path.join(train_folder, 'prev_2', id + '_03' + db['im_ext'])

                        # read gts
                        p2 = read_kitti_cal(calpath)
                        p2_inv = np.linalg.inv(p2)

                        gts = read_kitti_label(annpath, p2, self.use_3d_for_2d)

                        obj = edict()

                        # store gts
                        obj.id = id
                        obj.gts = gts
                        obj.p2 = p2
                        obj.p2_inv = p2_inv

                        # im properties
                        im = Image.open(impath)
                        obj.path = impath
                        obj.path_pre = impath_pre
                        obj.path_pre2 = impath_pre2
                        obj.path_pre3 = impath_pre3
                        obj.imW, obj.imH = im.size

                        # database properties
                        obj.dbname = db.name
                        obj.scale = db.scale
                        obj.dbind = dbind

                        # store
                        imdb_single_db.append(obj)

                        if (anind % 1000) == 0 and anind > 0:
                            time_str, dt = compute_eta(imdb_start, anind, len(annlist))
                            logging.info('{}/{}, dt: {:0.4f}, eta: {}'.format(anind, len(annlist), dt, time_str))

                # kitti with some tracking information
                elif db['anno_fmt'].lower() == 'kitti_tracking':

                    train_folder = os.path.join(root, db['name'], data_type)

                    ann_folder = os.path.join(train_folder, 'label_2', '')
                    cal_folder = os.path.join(train_folder, 'calib', '')
                    im_folder = os.path.join(train_folder, 'image_2', '')

                    # get sorted filepaths
                    annlist = sorted(glob(ann_folder + '*.txt'))
                    imlist = sorted(glob(im_folder + '*.png'))

                    imdb_start = time()

                    if data_type == 'testing':


                        for anind, annpath in enumerate(imlist):

                            # get file parts
                            base = os.path.basename(annpath)
                            id, ext = os.path.splitext(base)

                            calpath = os.path.join(cal_folder, id + '.txt')
                            impath_00 = os.path.join(im_folder, id + db['im_ext'])
                            impath_01 = os.path.join(train_folder, 'prev_2', id + '_01' + db['im_ext'])
                            impath_02 = os.path.join(train_folder, 'prev_2', id + '_02' + db['im_ext'])
                            impath_03 = os.path.join(train_folder, 'prev_2', id + '_03' + db['im_ext'])

                            # read calibration
                            p2 = read_kitti_cal(calpath)
                            p2_inv = np.linalg.inv(p2)

                            obj = edict()

                            # store gts
                            obj.id = id
                            obj.p2 = p2
                            obj.p2_inv = p2_inv

                            # im properties
                            im = Image.open(impath_00)
                            obj.path = impath_00
                            obj.impath_01 = impath_01
                            obj.impath_02 = impath_02
                            obj.impath_03 = impath_03
                            obj.imW, obj.imH = im.size

                            obj.has_track = False

                            # database properties
                            obj.dbname = db.name
                            obj.scale = db.scale
                            obj.dbind = dbind

                            imdb_single_db.append(obj)

                            if (anind % 1000) == 0 and anind > 0:
                                time_str, dt = compute_eta(imdb_start, anind, len(annlist))
                                logging.info('{}/{}, dt: {:0.4f}, eta: {}'.format(anind, len(annlist), dt, time_str))
                    else:

                        mapping_file = os.path.join(root, db['name'], 'devkit', 'mapping', 'train_mapping.txt')

                        mapping = []
                        ids = []

                        # read train_mapping.txt
                        text_file = open(mapping_file, 'r')
                        for line in text_file:

                            # 2011_09_26 2011_09_26_drive_0005_sync 0000000109
                            parsed = re.search('(\S+)\s+(\S+)\s+(\S+)', line)

                            if parsed is not None:
                                raw_seq = str(parsed[2])
                                raw_id = str(parsed[3])
                                mapping.append([raw_seq, raw_id])

                        text_file.close()

                        rand_file = os.path.join(root, db['name'], 'devkit', 'mapping', 'train_rand.txt')
                        rand_map = []

                        # read train_rand.txt
                        text_file = open(rand_file, 'r')
                        for line in text_file:
                            parsed = re.findall('(\d+)', line)
                            for p in parsed:
                                rand_map.append(int(p))
                        text_file.close()

                        if db['name'].lower() == 'kitti_split1':

                            if data_type == 'training':
                                ids_file = os.path.join(root, db['name'], 'train.txt')
                            elif data_type == 'validation':
                                ids_file = os.path.join(root, db['name'], 'val.txt')

                            # read train ids
                            text_file = open(ids_file, 'r')
                            for line in text_file:

                                parsed = re.search('(\d+)', line)

                                if parsed is not None:
                                    id = int(parsed[0])
                                    raw = mapping[rand_map[id] - 1]
                                    ids.append(raw)

                        elif db['name'].lower() == 'kitti_split2':

                            split_data = loadmat(os.path.join(root, db['name'], 'kitti_ids_new.mat'))

                            for id_num in split_data['ids_train'][0]:
                                raw = mapping[rand_map[id_num] - 1]
                                ids.append(raw)

                        elif db['name'].lower() == 'kitti':

                            for id_num in range(len(mapping)):
                                raw = mapping[rand_map[id_num] - 1]
                                ids.append(raw)

                        for anind, annpath in enumerate(annlist):

                            # get file parts
                            base = os.path.basename(annpath)
                            id, ext = os.path.splitext(base)

                            calpath = os.path.join(cal_folder, id + '.txt')
                            impath_00 = os.path.join(im_folder, id + db['im_ext'])
                            impath_01 = os.path.join(train_folder, 'prev_2', id + '_01' + db['im_ext'])
                            impath_02 = os.path.join(train_folder, 'prev_2', id + '_02' + db['im_ext'])
                            impath_03 = os.path.join(train_folder, 'prev_2', id + '_03' + db['im_ext'])

                            # read calibration
                            p2 = read_kitti_cal(calpath)
                            p2_inv = np.linalg.inv(p2)

                            # read gts
                            gts = read_kitti_label(annpath, p2, self.use_3d_for_2d)

                            obj = edict()

                            # store gts
                            obj.id = id
                            obj.gts = gts
                            obj.p2 = p2
                            obj.p2_inv = p2_inv

                            # store the raw data sequence and number
                            obj.raw_id = ids[anind]

                            # im properties
                            im = Image.open(impath_00)
                            obj.path = impath_00
                            obj.impath_01 = impath_01
                            obj.impath_02 = impath_02
                            obj.impath_03 = impath_03
                            obj.imW, obj.imH = im.size

                            # read in the raw poses
                            all_poses = read_kitti_poses(
                                os.path.join(root, db['name'], 'raw_extra', obj.raw_id[0], 'pose.txt'))

                            # read in all poses
                            ego_0 = all_poses[int(obj.raw_id[1]) - 0]
                            ego_1 = all_poses[int(obj.raw_id[1]) - 1]
                            ego_2 = all_poses[int(obj.raw_id[1]) - 2]
                            ego_3 = all_poses[int(obj.raw_id[1]) - 3]

                            # compute the relative poses; store
                            #  32, 31, 30, 21, 20, 10
                            obj.ego_32 = (0, 0, 0, 0, 0, 0,) if (int(obj.raw_id[1]) - 3) < 0 else compute_rel_pose(ego_3, ego_2)
                            obj.ego_31 = (0, 0, 0, 0, 0, 0,) if (int(obj.raw_id[1]) - 3) < 0 else compute_rel_pose(ego_3, ego_1)
                            obj.ego_30 = (0, 0, 0, 0, 0, 0,) if (int(obj.raw_id[1]) - 3) < 0 else compute_rel_pose(ego_3, ego_0)
                            obj.ego_21 = (0, 0, 0, 0, 0, 0,) if (int(obj.raw_id[1]) - 2) < 0 else compute_rel_pose(ego_2, ego_1)
                            obj.ego_20 = (0, 0, 0, 0, 0, 0,) if (int(obj.raw_id[1]) - 2) < 0 else compute_rel_pose(ego_2, ego_0)
                            obj.ego_10 = (0, 0, 0, 0, 0, 0,) if (int(obj.raw_id[1]) - 1) < 0 else compute_rel_pose(ego_1, ego_0)

                            extra_path = os.path.join(root, db['name'], 'raw_extra', obj.raw_id[0], 'label_2')

                            # check to see if there is tracking available
                            if os.path.exists(os.path.join(extra_path)) and (int(obj.raw_id[1]) - 3) >= 0:

                                # yes? then read in each gts; store as gts_1, gts_2, gts_3
                                # and set has_tracklets to True
                                obj.has_track = True

                                obj.gts = read_kitti_label(
                                    os.path.join(extra_path, '{:06d}.txt'.format(int(obj.raw_id[1]) - 0)), p2, self.use_3d_for_2d)
                                obj.gts_1 = read_kitti_label(
                                    os.path.join(extra_path, '{:06d}.txt'.format(int(obj.raw_id[1]) - 1)), p2, self.use_3d_for_2d)
                                obj.gts_2 = read_kitti_label(
                                    os.path.join(extra_path, '{:06d}.txt'.format(int(obj.raw_id[1]) - 2)), p2, self.use_3d_for_2d)
                                obj.gts_3 = read_kitti_label(
                                    os.path.join(extra_path, '{:06d}.txt'.format(int(obj.raw_id[1]) - 3)), p2, self.use_3d_for_2d)

                                self.compute_track_velocity(obj.ego_10, obj.gts_1, obj.gts)
                                self.compute_track_velocity(obj.ego_21, obj.gts_2, obj.gts_1)
                                self.compute_track_velocity(obj.ego_32, obj.gts_3, obj.gts_2)

                            else:
                                # no? set has_tracklets to False
                                obj.has_track = False

                            # database properties
                            obj.dbname = db.name
                            obj.scale = db.scale
                            obj.dbind = dbind

                            if not ('only_tracks' in conf and conf.only_tracks and (not obj.has_track)):
                                # store
                                imdb_single_db.append(obj)

                            if (anind % 1000) == 0 and anind > 0:
                                time_str, dt = compute_eta(imdb_start, anind, len(annlist))
                                logging.info('{}/{}, dt: {:0.4f}, eta: {}'.format(anind, len(annlist), dt, time_str))

                # concatenate single imdb into full imdb
                imdb += imdb_single_db

            imdb = np.array(imdb)

            # cache off the imdb?
            if cache_folder is not None:
                if data_type == 'training':
                    pickle_write(os.path.join(cache_folder, 'imdb.pkl'), imdb)
                elif data_type == 'validation':
                    pickle_write(os.path.join(cache_folder, 'imdb_val.pkl'), imdb)

        # store more information
        self.datasets_train = conf.datasets_train
        self.len = len(imdb)
        self.imdb = imdb

        # setup data augmentation transforms
        self.transform = Augmentation(conf)

        # for depth
        if conf.datasets_train[0]['anno_fmt'].lower() == 'kitti_depth':
            self.loader = torch.utils.data.DataLoader(self, conf.batch_size, collate_fn=self.collate, shuffle=True)

        elif conf.datasets_train[0]['anno_fmt'].lower() == 'kitti_vo':
            self.loader = torch.utils.data.DataLoader(self, conf.batch_size, collate_fn=self.collate, shuffle=True)

        # for detection datasets
        else:

            if not (data_type == 'testing'):

                # setup sampler and data loader for this dataset
                self.sampler = torch.utils.data.sampler.WeightedRandomSampler(balance_samples(conf, imdb), self.len)
                self.loader = torch.utils.data.DataLoader(self, conf.batch_size, sampler=self.sampler, collate_fn=self.collate)

        # check classes
        cls_not_used = []
        for imobj in imdb:

            for gt in imobj.gts:
                cls = gt.cls
                if not(cls in conf.lbls or cls in conf.ilbls) and (cls not in cls_not_used):
                    cls_not_used.append(cls)

        if len(cls_not_used) > 0:
            logging.info('Labels not used.. {}'.format(cls_not_used))


    def __getitem__(self, index):
        """
        Grabs the item at the given index. Specifically,
          - read the image from disk
          - read the imobj from RAM
          - applies data augmentation to (im, imobj)
          - converts image to RGB and [B C W H]
        """

        if not self.video_det:

            # read image
            im = cv2.imread(self.imdb[index].path)

        else:

            # read images
            im = cv2.imread(self.imdb[index].path)

            video_count = 1 if self.video_count is None else self.video_count

            if self.use_all_tracks and self.imdb[index].has_track:
                video_count = 4


            if video_count >= 2:
                im_pre = cv2.imread(self.imdb[index].impath_01)

                if im_pre is None:
                    im_pre = im

                if not im_pre.shape == im.shape:
                    im_pre = cv2.resize(im_pre, (im.shape[1], im.shape[0]))

                im = np.concatenate((im, im_pre), axis=2)

            if video_count >= 3:

                im_pre2 = cv2.imread(self.imdb[index].impath_02)

                if im_pre2 is None:
                    im_pre2 = im_pre

                if not im_pre2.shape == im.shape:
                    im_pre2 = cv2.resize(im_pre2, (im.shape[1], im.shape[0]))

                im = np.concatenate((im, im_pre2), axis=2)

            if video_count >= 4:

                im_pre3 = cv2.imread(self.imdb[index].impath_03)

                if im_pre3 is None:
                    im_pre3 = im_pre2

                if not im_pre3.shape == im.shape:
                    im_pre3 = cv2.resize(im_pre3, (im.shape[1], im.shape[0]))

                im = np.concatenate((im, im_pre3), axis=2)

        imobj = deepcopy(self.imdb[index])

        # transform / data augmentation
        im, imobj = self.transform(im, imobj)

        #im_tmp = deepcopy(im)
        #im_tmp *= np.tile(self.transform.stds, int(im_tmp.shape[2] / len(self.transform.stds)))
        #im_tmp += np.tile(self.transform.mean, int(im_tmp.shape[2] / len(self.transform.mean)))
        #im_tmp *= 255.0
        #im_tmp = im_tmp.astype(np.uint8)
        #imshow(im_tmp)
        #plt.show(block=True)

        # carefully convert to RGB
        for i in range(int(im.shape[2]/3)):
            im[:, :, (i*3):(i*3) + 3] = im[:, :, (i*3+2, i*3+1, i*3)]

        # then permute to be [B C H W]
        im = np.transpose(im, [2, 0, 1])

        return im, imobj

    @staticmethod
    def collate(batch):
        """
        Defines the methodology for PyTorch to collate the objects
        of a batch together, for some reason PyTorch doesn't function
        this way by default.
        """

        imgs = []
        imobjs = []

        # go through each batch
        for sample in batch:
            # append images and object dictionaries
            imgs.append(sample[0])
            imobjs.append(sample[1])

        # stack images
        imgs = np.array(imgs)
        imgs = torch.from_numpy(imgs).cuda()

        return imgs, np.array(imobjs)

    def __len__(self):
        """
        Simply return the length of the dataset.
        """
        return self.len


def get_kitti_raw_ids(dataset_root, dataset_name, data_type='validation'):

    mapping_file = os.path.join(dataset_root, dataset_name, 'devkit', 'mapping', 'train_mapping.txt')

    mapping = []
    ids = []

    # read train_mapping.txt
    text_file = open(mapping_file, 'r')
    for line in text_file:

        # 2011_09_26 2011_09_26_drive_0005_sync 0000000109
        parsed = re.search('(\S+)\s+(\S+)\s+(\S+)', line)

        if parsed is not None:
            raw_seq = str(parsed[2])
            raw_id = str(parsed[3])
            mapping.append([raw_seq, raw_id])

    text_file.close()

    rand_file = os.path.join(dataset_root, dataset_name, 'devkit', 'mapping', 'train_rand.txt')
    rand_map = []

    # read train_rand.txt
    text_file = open(rand_file, 'r')
    for line in text_file:
        parsed = re.findall('(\d+)', line)
        for p in parsed:
            rand_map.append(int(p))
    text_file.close()

    if dataset_name.lower() == 'kitti_split1':

        if data_type == 'training':
            ids_file = os.path.join(dataset_root, dataset_name, 'train.txt')
        elif data_type == 'validation':
            ids_file = os.path.join(dataset_root, dataset_name, 'val.txt')

        # read train ids
        text_file = open(ids_file, 'r')
        for line in text_file:

            parsed = re.search('(\d+)', line)

            if parsed is not None:
                id = int(parsed[0])
                raw = mapping[rand_map[id] - 1]
                ids.append(raw)

    return ids


def read_kitti_cal(calfile):
    """
    Reads the kitti calibration projection matrix (p2) file from disc.

    Args:
        calfile (str): path to single calibration file
    """

    text_file = open(calfile, 'r')

    p2pat = re.compile(('(P2:)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)' +
                        '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat', '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'))

    for line in text_file:

        parsed = p2pat.fullmatch(line)

        if parsed is None:
            p2pat2 = re.compile(('(P2:)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)' + '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat', '[-+]?[\d]+\.?[\d]*'));
            parsed = p2pat2.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:
            p2 = np.zeros([4, 4], dtype=float)
            p2[0, 0] = parsed.group(2)
            p2[0, 1] = parsed.group(3)
            p2[0, 2] = parsed.group(4)
            p2[0, 3] = parsed.group(5)
            p2[1, 0] = parsed.group(6)
            p2[1, 1] = parsed.group(7)
            p2[1, 2] = parsed.group(8)
            p2[1, 3] = parsed.group(9)
            p2[2, 0] = parsed.group(10)
            p2[2, 1] = parsed.group(11)
            p2[2, 2] = parsed.group(12)
            p2[2, 3] = parsed.group(13)

            p2[3, 3] = 1

    text_file.close()

    return p2


def read_kitti_poses(posefile):

    text_file = open(posefile, 'r')

    ppat1 = re.compile(('(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)' +
                        '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat', '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'))

    ppat2 = re.compile(('(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)' +
                       '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat', '[-+]?[\d]+\.?[\d]*'));

    ps = []

    for line in text_file:

        parsed1 = ppat1.fullmatch(line)
        parsed2 = ppat2.fullmatch(line)

        if parsed1 is not None:
            p = np.zeros([4, 4], dtype=float)
            p[0, 0] = parsed1.group(1)
            p[0, 1] = parsed1.group(2)
            p[0, 2] = parsed1.group(3)
            p[0, 3] = parsed1.group(4)
            p[1, 0] = parsed1.group(5)
            p[1, 1] = parsed1.group(6)
            p[1, 2] = parsed1.group(7)
            p[1, 3] = parsed1.group(8)
            p[2, 0] = parsed1.group(9)
            p[2, 1] = parsed1.group(10)
            p[2, 2] = parsed1.group(11)
            p[2, 3] = parsed1.group(12)

            p[3, 3] = 1

            ps.append(p)

        elif parsed2 is not None:

            p = np.zeros([4, 4], dtype=float)
            p[0, 0] = parsed2.group(1)
            p[0, 1] = parsed2.group(2)
            p[0, 2] = parsed2.group(3)
            p[0, 3] = parsed2.group(4)
            p[1, 0] = parsed2.group(5)
            p[1, 1] = parsed2.group(6)
            p[1, 2] = parsed2.group(7)
            p[1, 3] = parsed2.group(8)
            p[2, 0] = parsed2.group(9)
            p[2, 1] = parsed2.group(10)
            p[2, 2] = parsed2.group(11)
            p[2, 3] = parsed2.group(12)

            p[3, 3] = 1

            ps.append(p)

    text_file.close()

    return ps


def read_kitti_label(file, p2, use_3d_for_2d=False):

    gts = []

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

    pattern = re.compile(('([a-zA-Z\-\?\_]+)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+'
                          + '(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s*((fpat)?)\n')
                         .replace('fpat', '[-+]?\d*\.\d+|[-+]?\d+'))


    for line in text_file:

        parsed = pattern.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:

            obj = edict()

            ign = False

            cls = parsed.group(1)
            trunc = float(parsed.group(2))
            occ = float(parsed.group(3))
            alpha = float(parsed.group(4))

            x = float(parsed.group(5))
            y = float(parsed.group(6))
            x2 = float(parsed.group(7))
            y2 = float(parsed.group(8))

            width = x2 - x + 1
            height = y2 - y + 1

            h3d = float(parsed.group(9))
            w3d = float(parsed.group(10))
            l3d = float(parsed.group(11))

            cx3d = float(parsed.group(12)) # center of car in 3d
            cy3d = float(parsed.group(13)) # bottom of car in 3d
            cz3d = float(parsed.group(14)) # center of car in 3d
            rotY = float(parsed.group(15))

            # store the elevation of car (should be ~ 1.65)
            elevation = cy3d

            # actually center the box
            cy3d -= (h3d / 2)

            if use_3d_for_2d and h3d > 0 and w3d > 0 and l3d > 0:

                # re-compute the 2D box using 3D (finally, avoids clipped boxes)
                verts3d, corners_3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

                # any boxes behind camera plane?
                if np.any(corners_3d[2, :] <= 0):
                    ign = True

                else:
                    x = min(verts3d[:, 0])
                    y = min(verts3d[:, 1])
                    x2 = max(verts3d[:, 0])
                    y2 = max(verts3d[:, 1])

                    width = x2 - x + 1
                    height = y2 - y + 1

            # project cx, cy, cz
            coord3d = p2.dot(np.array([cx3d, cy3d, cz3d, 1]))

            # store the projected instead
            cx3d_2d = coord3d[0]
            cy3d_2d = coord3d[1]
            cz3d_2d = coord3d[2]

            cx = cx3d_2d / cz3d_2d
            cy = cy3d_2d / cz3d_2d

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

            # label for axis (1 == horizontal, 2 == vertical!)
            axis_lbl = np.abs(np.sin(alpha)) < np.abs(np.cos(alpha))

            # use sin, more horizontal
            while alpha_sin > (math.pi / 2): alpha_sin -= math.pi
            while alpha_sin <= (-math.pi / 2): alpha_sin += math.pi

            # use cos, more vertical
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

            # is there an extra field? (assume to be track)
            if len(parsed.groups()) >= 16 and parsed.group(16).isdigit(): obj.track = int(parsed.group(16))

            obj.bbox_full = np.array([x, y, width, height])
            obj.bbox_3d = [cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY, elevation, alpha_sin, alpha_cos, axis_lbl, head_lbl]
            obj.center_3d = [cx3d, cy3d, cz3d]

            gts.append(obj)

    text_file.close()

    return gts


def balance_samples(conf, imdb):
    """
    Balances the samples in an image dataset according to the given configuration.
    Basically we check which images have relevant foreground samples and which are empty,
    then we compute the sampling weights according to a desired fg_image_ratio.

    This is primarily useful in datasets which have a lot of empty (background) images, which may
    cause instability during training if not properly balanced against.
    """

    sample_weights = np.ones(len(imdb))

    if conf.fg_image_ratio >= 0:

        empty_inds = []
        valid_inds = []

        for imind, imobj in enumerate(imdb):

            valid = 0

            scale = conf.test_scale / imobj.imH
            igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls, conf.min_gt_vis,
                                           conf.min_gt_h, conf.max_gt_h, scale)

            for gtind, gt in enumerate(imobj.gts):

                if (not igns[gtind]) and (not rmvs[gtind]):
                    valid += 1

            sample_weights[imind] = valid

            if valid>0:
                valid_inds.append(imind)
            else:
                empty_inds.append(imind)

        if not (conf.fg_image_ratio == 2):
            fg_weight = len(imdb) * conf.fg_image_ratio / len(valid_inds)
            bg_weight = len(imdb) * (1 - conf.fg_image_ratio) / len(empty_inds)
            sample_weights[valid_inds] = fg_weight
            sample_weights[empty_inds] = bg_weight

            logging.info('weighted respectively as {:.2f} and {:.2f}'.format(fg_weight, bg_weight))

        logging.info('Found {} foreground and {} empty images'.format(np.sum(sample_weights > 0), np.sum(sample_weights <= 0)))

    # force sampling weights to sum to 1
    sample_weights /= np.sum(sample_weights)

    return sample_weights


def compute_pose_stats(conf, imdb, cache_folder=''):

    if (cache_folder is not None) and os.path.exists(os.path.join(cache_folder, 'pose_means.pkl')) \
            and os.path.exists(os.path.join(cache_folder, 'pose_stds.pkl')):

        means = pickle_read(os.path.join(cache_folder, 'pose_means.pkl'))
        stds = pickle_read(os.path.join(cache_folder, 'pose_stds.pkl'))

    else:

        squared_sums = np.zeros([1, 6], dtype=np.float128)
        sums = np.zeros([1, 6], dtype=np.float128)

        class_counts = np.zeros([1], dtype=np.float128) + 1e-10

        # compute the mean first
        logging.info('Computing pose regression mean..')

        pose_codes = ['ego_10']

        for imind, imobj in enumerate(imdb):

            for pose_code in pose_codes:

                dx, dy, dz, rx, ry, rz = imobj[pose_code]

                sums[:, 0] += dx
                sums[:, 1] += dy
                sums[:, 2] += dz
                sums[:, 3] += rx
                sums[:, 4] += ry
                sums[:, 5] += rz

                class_counts += 1

        means = sums/class_counts

        logging.info('Computing pose regression stds..')

        for imobj in imdb:

            for pose_code in pose_codes:
                dx, dy, dz, rx, ry, rz = imobj[pose_code]

                squared_sums[:, 0] += np.power(dx - means[:, 0], 2)
                squared_sums[:, 1] += np.power(dy - means[:, 1], 2)
                squared_sums[:, 2] += np.power(dz - means[:, 2], 2)
                squared_sums[:, 3] += np.power(rx - means[:, 3], 2)
                squared_sums[:, 4] += np.power(ry - means[:, 4], 2)
                squared_sums[:, 5] += np.power(rz - means[:, 5], 2)

        stds = np.sqrt((squared_sums/class_counts))

        means = means.astype(float)
        stds = stds.astype(float)

        if (cache_folder is not None):
            pickle_write(os.path.join(cache_folder, 'pose_means.pkl'), means)
            pickle_write(os.path.join(cache_folder, 'pose_stds.pkl'), stds)

    conf.pose_means = means
    conf.pose_stds = stds


def scale_gts(imobj, scale_factor, key):

    # scale all coordinates
    for gtind, gt in enumerate(imobj[key]):

        if 'bbox_full' in imobj[key][gtind]:
            imobj[key][gtind].bbox_full *= scale_factor

        if 'bbox_vis' in imobj[key][gtind]:
            imobj[key][gtind].bbox_vis *= scale_factor

        if 'bbox_3d' in imobj[key][gtind]:
            # only scale x/y center locations (in 2D space!)
            imobj[key][gtind].bbox_3d[0] *= scale_factor
            imobj[key][gtind].bbox_3d[1] *= scale_factor