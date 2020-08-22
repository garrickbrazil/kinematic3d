# -----------------------------------------
# python modules
# -----------------------------------------
from easydict import EasyDict as edict
from getopt import getopt
import sys
import numpy as np
import torch

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.core import *
from lib.imdb_util import *
from lib.loss.rpn_3d import *
from lib.loss.pose import *


def main(argv):

    # -----------------------------------------
    # parse arguments
    # -----------------------------------------
    opts, args = getopt(argv, '', ['config=', 'restore='])

    # defaults
    conf_name = None
    restore = None

    # read opts
    for opt, arg in opts:

        if opt in ('--config'): conf_name = arg
        if opt in ('--restore'): restore = int(arg)


    # -----------------------------------------
    # basic setup
    # -----------------------------------------

    conf = init_config(conf_name)
    paths = init_training_paths(conf_name)

    init_torch(conf.rng_seed, conf.cuda_seed)
    init_log_file(paths.logs)

    start_iter = 0
    tracker = edict()
    iterator = None

    if 'copy_stats' in conf and conf.copy_stats and 'pretrained' in conf:
        copy_stats(paths.output, conf.pretrained)

    # -----------------------------------------
    # data and anchors
    # -----------------------------------------

    dataset = Dataset(conf, paths.data, paths.output)

    generate_anchors(conf, dataset.imdb, paths.output)
    compute_bbox_stats(conf, dataset.imdb, paths.output)
    compute_pose_stats(conf, dataset.imdb, paths.output)

    # store configuration
    pickle_write(os.path.join(paths.output, 'conf.pkl'), conf)

    # show configuration
    pretty = pretty_print('conf', conf)
    logging.info(pretty)

    # -----------------------------------------
    # network and loss
    # -----------------------------------------

    # training network
    rpn_net, optimizer = init_training_model(conf, paths.output)

    # setup loss
    criterion_pose = Pose_loss(conf, verbose=False)

    # copy from pretrained network
    if 'pretrained' in conf:
        load_weights(rpn_net, conf.pretrained)

    # resume training
    if restore:
        start_iter = (restore - 1)
        resume_checkpoint(optimizer, rpn_net, paths.weights, restore)

    freeze_blacklist = None if 'freeze_blacklist' not in conf else conf.freeze_blacklist
    freeze_whitelist = None if 'freeze_whitelist' not in conf else conf.freeze_whitelist

    freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist, verbose=True)
    if ('freeze_bn' in conf and conf.freeze_bn): freeze_bn(rpn_net)

    optimizer.zero_grad()

    start_time = time()

    bool_type = torch.cuda.ByteTensor if not hasattr(torch, 'bool') else torch.cuda.BoolTensor

    # -----------------------------------------
    # train
    # -----------------------------------------

    for iteration in range(start_iter, conf.max_iter):

        # next iteration
        iterator, images, imobjs = next_iteration(dataset.loader, iterator)

        imobjs = np.array(imobjs)

        #  learning rate
        adjust_lr(conf, optimizer, iteration)

        if ('use_all_tracks' in conf) and conf.use_all_tracks:
            conf.video_count = 4 if imobjs[0].has_track else 1
            rpn_net.video_count = conf.video_count
            dataset.video_count = conf.video_count

        # gather projection matrices and scale factors
        p2s = None
        p2_invs = None
        scales = []
        for imobj in imobjs:

            if p2s is None: p2s = imobj.p2[np.newaxis, :, :]
            else: p2s = np.concatenate((p2s, imobj.p2[np.newaxis, :, :]), axis=0)

            if p2_invs is None: p2_invs = imobj.p2_inv[np.newaxis, :, :]
            else: p2_invs = np.concatenate((p2_invs, imobj.p2_inv[np.newaxis, :, :]), axis=0)

            scales.append(imobj.scale_factor)

        # forward
        si_shots, tr_shots, poses = rpn_net(images, p2s, p2_invs, scales)

        pose_loss, pose_stats = criterion_pose(poses[-1], imobjs, 'ego_10')
        total_loss = pose_loss
        stats = pose_stats

        # backprop
        if total_loss > 0:

            total_loss.backward()

            if (not 'batch_skip' in conf) or ((iteration + 1) % conf.batch_skip) == 0:

                if ('clip_grad' in conf) and conf.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(rpn_net.parameters(), conf.clip_grad)

                optimizer.step()
                optimizer.zero_grad()

        # keep track of stats
        compute_stats(tracker, stats)

        # -----------------------------------------
        # display
        # -----------------------------------------
        if (iteration + 1) % conf.display == 0 and iteration > start_iter:

            # log results
            log_stats(tracker, iteration, start_time, start_iter, conf.max_iter)

            # reset tracker
            tracker = edict()

        # -----------------------------------------
        # test network
        # -----------------------------------------
        if (iteration + 1) % conf.snapshot_iter == 0 and iteration > start_iter:

            # store checkpoint
            modelpath, optimpath = save_checkpoint(optimizer, rpn_net, paths.weights, (iteration + 1))

            if conf.do_test:

                if ('use_all_tracks' in conf) and conf.use_all_tracks:
                    conf.video_count = 4
                    rpn_net.video_count = conf.video_count
                    dataset.video_count = conf.video_count

                # eval mode
                rpn_net.eval()

                # necessary paths
                results_path = os.path.join(paths.results, 'results_{}'.format((iteration + 1)))

                # -----------------------------------------
                # test kitti
                # -----------------------------------------
                if conf.test_protocol.lower() == 'kitti':

                    # delete and re-make
                    results_path_pre = os.path.join(results_path + '_pre', 'data')
                    results_path_cur = os.path.join(results_path + '_cur', 'data')
                    results_path_ego = os.path.join(results_path + '_ego', 'data')

                    mkdir_if_missing(results_path_pre, delete_if_exist=True)
                    mkdir_if_missing(results_path_cur, delete_if_exist=True)
                    mkdir_if_missing(results_path_ego, delete_if_exist=True)

                    test_kitti_3d_forecast(conf.dataset_test, rpn_net, conf, results_path_pre, results_path_cur,
                                           results_path_ego, paths.data)

                else:
                    logging.warning('Testing protocol {} not understood.'.format(conf.test_protocol))

                # train mode
                rpn_net.train()

                freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)
                if ('freeze_bn' in conf and conf.freeze_bn): freeze_bn(rpn_net)


# run from command line
if __name__ == "__main__":
    main(sys.argv[1:])
