# -----------------------------------------
# python modules
# -----------------------------------------
from easydict import EasyDict as edict
from getopt import getopt
import numpy as np
import sys
import os

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.core import *
from lib.imdb_util import *
from lib.loss.rpn_3d import *


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

    # required opt
    if conf_name is None:
        raise ValueError('Please provide a configuration file name, e.g., --config=<config_name>')

    # -----------------------------------------
    # basic setup
    # -----------------------------------------

    conf = init_config(conf_name)
    paths = init_training_paths(conf_name)

    init_torch(conf.rng_seed, conf.cuda_seed)
    init_log_file(paths.logs)

    if 'copy_stats' in conf and conf.copy_stats and 'pretrained' in conf:
        copy_stats(paths.output, conf.pretrained)

    # defaults
    start_iter = 0
    tracker = edict()
    iterator = None

    dataset = Dataset(conf, paths.data, paths.output)

    generate_anchors(conf, dataset.imdb, paths.output)
    compute_bbox_stats(conf, dataset.imdb, paths.output)

    # -----------------------------------------
    # defaults mostly to False
    # -----------------------------------------
    conf.infer_2d_from_3d = False if not ('infer_2d_from_3d' in conf) else conf.infer_2d_from_3d
    conf.bbox_un_dynamic = False if not ('bbox_un_dynamic' in conf) else conf.bbox_un_dynamic

    # -----------------------------------------
    # store config
    # -----------------------------------------

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
    criterion_det = RPN_3D_loss(conf)

    # custom pretrained network
    if 'pretrained' in conf:
        load_weights(rpn_net, conf.pretrained)

    # resume training
    if restore:
        start_iter = (restore - 1)
        resume_checkpoint(optimizer, rpn_net, paths.weights, restore)

    freeze_blacklist = None if 'freeze_blacklist' not in conf else conf.freeze_blacklist
    freeze_whitelist = None if 'freeze_whitelist' not in conf else conf.freeze_whitelist

    freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist, verbose=True)
    if 'slow_bn' in conf and conf.slow_bn: slow_bn(rpn_net, conf.slow_bn)
    if 'freeze_bn' in conf and conf.freeze_bn: freeze_bn(rpn_net)

    optimizer.zero_grad()
    iteration = start_iter

    start_time = time()

    # -----------------------------------------
    # train
    # -----------------------------------------

    while iteration < conf.max_iter:

        # next iteration
        iterator, images, imobjs = next_iteration(dataset.loader, iterator)

        #  learning rate
        adjust_lr(conf, optimizer, iteration)

        # forward
        cls, prob, bbox_2d, bbox_3d, feat_size, rois, rois_3d, rois_3d_cen = rpn_net(images)

        # loss
        det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, rois, rois_3d, rois_3d_cen)

        loss_backprop(det_loss, rpn_net, optimizer, conf=conf, iteration=iteration)

        # keep track of stats
        compute_stats(tracker, det_stats)

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
            save_checkpoint(optimizer, rpn_net, paths.weights, (iteration + 1))

            if conf.do_test:

                # eval mode
                rpn_net.eval()

                # necessary paths
                results_path = os.path.join(paths.results, 'results_{}'.format((iteration + 1)))

                # -----------------------------------------
                # test kitti
                # -----------------------------------------
                if conf.test_protocol.lower() == 'kitti':

                    # delete and re-make
                    results_path = os.path.join(results_path, 'data')
                    mkdir_if_missing(results_path, delete_if_exist=True)

                    test_kitti_3d(conf.dataset_test, rpn_net, conf, results_path, paths.data)

                else:
                    logging.warning('Testing protocol {} not understood.'.format(conf.test_protocol))

                # train mode
                rpn_net.train()

                freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)
                if 'slow_bn' in conf and conf.slow_bn: slow_bn(rpn_net, conf.slow_bn)
                if 'freeze_bn' in conf and conf.freeze_bn: freeze_bn(rpn_net)

        iteration += 1

# run from command line
if __name__ == "__main__":
    main(sys.argv[1:])
