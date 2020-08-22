"""
This file contains all PyTroch data augmentation functions.

Every transform should have a __call__ function which takes in (self, image, imobj)
where imobj is an arbitary dict containing relevant information to the image.

In many cases the imobj can be None, which enables the same augmentations to be used
during testing as they are in training.

Optionally, most transforms should have an __init__ function as well, if needed.
"""

import numpy as np
import cv2
import math
from lib.util import *
from lib.math_3d import *
import torchvision
from PIL import Image

class Compose(object):
    """
    Composes a set of functions which take in an image and an object, into a single transform
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, imobj=None):
        for t in self.transforms:
            img, imobj = t(img, imobj)
        return img, imobj


class ConvertToFloat(object):
    """
    Converts image data type to float.
    """
    def __call__(self, image, imobj=None):
        return image.astype(np.float32), imobj


class Normalize(object):
    """
    Normalize the image
    """
    def __init__(self, mean, stds):
        self.mean = np.array(mean, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    def __call__(self, image, imobj=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= np.tile(self.mean, int(image.shape[2]/self.mean.shape[0]))
        image /= np.tile(self.stds, int(image.shape[2]/self.stds.shape[0]))
        return image.astype(np.float32), imobj


class Resize(object):
    """
    Resize the image according to the target size height and the image height.
    If the image needs to be cropped after the resize, we crop it to self.size,
    otherwise we pad it with zeros along the right edge

    If the object has ground truths we also scale the (known) box coordinates.
    """
    def __init__(self, size):
        self.size = size

    def scale_gts(self, imobj, scale_factor, key):

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


    def __call__(self, image, imobj=None):

        scale_factor = self.size[0] / image.shape[0]

        h = np.round(image.shape[0] * scale_factor).astype(int)
        w = np.round(image.shape[1] * scale_factor).astype(int)

        # resize
        image = cv2.resize(image, (w, h))

        if len(self.size) > 1:

            # crop in
            if image.shape[1] > self.size[1]:
                image = image[:, 0:self.size[1], :]

            # pad out
            elif image.shape[1] < self.size[1]:
                padW = self.size[1] - image.shape[1]
                image = np.pad(image, [(0, 0), (0, padW), (0, 0)], 'constant')

        if imobj:

            # store scale factor, just in case
            imobj.scale_factor = scale_factor

            if 'gts' in imobj: self.scale_gts(imobj, scale_factor, 'gts')
            if 'gts_1' in imobj: self.scale_gts(imobj, scale_factor, 'gts_1')
            if 'gts_2' in imobj: self.scale_gts(imobj, scale_factor, 'gts_2')
            if 'gts_3' in imobj: self.scale_gts(imobj, scale_factor, 'gts_3')

        return image, imobj


class RandomSaturation(object):
    """
    Randomly adjust the saturation of an image given a lower and upper bound,
    and a distortion probability.

    This function assumes the image is in HSV!!
    """
    def __init__(self, distort_prob, lower=0.5, upper=1.5):

        self.distort_prob = distort_prob
        self.lower = lower
        self.upper = upper

        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, imobj=None):
        if np.random.rand() <= self.distort_prob:
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        return image, imobj


class RandomHue(object):
    """
    Randomly adjust the hue of an image given a delta degree to rotate by,
    and a distortion probability.

    This function assumes the image is in HSV!!
    """
    def __init__(self, distort_prob, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
        self.distort_prob = distort_prob

    def __call__(self, image, imobj=None):
        if np.random.rand() <= self.distort_prob:
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, imobj


class ConvertColor(object):
    """
    Converts color spaces to/from HSV and BGR
    """
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, imobj=None):

        # BGR --> HSV
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV --> BGR
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        else:
            raise NotImplementedError

        return image, imobj


class RandomContrast(object):
    """
    Randomly adjust contrast of an image given lower and upper bound,
    and a distortion probability.
    """
    def __init__(self, distort_prob, lower=0.5, upper=1.5):

        self.lower = lower
        self.upper = upper
        self.distort_prob = distort_prob

        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, imobj=None):
        if np.random.rand() <= self.distort_prob:
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, imobj


class RandomMirror(object):
    """
    Randomly mirror an image horzontially, given a mirror probabilty.

    Also, adjust all box cordinates accordingly.
    """
    def __init__(self, mirror_prob):
        self.mirror_prob = mirror_prob

    def flip_gts(self, image, imobj, key):

        # flip the coordinates w.r.t the horizontal flip (only adjust X)
        for gtind, gt in enumerate(imobj[key]):

            if 'bbox_full' in imobj[key][gtind]:
                imobj[key][gtind].bbox_full[0] = image.shape[1] - gt.bbox_full[0] - gt.bbox_full[2]

            if 'bbox_vis' in imobj[key][gtind]:
                imobj[key][gtind].bbox_vis[0] = image.shape[1] - gt.bbox_vis[0] - gt.bbox_vis[2]

            if 'bbox_3d' in imobj[key][gtind]:
                imobj[key][gtind].bbox_3d[0] = image.shape[1] - gt.bbox_3d[0] - 1
                rotY = gt.bbox_3d[10]

                rotY = (-math.pi - rotY) if rotY < 0 else (math.pi - rotY)

                while rotY > math.pi: rotY -= math.pi * 2
                while rotY < (-math.pi): rotY += math.pi * 2

                cx2d = gt.bbox_3d[0]
                cy2d = gt.bbox_3d[1]
                cz2d = gt.bbox_3d[2]

                coord3d = imobj.p2_inv.dot(np.array([cx2d * cz2d, cy2d * cz2d, cz2d, 1]))

                alpha = convertRot2Alpha(rotY, coord3d[2], coord3d[0])

                # [cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY, elevation, alpha_sin, alpha_cos, axis_lbl]
                # label for axis
                axis_lbl = np.abs(np.sin(alpha)) < np.abs(np.cos(alpha))

                alpha_sin = alpha
                alpha_cos = alpha

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

                imobj[key][gtind].bbox_3d[10] = rotY
                imobj[key][gtind].bbox_3d[6] = alpha
                imobj[key][gtind].bbox_3d[12] = alpha_sin
                imobj[key][gtind].bbox_3d[13] = alpha_cos
                imobj[key][gtind].bbox_3d[14] = axis_lbl
                imobj[key][gtind].bbox_3d[15] = head_lbl
                imobj[key][gtind].bbox_3d[7:10] = coord3d[:3]
                imobj[key][gtind].center_3d = coord3d[:3]


    def __call__(self, image, imobj):

        _, width, _ = image.shape

        if np.random.rand() <= self.mirror_prob:

            image = image_mirror(image)

            if 'ego_32' in imobj: imobj.ego_32 = ego_mirror(imobj.ego_32)
            if 'ego_31' in imobj: imobj.ego_31 = ego_mirror(imobj.ego_31)
            if 'ego_30' in imobj: imobj.ego_30 = ego_mirror(imobj.ego_30)
            if 'ego_21' in imobj: imobj.ego_21 = ego_mirror(imobj.ego_21)
            if 'ego_20' in imobj: imobj.ego_20 = ego_mirror(imobj.ego_20)
            if 'ego_10' in imobj: imobj.ego_10 = ego_mirror(imobj.ego_10)

            if 'gts' in imobj: self.flip_gts(image, imobj, 'gts')
            if 'gts_1' in imobj: self.flip_gts(image, imobj, 'gts_1')
            if 'gts_2' in imobj: self.flip_gts(image, imobj, 'gts_2')
            if 'gts_3' in imobj: self.flip_gts(image, imobj, 'gts_3')

        return image, imobj


class RandomBrightness(object):
    """
    Randomly adjust the brightness of an image given given a +- delta range,
    and a distortion probability.
    """
    def __init__(self, distort_prob, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        self.distort_prob = distort_prob

    def __call__(self, image, imobj=None):
        if np.random.rand() <= self.distort_prob:
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, imobj


class PhotometricDistort(object):
    """
    Packages all photometric distortions into a single transform.
    """
    def __init__(self, distort_prob):

        self.distort_prob = distort_prob

        # contrast is duplicated because it may happen before or after
        # the other transforms with equal probability.
        '''
        self.transforms = [
            RandomContrast(distort_prob),
            ConvertColor(transform='HSV'),
            RandomSaturation(distort_prob),
            RandomHue(distort_prob),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast(distort_prob)
        ]
        '''

        self.transform = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

        #self.rand_brightness = RandomBrightness(distort_prob)

    def __call__(self, image, imobj):

        '''
        # do contrast first
        if np.random.rand() <= -1:
            distortion = self.transforms[:-1]

        # do contrast last
        else:
            distortion = self.transforms[1:]

        # add random brightness
        distortion.insert(0, self.rand_brightness)

        # compose transformation
        distortion = Compose(distortion)
        im_new, imobj = distortion(image.copy(), imobj)
        '''

        if np.random.rand() <= self.distort_prob:

            im_new = Image.fromarray(np.uint8(image))
            im_new = self.transform(im_new)
            image = np.array(im_new).clip(0, 255)

        return image, imobj


class Augmentation(object):
    """
    Data Augmentation class which packages the typical pre-processing
    and all data augmentation transformations (mirror and photometric distort)
    into a single transform.
    """
    def __init__(self, conf):

        self.mean = conf.image_means
        self.stds = conf.image_stds
        self.size = conf.crop_size
        self.mirror_prob = conf.mirror_prob
        self.distort_prob = conf.distort_prob

        if conf.distort_prob <= 0:
            self.augment = Compose([
                ConvertToFloat(),
                RandomMirror(self.mirror_prob),
                Resize(self.size),
                Normalize(self.mean, self.stds)
            ])
        else:
            self.augment = Compose([
                ConvertToFloat(),
                PhotometricDistort(self.distort_prob),
                RandomMirror(self.mirror_prob),
                Resize(self.size),
                Normalize(self.mean, self.stds)
            ])

    def __call__(self, img, imobj):
        return self.augment(img, imobj)


class Preprocess(object):
    """
    Preprocess function which ONLY does the basic pre-processing of an image,
    meant to be used during the testing/eval stages.
    """
    def __init__(self, size, mean, stds):

        self.mean = mean
        self.stds = stds
        self.size = size

        self.preprocess = Compose([
            ConvertToFloat(),
            Resize(self.size),
            Normalize(self.mean, self.stds)
        ])

    def __call__(self, img):

        img = self.preprocess(img)[0]

        for i in range(int(img.shape[2]/3)):

            # convert to RGB then permute to be [B C H W]
            img[:, :, (i*3):(i*3) + 3] = img[:, :, (i*3+2, i*3+1, i*3)]

        img = np.transpose(img, [2, 0, 1])

        return img
