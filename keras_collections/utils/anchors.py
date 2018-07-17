import numpy as np


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size
    This is for ordinary np.array
    """

    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    targets = targets.T

    targets = (targets - mean) / std

    return targets


def generate_anchors(base_size=16, ratios=None, scales=None):
    """ Generate anchors based on a size a set of ratios and scales
    w.r.t a reference window
    """

    if ratios is None:
        ratios = np.array([0.5, 1., 2.])
    else:
        ratios = np.array(ratios)

    if scales is None:
        scales = np.array([2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)])
    else:
        scales = np.array(scales)

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def compute_all_anchors(
    image_shape,
    sizes           = [8, 16, 32, 64, 128],
    strides         = [32, 64, 128, 256, 512],
    ratios          = [0.5, 1., 2.],
    scales          = [2. ** 0., 2. ** (1. / 3.), 2 ** (2. / 3.)],
    shapes_callback = None,
):
    """ Generate all anchors based on image_shape as well as anchor configs and pyramid_levels

    Args
        image_shape     : (height, width) of an image
        sizes           : List of sizes to use. Each size corresponds to one feature level
        strides         : List of strides to use. Each stride corresponds to one feature level
        ratios          : List of ratios to use per location in a feature map
        scales          : List of scales to use per location in a feature map
        shapes_callback : A function that calculates the pyramid_feature_shapes given an image_shape

    Returns
        All anchors for image_shape

    """

    assert len(sizes) == len(strides), 'length of sizes must be same as strides'

    if shapes_callback is None:
        def guess_shapes(image_shape):
            image_shape = np.array(image_shape[:2])
            image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in [3, 4, 5, 6, 7]]
            return image_shapes

        shapes_callback = guess_shapes

    image_shapes = shapes_callback(image_shape)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx in range(len(sizes)):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def anchor_targets_bbox(
    anchors,
    annotations,
    num_classes,
    mask_shape=None,
    negative_overlap=0.4,
    positive_overlap=0.5,
    **kwargs
):
    """ Generate anchor targets for bbox detection.
    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).
    Returns
        labels: np.array of shape (A, num_classes) where a row consists of 0 for negative and 1 for positive for a certain class.
        annotations: np.array of shape (A, 5) for (x1, y1, x2, y2, label) containing the annotations corresponding to each anchor or 0 if there is no corresponding anchor.
        anchor_states: np.array of shape (N,) containing the state of an anchor (-1 for ignore, 0 for bg, 1 for fg).
    """
    # anchor states: 1 is positive, 0 is negative, -1 is dont care
    anchor_states = np.zeros((anchors.shape[0],))
    labels        = np.zeros((anchors.shape[0], num_classes))

    if annotations.shape[0]:
        # obtain indices of gt annotations with the greatest overlap
        overlaps             = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps         = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

        # assign "dont care" labels
        positive_indices                = max_overlaps >= positive_overlap
        ignore_indices                  = (max_overlaps > negative_overlap) & ~positive_indices
        anchor_states[ignore_indices]   = -1
        anchor_states[positive_indices] = 1

        # compute box regression targets
        annotations = annotations[argmax_overlaps_inds]

        # compute target class labels
        labels[positive_indices, annotations[positive_indices, 4].astype(int)] = 1
    else:
        # no annotations? then everything is background
        annotations = np.zeros((anchors.shape[0], annotations.shape[1]))

    # ignore annotations outside of image
    if mask_shape:
        anchors_centers        = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
        indices                = np.logical_or(anchors_centers[:, 0] >= mask_shape[1], anchors_centers[:, 1] >= mask_shape[0])
        anchor_states[indices] = -1

    return labels, annotations, anchor_states
