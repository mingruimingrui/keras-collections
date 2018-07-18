from __future__ import division

import os
import tqdm

import numpy as np
from PIL import Image

from ..utils.anchors import compute_overlap
from ..utils.visualization import draw_detections


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args
        recall:    The recall curve (list).
        precision: The precision curve (list).
    Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_annotations_and_detections(
    generator, model,
    score_threshold=0.05,
    max_detections=100,
    max_images=None,
    max_plots=5,
    save_dir=None,
    label_to_name=None
):
    """ Get the annotations and detections from the model using the generator.

    Annotations (and detections) are list of lists in the format
        annotations[num_images][num_classes] = annotations[num_annotations, 4 + label]
        annotations[num_images][num_classes] = annotations[num_annotations, 4 + score + label]

    Args
        generator       : Generator for your dataset
        model           : Model to perform detection
        score_threshold : Score threshold used for detection
        max_detections  : Max number of detections to use per image
        max_images      : Max number of images to extract
        max_plots       : Max number of images to visualize with detections
        save_dir       : Path to save images with visualized detections

    Returns

    """
    # Get number of images to extract on
    if max_images is None:
        num_images = len(generator)
    else:
        num_images = min(max_images, len(generator))

    # Create blob to store annotations and detections
    all_annotations = [[None for label in range(generator.num_classes)] for i in range(num_images)]
    all_detections  = [[None for label in range(generator.num_classes)] for i in range(num_images)]

    # Create progress bar
    pbar = tqdm.tqdm(total=num_images, desc='Getting annotations and detections')

    for i in range(num_images):
        pbar.update(1)
        image_index = generator.all_image_index[i]

        # load group X and Y
        X_group = generator.load_X_group([image_index])
        Y_group = generator.load_Y_group([image_index])

        # Get original image and annotations
        image       = X_group[0]
        annotations = Y_group[0]

        # Get model input and scale
        image_input, image_scale = generator.resize_image(image.copy())
        image_input = np.expand_dims(image_input, 0)

        # Force correct types
        image = image.astype('uint8')

        # Perform predictions
        boxes, scores, labels = model.predict(image_input)

        # Correct boxes for scale
        boxes /= image_scale

        # Select scores above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]
        scores = scores[0][indices]

        # Find the order to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # Select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([
            image_boxes,
            np.expand_dims(image_scores, axis=1),
            np.expand_dims(image_labels, axis=1)
        ], axis=1)

        # Save detections if necessary
        if (save_dir is not None) and (i < max_plots):
            image = image.copy()

            draw_detections(image, image_boxes, image_scores, image_labels,
                label_to_name=label_to_name, score_threshold=score_threshold)

            save_path = os.path.join(save_dir, 'detections_{}.png'.format(image_index))
            Image.fromarray(image).save(save_path)

        for label in range(generator.num_classes):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
            all_detections[i][label]  = image_detections[image_detections[:, -1] == label, :-1].copy()

    pbar.close()

    return all_annotations, all_detections

def evaluate_detection(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    max_images=None,
    max_plots=5,
    save_dir=None,
    label_to_name=None
):
    """ Evaluate a detection model on a dataset
    At present evaluation is working only at batch sizes of 1

    Args
        generator       : Generator for your dataset
        model           : Model to evaluate
        iou_threshold   : Threshold used to consider if detection is positive or negative
        score_threshold : Score threshold used for detection
        max_detections  : Max number of detections to use per image
        max_images      : Max number of images to evaluate on (if None will evaluate on entire dataset)
        max_plots       : Max number of images to visualize with detections
        save_dir        : Directory to save images with visualized detections

    Returns
        A dict containing AP scores for each class

    """
    # Create blob for average_precision
    average_precisions = {}

    # Gather all detections and annotations
    all_annotations, all_detections = _get_annotations_and_detections(
        generator, model,
        score_threshold=score_threshold,
        max_detections=max_detections,
        max_images=max_images,
        max_plots=max_plots,
        save_dir=save_dir,
        label_to_name=label_to_name
    )

    # Record number of images to be used in evaluation
    num_images = len(all_annotations)

    for label in range(generator.num_classes):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.


        for i in range(num_images):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += len(annotations)
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # If no annotations then AP will be 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision


    return average_precisions
