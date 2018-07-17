import keras
from ..evaluation.eval import evaluate_detection


class EvaluateDetection(keras.callbacks.Callback):
    def __init__(
        self, generator,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        max_images=None,
        max_plots=5,
        save_path=None,
        tensorboard=None,
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            max_images      : The maximum number of images to evaluate on (if None will evaluate on entire dataset).
            max_plots       : The maximum number of images to visualize with detections.
            save_path       : The path to save images with visualized detections to.
            tensorboard     : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            verbose         : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.max_images      = max_images,
        self.max_plots       = max_plots,
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.verbose         = verbose

        super(EvaluateDetection, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions = evaluate_detection(
            self.generator, self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            max_images=self.max_images,
            max_plots=self.max_plots,
            save_path=self.save_path
        )

        self.mean_ap = sum(average_precisions.values()) / len(average_precisions)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.mean_ap
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            for label, average_precision in average_precisions.items():
                print(self.generator.label_to_name(label), '{:.4f}'.format(average_precision))
            print('mAP: {:.4f}'.format(self.mean_ap))
