""" Model training function using keras.models.Model.fit_generator """

import logging

from ._keras_train import (
    config_session,
    log_training_args,
    log_sample_input,
    create_callback,
)


def train_model(
    model,
    generator,
    job_name=None,
    model_config=None,
    num_gpu=-1,
    epochs=50,
    steps_per_epoch=None,
    log_dir='./{job_name}/logs',
    tensorboard_dir='./{job_name}/logs',
    snapshot_dir='./{job_name}/snapshot',
    snapshot_name='model_{epoch:02d}.h5',
    snapshot=None,
    initial_epoch=0,
    workers=1,
    use_multiprocessing=False
):
    """ Trains a model using a batch generator and logs information using logging.info

    This function does not initialize your logging configs, initialize it before starting this function

    Args
        model     : A keras.models.Model object
        generator : A generator which generates batches of model inputs and outputs

        job_name     : Name of this training job (no name if None)
        model_config : Model configs in dict form to log as well as to save (no logging if None)

        num_gpu         : Number of GPUs to train model on (-1 sets to keras default)
        epochs          : Number of epoches to train
        steps_per_epoch : Number of steps per epoch

        log_dir         : Directory to store log file and misc outputs (Set to None for no outputs)
        tensorboard_dir : Directory to store tensorboard (Set to None for no outputs)
        snapshot_dir    : Directory to store training snapshots (Set to None for no outputs)
        snapshot_name   : Name of snapshot files (follows https://keras.io/callbacks/#ModelCheckpoint convention)

        snapshot      : A h5 model file to resume training from
        initial_epoch : Epoch at which to start training

        workers             : Number of workers to generate training data with
        use_multiprocessing : Use process based threading
    """
    # We config directories with job name
    if not job_name:
        job_name = ''

    for k in ['log_dir', 'tensorboard_dir', 'snapshot_dir']:
        dir = locals()[k]
        if dir:
            locals()[k] = os.path.absath(dir.format(job_name=args['job_name']))

    logging.info('==================== Initializing Training Job ====================')

    config_session()
    log_training_args(args=locals())
    batch_size = log_sample_input(generator, log_dir)
    callbacks = create_callback(
        model,
        batch_size,
        log_dir,
        tensorboard_dir,
        snapshot_dir,
        snapshot_name
    )

    logging.info('')
    logging.info('==================== Training Start ====================')

    model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        workers=workers,
        use_multiprocessing=use_multiprocessing
    )

    logging.info('')
    logging.info('==================== Training Done ====================')
