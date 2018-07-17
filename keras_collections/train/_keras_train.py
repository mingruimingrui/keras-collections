""" Utility functions for keras_train.py """

import os
import json
import pickle
import logging
import numpy as np
import keras


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_dict(dict_obj, file_path):
    # Identify items that annot be converted to json
    keys_to_delete = []
    for key, value in dict_obj.items():
        try:
            json.dumps(value)
        except TypeError:
            keys_to_delete.append(key)

    # Delete items
    if len(keys_to_delete) > 0:
        logging.warn('The objects {} cannot be converted to json'.format(keys_to_delete))
        for key in keys_to_delete:
            del dict_obj[key]

    # Save
    with open(file_path, 'w') as f:
        json.dump(dict_obj, f, indent=4)


def config_session():
    session_config = tf.ConfigProto()

    # Allow growth
    session_config.gpu_options.allow_growth = True
    logging.info('Graph set to allow growth')

    # Set config
    current_session = tf.Session(config=session_config)
    keras.backend.tensorflow_backend.set_session(current_session)


def log_training_args(args):
    if not args['log_dir']:
        logging.warn('logging directory not provided, no training byproducts will be created')
    if not args['tensorboard_dir']:
        logging.warn('tensorboard directory not provided, no tensorboard objects will be created')
    if not args['snapshot_dir']:
        logging.warn('snapshot_dir not found, no training snapshots will be saved')
    if args['snapshot']:
        logging.info('Previous model found, training will continue from {}'.format(args['snapshot']))

    logging.info('')
    logging.info('==================== General Info ====================')
    if args['job_name']:
        logging.info('job_name   : {}'.format(args['job_name']))
    logging.info('model_name : {}'.format(args['model'].name))

    logging.info('')
    logging.info('==================== Logging Info ====================')
    if args['log_dir']:
        logging.info('log_dir         : {}'.format(args['log_dir']))
    if args['tensorboard_dir']:
        logging.info('tensorboard_dir : {}'.format(args['tensorboard_dir']))
    if args['snapshot_dir']:
        logging.info('snapshot_dir    : {}'.format(args['snapshot_dir']))
    if (not args['log_dir']) and (not args['tensorboard_dir']) and (not args['snapshot_dir']):
        logging.warn('No logs will be saved')

    logging.info('')
    logging.info('==================== Training Info ====================')
    logging.info('num_gpu         : {}'.format(args['num_gpu']))
    logging.info('epochs          : {}'.format(args['epochs']))
    logging.info('steps_per_epoch : {}'.format(args['steps_per_epoch']))

    if args['log_dir'] and args['model_config']:
        makedirs(args['log_dir'])
        model_config_path = os.path.join(args['log_dir'], 'model_config.json')
        model_config_path = os.path.abspath(model_config_path)
        save_dict(args['model_config'], model_config_path)
        logging.info('')
        logging.info('==================== Model Config Info ====================')
        logging.info('model_config_json_file : {}'.format(model_config_path))


def log_sample_input(generator, log_dir):
    # Get entry from generator
    entry = next(generator)
    assert len(entry) == 2, 'generator should output a tuple of len 2, got {}'.format(len(entry))

    # Get inputs and outputs
    inputs  = entry[0]
    outputs = entry[1]

    # Identify batch size
    if isinstance(inputs, np.ndarray):
        batch_size = inputs.shape[0]
    else:
        batch_size = inputs[0].shape[0]

    if log_dir:
        makedirs(log_dir)
        with open(os.path.join('log_dir', 'sample_input.pkl'), 'wb') as f:
            pickle.dump({
                'inputs' : inputs,
                'outputs': outputs
            }, f)

    return batch_size


def create_callback(
    model,
    batch_size,
    log_dir,
    tensorboard_dir,
    snapshot_dir,
    snapshot_name
):
    logging.info('')
    logging.info('==================== Making Callbacks ====================')
    logging.info('This can take a while')

    callbacks = []

    # Create loss logger
    if log_dir:
        from ..callbacks._misc import ProgressLogger
        makedirs(log_dir)
        train_progress_file = os.path.join(log_dir, 'train.log')
        progress_callback = ProgressLogger(log_path=train_progress_file, stdout=True)
        callbacks.append(progress_callback)

    # Create tensorboard
    if tensorboard_dir:
        makedirs(tensorboard_dir)
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = batch_size,
            write_graph            = True,
            write_grads            = True,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    # Save model
    if snapshot_dir:
        makedirs(snapshot_dir)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(snapshot_dir, snapshot_name),
            verbose=1,
            save_weights_only=False,
        )
        callbacks.append(checkpoint)

    # Plateau LR
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor   = 'loss',
        factor    = 0.1,
        patience  = 2,
        verbose   = 1,
        mode      = 'auto',
        min_delta = 1e-4,
        cooldown  = 0,
        min_lr    = 0
    ))

    logging.info('Callbacks created')

    return callbacks
