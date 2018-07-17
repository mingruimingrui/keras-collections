from __future__ import division

import os
import math
import time
import datetime

import keras


class RedirectModel(keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.
    # Arguments
        callback: callback to wrap.
        model: model to use when executing callbacks.
    # Example
        ```python
        model = keras.models.load_model('model.h5')
        model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
        ```
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)


class ProgressLogger(keras.callbacks.Callback):
    """ Logs the training progress into a log file and optionally also to stdout
    # Arguments
        log_path : Path to log file, ensure that directory is already created
        std_out  : Flag to indicate if progress should be logged to stdout
    """
    def __init__(self, log_path, stdout=False):
        super(ProgressLogger, self).__init__()

        log_dir = os.path.dirname(log_path)
        assert os.path.isdir(log_dir) or log_dir == '', '{} does not exist'.format(log_dir)

        self.log_path = log_path
        self.stdout = stdout
        self.width = 40

    def _format_time(self, seconds):
        return str(datetime.timedelta(seconds=int(seconds)))

    def _log_progress_message(self, logs, cur_step):
        cur_step += 1

        # Make progress bar message
        ratio_done = cur_step / self.total_steps
        width_done = int(self.width * ratio_done)
        pbar_msg = '|{}{}|  {}/{}  {:.1f}%'.format(
            '#' * width_done,
            ' ' * (self.width - width_done),
            cur_step, self.total_steps,
            100 * ratio_done,
        )

        # Make loss message
        loss_msgs = []
        for k in self.params['metrics']:
            if k in logs:
                loss_msgs.append('{}: {:.3f}'.format(k, logs[k]))
        loss_msg = ' - '.join(loss_msgs)

        # Calc time to completion
        time_elapsed = time.time() - self.start_time
        steps_elapsed = cur_step - self.initial_step
        steps_to_go = self.total_steps - cur_step
        time_to_go = time_elapsed / steps_elapsed * steps_to_go
        time_msg = '[{}<{}]'.format(
            self._format_time(time_elapsed),
            self._format_time(time_to_go)
        )

        msg = '{}  {} {}'.format(pbar_msg, loss_msg, time_msg)

        # Log to file and to stdout
        self.log_file.write('{}\n'.format(msg))
        if self.stdout:
            print(msg)

    def on_train_begin(self, logs=None):
        # Create log file
        self.log_file = open(self.log_path, 'w')
        if self.params['steps'] is None:
            self.steps_per_epoch = math.ceil(self.params['samples'] / self.params['batch_size'])
        else:
            self.steps_per_epoch = self.params['steps']
        self.total_steps = self.steps_per_epoch * self.params['epochs']
        self.gap = math.ceil(self.total_steps / 1000)
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        # Close log file
        self._log_progress_message(logs, self.total_steps - 1)
        self.log_file.close()

    def on_epoch_begin(self, epoch, logs=None):
        self.cur_epoch = epoch
        if not hasattr(self, 'initial_step'):
            self.initial_step = epoch * self.steps_per_epoch

    def on_batch_end(self, batch, logs=None):
        # Calc current step
        cur_step = self.steps_per_epoch * self.cur_epoch + batch

        # Determine if logging is to be done at this step
        if cur_step % self.gap == 0:
            self._log_progress_message(logs, cur_step)
