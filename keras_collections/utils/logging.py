import os
import logging


def set_default_logging(log_path=None):
    """ Configs logging to the following settings
    - level set to INFO
    - logs saved to file and output to stdout
    - format in log file has the heading %(asctime)s [%(levelname)-4.4s]

    Args
        log_path : path to log file
    """
    # Make log path an abs path
    log_path = os.path.abspath(log_path)

    # Log to file
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s [%(levelname)-4.4s] %(message)s',
        datefmt='%m-%d %H:%M',
        level=logging.INFO
    )

    # Log to stdout
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('logging will be automatically saved to {}'.format(log_path))
