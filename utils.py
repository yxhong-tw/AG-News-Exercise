import logging
import torch
import os

from tabulate import tabulate


logger = logging.getLogger(__name__)


def initialize_logger(config):
    AGNews_path = config['AGNews_path']
    logger_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logger_formatter)

    fh = logging.FileHandler(
        filename=f'{AGNews_path}/logs/{config["version"]}.log'
        , mode='a'
        , encoding='UTF-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logger_formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


def get_time_info_str(total_seconds):
    total_seconds = int(total_seconds)
    hours = (total_seconds // 60 // 60)
    minutes = (total_seconds // 60 % 60)
    seconds = (total_seconds % 60)

    return ('%2d:%02d:%02d' % (hours, minutes, seconds))


def log(
        epoch
        , stage
        , iterations
        , time
        , loss
        , lr
        , other):
    header2item = {
        'epoch': epoch
        , 'stage': stage
        , 'iterations': iterations
        , 'time': time
        , 'loss': loss
        , 'lr': lr
    }

    info_headers = []
    info_table = [[]]

    for header in header2item:
        if header2item != None:
            info_headers.append(header)
            info_table[0].append(header2item[header])

    other_headers = None
    other_table = None

    if isinstance(other, dict):
        other_headers = [
            'MiP', 'MiR', 'MiF', 'MiE'
            , 'MaP', 'MaR', 'MaF', 'MaE'
        ]

        other_table = [
            [
                other['mip']
                , other['mir']
                , other['mif']
                , other['mie']
                , other['map']
                , other['mar']
                , other['maf']
                , other['mae']
            ]
        ]
    else:
        # isinstance(other, str) == True
        other_headers = ['Other Message']
        other_table = [[other]]

    log_str = (
        '\n'
        + tabulate(
            tabular_data=info_table
            , headers=info_headers
            , tablefmt='pretty'
        )
        + '\n'
        + tabulate(
            tabular_data=other_table
            , headers=other_headers
            , tablefmt='pretty'
        )
    )

    logger.info(f'{log_str}\n')


def save_checkpoint(
        model
        , optimizer
        , scheduler
        , trained_epoch
        , directory_path
        , file_name):
    save_params = {
        'model': model.state_dict()
        , 'optimizer': optimizer.state_dict()
        , 'scheduler': scheduler.state_dict()
        , 'trained_epoch': trained_epoch
    }

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    try:
        torch.save(obj=save_params, f=f'{directory_path}/{file_name}')
    except Exception:
        logger.error(f'Failed to save model with error {Exception}.')
        raise Exception
