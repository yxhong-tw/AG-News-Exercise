import logging
import torch

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
    minutes = (total_seconds // 60)
    seconds = (total_seconds % 60)

    return ('%2d:%02d:%02d' % (hours, minutes, seconds))


def log(
        epoch
        , stage
        , iterations
        , time
        , loss
        , lr
        , mima_prf):
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

    mima_prf_headers = [
        'MiP', 'MiR', 'MiF'
        , 'MaP', 'MaR', 'MaF'
    ]

    mima_prf_table = [
        [
            mima_prf['mip']
            , mima_prf['mir']
            , mima_prf['mif']
            , mima_prf['map']
            , mima_prf['mar']
            , mima_prf['maf']
        ]
    ]

    log_str = (
        '\n'
        , tabulate(
            tabular_data=info_table
            , headers=info_headers
            , tablefmt='pretty'
        )
        , '\n'
        , tabulate(
            tabular_data=mima_prf_table
            , headers=mima_prf_headers
            , tablefmt='pretty'
        )
    )

    logger.info(f'{log_str}\n')


def save_checkpoint(
        model
        , optimizer
        , scheduler
        , trained_epoch
        , file):
    save_params = {
        'model': model
        , 'optimizer': optimizer
        , 'scheduler': scheduler
        , 'trained_epoch': trained_epoch
    }

    try:
        torch.save(obj=save_params, f=file)
    except Exception:
        logger.errr(f'Failed to save model with error {Exception}.')
        raise Exception
