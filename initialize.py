import logging
import numpy as np
import random
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.BertWithNN.BertWithNN import BertWithNN
from AGNewsFormatter import AGNewsFormatter
from AGNewsDataset import AGNewsDataset


logger = logging.getLogger(__name__)


def initialize_seeds(seed=48763):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f'Initializing seed to {seed} successfully.')


def initialize_device(device_type='cuda'):
    if not torch.cuda.is_available():
        device_type = 'cpu'

    logger.info(f'Initializing device to {device_type} successfully.')

    return torch.device(device_type)


def initialize_model(config, device):
    model_name = config['model_name']
    model = None

    if model_name == 'BertWithNN':
        model = BertWithNN(config=config)
    else:
        logger.error(f'There is no model named {model_name}.')
        raise ValueError(f'There is no model named {model_name}.')

    logger.info(f'Initializing model to {model_name} successfully.')

    return model.to(device)


def initialize_optimizer(config, model):
    optimizer_name = config['optimizer_name']
    optimizer = None

    if optimizer_name == 'Adam':
        optimizer = Adam(params=model.parameters(), lr=config['optimizer_lr'])
    else:
        logger.error(f'There is no optimizer named {optimizer_name}.')
        raise ValueError(f'There is no optimizer named {optimizer_name}.')

    logger.info(f'Initializing optimizer to {optimizer_name} successfully.')

    return optimizer


def initialize_scheduler(config, optimizer):
    scheduler_name = config['scheduler_name']
    scheduler = None

    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer
            , mode=config['scheduler_mode']
            , factor=config['scheduler_factor']
            , patience=config['scheduler_patience']
            , verbose=config['scheduler_verbose']
        )
    else:
        logger.error(f'There is no scheduler named {scheduler_name}.')
        raise ValueError(f'There is no scheduler named {scheduler_name}.')

    logger.info(f'Initializing scheduler to {scheduler_name} successfully.')

    return scheduler


def initialize_formatter(config):
    formatter_name = config['formatter_name']
    formatter = None

    if formatter_name == 'AGNewsFormatter':
        formatter = AGNewsFormatter(config=config)
    else:
        logger.error(f'There is no formatter named {formatter_name}.')
        raise ValueError(f'There is no formatter named {formatter_name}.')

    def collate_fn(data):
        return formatter.process(data)

    logger.info(f'Initializing formatter to {formatter_name} successfully.')

    return collate_fn


def initialize_dataloader(config, task_name):
    dataset_name = config['dataset_name']
    dataloader = None

    if dataset_name == 'AGNewsDataset':
        dataset = AGNewsDataset(config=config, task_name=task_name)
        batch_size = config['batch_size']
        shuffle = config['dataloader_shuffle']
        collate_fn = initialize_formatter(config=config)

        dataloader = DataLoader(
            dataset=dataset
            , batch_size=batch_size
            , shuffle=shuffle
            , collate_fn=collate_fn
        )
    else:
        logger.error(f'There is no dataset named {dataset_name}.')
        raise ValueError(f'There is no dataset named {dataset_name}.')

    logger.info(f'Initializing {task_name} dataloader successfully.')

    return dataloader


def load_checkpoint(config, model, optimizer, scheduler, trained_epoch):
    try:
        checkpoint_parameters = torch.load(f=config['load_checkpoint_path'])

        model.load_state_dict(checkpoint_parameters['model'])
        optimizer.load_state_dict(checkpoint_parameters['optimizer'])
        scheduler.load_state_dict(checkpoint_parameters['scheduler'])
        trained_epoch = checkpoint_parameters['trained_epoch']
    except Exception:
        logger.error('Failed to Load checkpoint from checkpoint path.')
        raise Exception('Failed to Load checkpoint from checkpoint path.')

    logger.info('Load checkpoint from checkpoint path successfully.')

    return (model, optimizer, scheduler, trained_epoch)


def initialize(config, mode):
    initialize_seeds()

    device = initialize_device()
    model = initialize_model(config=config, device=device)
    trained_epoch = -1

    parameters = {
        'device': device
    }

    if mode == 'train':
        optimizer = initialize_optimizer(config=config, model=model)
        scheduler = initialize_scheduler(config=config, optimizer=optimizer)

        train_dataloader = initialize_dataloader(
            config=config
            , task_name='train')
        validation_dataloader = initialize_dataloader(
            config=config
            , task_name='validation')

    if config['load_checkpoint'] == True:
        model, optimizer, scheduler, trained_epoch = load_checkpoint(
            config=config
            , model=model
            , optimizer=optimizer
            , scheduler=scheduler
            , trained_epoch=trained_epoch)

    test_dataloader = initialize_dataloader(config=config, task_name='test')

    parameters['model'] = model
    parameters['trained_epoch'] = trained_epoch
    parameters['test_dataloader'] = test_dataloader

    if mode == 'train':
        parameters['optimizer'] = optimizer
        parameters['scheduler'] = scheduler
        parameters['train_dataloader'] = train_dataloader
        parameters['validation_dataloader'] = validation_dataloader

    logger.info('Initialize all parameters successfully.')
    logger.info(f'Details of all parameters: \n{parameters}')

    return parameters
