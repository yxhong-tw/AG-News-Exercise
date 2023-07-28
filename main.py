from utils import initialize_logger
from initialize import initialize
from train import train
from test import test


config = {}

# Basic related settings
config['version'] = 'pre-test-1'
config['logging_time'] = 1
config['drive_path'] = '/content/drive'
# config['AGNews_path'] = (
#     config['drive_path'] 
#     + '/Othercomputers/Pulsar-MSI/Workspace/NCKU/IKM_Lab/AG-News-Exercise'
# )
config['AGNews_path'] = '.'

# Training related settings
config['epoch'] = 2
config['batch_size'] = 8

## Model related settings
config['model_name'] = 'BertWithNN'
config['hidden_size'] = 768
config['class_number'] = 4
config['nn_layer_number'] = 3
config['n_factor'] = 0.5
config['freeze_lm'] = False

### Checkpoint related settings
config['load_checkpoint'] = True
config['checkpoint_epoch'] = 1

## Optimizer related settings
config['optimizer_name'] = 'Adam'
config['optimizer_lr'] = 1e-5

## Scheduler related settings
config['scheduler_name'] = 'ReduceLROnPlateau'
config['scheduler_mode'] = 'min'
config['scheduler_factor'] = 0.1
config['scheduler_patience'] = 2
config['scheduler_verbose'] = True

## Data related settings
config['max_sequence_len'] = 512
config['formatter_name'] = 'AGNewsFormatter'
config['dataset_name'] = 'AGNewsDataset'
config['dataloader_shuffle'] = True


def main():
    logger = initialize_logger(config=config)
    parameters = initialize(config=config, mode='train')

    # train(config=config, parameters=parameters)
    # test(config=config, parameters=parameters, stage='validate')
    test(config=config, parameters=parameters, stage='test')


if __name__ == '__main__':
    main()
