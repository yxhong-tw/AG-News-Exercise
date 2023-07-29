from utils import initialize_logger
from initialize import initialize
from do_train import do_train
from do_test import do_test


configs = {}

# Basic related settings
configs['version'] = 'pre-test-1'
configs['logging_time'] = 1
configs['drive_path'] = '/content/drive'
# config['AGNews_path'] = (
#     config['drive_path'] 
#     + '/Othercomputers/Pulsar-MSI/Workspace/NCKU/IKM_Lab/AG-News-Exercise'
# )
configs['AGNews_path'] = '.'

# Training related settings
configs['epoch'] = 2
configs['batch_size'] = 8

## Model related settings
configs['model_name'] = 'BertWithNN'
configs['hidden_size'] = 768
configs['class_number'] = 4
configs['nn_layer_number'] = 3
configs['n_factor'] = 0.5
configs['freeze_lm'] = False

### Checkpoint related settings
configs['load_checkpoint'] = False
configs['checkpoint_epoch'] = -1

## Optimizer related settings
configs['optimizer_name'] = 'Adam'
configs['optimizer_lr'] = 1e-5

## Scheduler related settings
configs['scheduler_name'] = 'ReduceLROnPlateau'
configs['scheduler_mode'] = 'min'
configs['scheduler_factor'] = 0.1
configs['scheduler_patience'] = 2
configs['scheduler_verbose'] = True

## Data related settings
configs['max_sequence_len'] = 512
configs['formatter_name'] = 'AGNewsFormatter'
configs['dataset_name'] = 'AGNewsDataset'
configs['dataloader_shuffle'] = True


def main():
    logger = initialize_logger(configs=configs)
    parameters = initialize(configs=configs, mode='train')

    do_train(configs=configs, parameters=parameters)
    # do_test(configs=configs, parameters=parameters, stage='validate')
    # do_test(configs=configs, parameters=parameters, stage='test')


if __name__ == '__main__':
    main()
