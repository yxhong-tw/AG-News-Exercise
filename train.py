import os

from torch.autograd import Variable
from timeit import default_timer as timer

from utils import get_time_info_str, log, save_checkpoint
from evaluation import get_mima_prfe
from test import test


def train(config, parameters):
    model = parameters['model']
    optimizer = parameters['optimizer']
    scheduler = parameters['scheduler']
    trained_epoch = parameters['trained_epoch']
    train_dataloader = parameters['train_dataloader']
    validation_dataloader = parameters['validation_dataloader']
    test_dataloader = parameters['test_dataloader']
    device = parameters['device']

    total_epoch = config['epoch']
    batch_size = config['batch_size']
    freeze_lm = config['freeze_lm']
    logging_time = config['logging_time']
    AGNews_path = config['AGNews_path']

    train_dataloader_len = len(train_dataloader)
    validation_dataloader_len = len(validation_dataloader)
    test_dataloader_len = len(test_dataloader)

    for current_epoch in range(trained_epoch+1, total_epoch):
        model.train()

        epoch_loss = 0
        lr = -1
        cm = None
        mima_prf = None

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        start_time = timer()

        for step, data in enumerate(iterable=train_dataloader):
            for key in data.keys():
                data[key] = Variable(data[key].to(device))

            optimizer.zero_grad()

            output = model(data=data, cm=cm, freeze_lm=freeze_lm)
            loss = output['loss']
            cm = output['cm']

            epoch_loss += float(loss)

            loss.backward()

            optimizer.step()

            if step % logging_time == 0:
                temp_epoch_loss = float(epoch_loss / (step + 1))
                delta_time = (timer() - start_time)
                dtime_str = get_time_info_str(total_seconds=delta_time)
                rtime_str = get_time_info_str(
                    total_seconds=delta_time*(
                        train_dataloader_len-step-1)/(step+1))

                if cm != None:
                    mima_prf = get_mima_prfe(cm=cm)

                log(
                    epoch=current_epoch
                    , stage='train'
                    , iterations=f'{(step+1)}/{train_dataloader_len}'
                    , time=f'{dtime_str}\{rtime_str}'
                    , loss=str(round(number=temp_epoch_loss, ndigits=7))
                    , lr=str(round(number=lr, ndigits=7))
                    , other=mima_prf
                )

        temp_epoch_loss = float(epoch_loss / train_dataloader_len / batch_size)
        delta_time = (timer() - start_time)
        dtime_str = get_time_info_str(delta_time)
        rtime_str = get_time_info_str(0)

        if cm != None:
            mima_prf = get_mima_prfe(cm=cm)

        log(
            epoch=current_epoch
            , stage='train'
            , iterations=f'{train_dataloader_len}/{train_dataloader_len}'
            , time=f'{dtime_str}\{rtime_str}'
            , loss=str(round(number=temp_epoch_loss, ndigits=7))
            , lr=str(round(number=lr, ndigits=7))
            , other=mima_prf
        )

        validation_loss = test(
            config=config
            , parameters=parameters
            , stage='validate'
            , epoch=current_epoch)

        scheduler.step(metrics=validation_loss)

        save_checkpoint(
            model=model
            , optimizer=optimizer
            , scheduler=scheduler
            , trained_epoch=current_epoch
            , directory_path=f'{AGNews_path}/checkpoints/{config["version"]}'
            , file_name=f'{current_epoch}.pkl'
        )

        test(
            config=config
            , parameters=parameters
            , stage='test'
            , epoch=current_epoch
        )
