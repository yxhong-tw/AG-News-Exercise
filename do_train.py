import gc
import os
import torch

from timeit import default_timer as timer
from torch.autograd import Variable

from do_test import do_test
from evaluation import get_mima_prfe
from utils import get_time_info_str, log, save_checkpoint


def do_train(configs, parameters):
    device = parameters['device']
    model = parameters['model']
    trained_epoch = parameters['trained_epoch']
    optimizer = parameters['optimizer']
    scheduler = parameters['scheduler']
    train_dataloader = parameters['train_dataloader']

    logging_time = configs['logging_time']
    AGNews_path = configs['AGNews_path']
    total_epoch = configs['epoch']
    freeze_lm = configs['freeze_lm']

    train_dataloader_len = len(train_dataloader)

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
                    , time=f'{dtime_str}/{rtime_str}'
                    , loss=str(round(number=temp_epoch_loss, ndigits=7))
                    , lr=str(round(number=lr, ndigits=7))
                    , other=mima_prf
                )

        temp_epoch_loss = float(epoch_loss / train_dataloader_len)
        delta_time = (timer() - start_time)
        dtime_str = get_time_info_str(delta_time)
        rtime_str = get_time_info_str(0)

        if cm != None:
            mima_prf = get_mima_prfe(cm=cm)

        log(
            epoch=current_epoch
            , stage='train'
            , iterations=f'{train_dataloader_len}/{train_dataloader_len}'
            , time=f'{dtime_str}/{rtime_str}'
            , loss=str(round(number=temp_epoch_loss, ndigits=7))
            , lr=str(round(number=lr, ndigits=7))
            , other=mima_prf
        )

        validation_loss = do_test(
            configs=configs
            , parameters=parameters
            , stage='validate'
            , epoch=current_epoch)

        scheduler.step(metrics=validation_loss)

        save_checkpoint(
            model=model
            , optimizer=optimizer
            , scheduler=scheduler
            , trained_epoch=current_epoch
            , directory_path=f'{AGNews_path}/checkpoints/{configs["version"]}'
            , file_name=f'{current_epoch}.pkl'
        )

        do_test(
            configs=configs
            , parameters=parameters
            , stage='test'
            , epoch=current_epoch
        )

        gc.collect()
        torch.cuda.empty_cache()
