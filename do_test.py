import torch

from torch.autograd import Variable
from timeit import default_timer as timer

from utils import get_time_info_str, log
from evaluation import get_mima_prfe


def do_test(configs, parameters, stage='test', epoch=None):
    model = parameters['model']
    current_epoch = parameters['trained_epoch']
    dataloader = parameters['test_dataloader']

    if stage == 'validate':
        dataloader = parameters['validation_dataloader']

    if epoch != None:
        current_epoch = epoch

    device = parameters['device']

    batch_size = configs['batch_size']
    logging_time = configs['logging_time']

    dataloader_len = len(dataloader)

    with torch.no_grad():
        model.eval()

        total_loss = 0
        cm = None
        mima_prf = None

        start_time = timer()

        for step, data in enumerate(iterable=dataloader):
            for key in data.keys():
                data[key] = Variable(data[key].to(device))

            output = model(data=data, cm=cm)
            loss = output['loss']
            cm = output['cm']

            total_loss += float(loss)

            if step % logging_time == 0:
                temp_epoch_loss = float(total_loss / (step + 1))
                delta_time = (timer() - start_time)
                dtime_str = get_time_info_str(delta_time)
                rtime_str = get_time_info_str(
                    delta_time*(dataloader_len-step-1)/(step+1))

                if cm != None:
                    mima_prf = get_mima_prfe(cm=cm)

                log(
                    epoch=current_epoch
                    , stage=stage
                    , iterations=f'{(step+1)}/{dataloader_len}'
                    , time=f'{dtime_str}/{rtime_str}'
                    , loss=str(round(number=temp_epoch_loss, ndigits=7))
                    , lr=None
                    , other=mima_prf
                )

        temp_epoch_loss = float(total_loss / dataloader_len)
        delta_time = (timer() - start_time)
        dtime_str = get_time_info_str(delta_time)
        rtime_str = get_time_info_str(0)

        if cm != None:
            mima_prf = get_mima_prfe(cm=cm)

        log(
            epoch=current_epoch
            , stage=stage
            , iterations=f'{dataloader_len}/{dataloader_len}'
            , time=f'{dtime_str}/{rtime_str}'
            , loss=str(round(number=temp_epoch_loss, ndigits=7))
            , lr=None
            , other=mima_prf
        )

        return total_loss
