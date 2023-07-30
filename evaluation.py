import numpy as np


def get_comfusion_matrix(preds, labels, cm):
    # Reference from https://blog.csdn.net/Answer3664/article/details/104417013.
    preds = preds.cpu().clone().detach().numpy()
    labels = labels.cpu().clone().detach().numpy()

    for pred in preds:
        for idx in range(len(pred)):
            pred[idx] = 1 if idx == np.argmax(pred) else 0

    if cm == None:
        cm = []

        for _ in range(4):
            cm.append({'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})

    for idx in range(4):
        part_preds = preds[:, idx]
        part_labels = labels[:, idx]

        cm[idx]['tp'] += int((part_labels * part_preds).sum())
        cm[idx]['fp'] += int(((1 - part_labels) * part_preds).sum())
        cm[idx]['tn'] += int((part_labels * (1 - part_preds)).sum())
        cm[idx]['fn'] += int(((1 - part_labels) * (1 - part_preds)).sum())

    return cm


def get_prfe(cm):
    pre = 0.0
    re = 0.0
    f1 = 0.0
    err = 0.0

    if cm['tp'] != 0:
        pre = float(cm['tp'] / (cm['tp'] + cm['fp']))
        re = float(cm['tp'] / (cm['tp'] + cm['fn']))
        f1 = float(2 * pre * re / (pre + re))

    if cm['fp'] != 0 and cm['fn'] != 0:
        err = float(
            (cm['fp'] + cm['fn']) / (cm['tp'] + cm['fp'] + cm['tn'] + cm['fn']))

    return pre, re, f1, err


def get_mima_prfe(cm):
    precisions = []
    recalls = []
    f1s = []
    errs = []

    cm_sum = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    for idx in range(4):
        cm_sum['tp'] += cm[idx]['tp']
        cm_sum['fp'] += cm[idx]['fp']
        cm_sum['tn'] += cm[idx]['tn']
        cm_sum['fn'] += cm[idx]['fn']

        precision, recall, f1, err = get_prfe(cm=cm[idx])

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        errs.append(err)

    micro_precision, micro_recall, micro_f1, micro_err = get_prfe(cm=cm_sum)

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    macro_err = 0.0

    for idx in range(4):
        macro_precision += precisions[idx]
        macro_recall += recalls[idx]
        macro_f1 += f1s[idx]
        macro_err += errs[idx]

    macro_precision /= len(precisions)
    macro_recall /= len(recalls)
    macro_f1 /= len(f1s)
    macro_err /= len(errs)

    return {
        'mip': round(micro_precision, 3)
        , 'mir': round(micro_recall, 3)
        , 'mif': round(micro_f1, 3)
        , 'mie': round(micro_err, 3)
        , 'map': round(macro_precision, 3)
        , 'mar': round(macro_recall, 3)
        , 'maf': round(macro_f1, 3)
        , 'mae': round(macro_err, 3)
    }
