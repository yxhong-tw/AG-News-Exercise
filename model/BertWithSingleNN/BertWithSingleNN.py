import torch.nn as nn

from model.Bert import Bert
from model.BertWithSingleNN.SingleClassifier import SingleClassifier
from evaluation import get_comfusion_matrix


class BertWithSingleNN(nn.Module):
    def __init__(self, configs, device):
        super().__init__()

        self.lm = Bert()
        self.classifier = SingleClassifier(configs=configs, device=device)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, data, cm, freeze_lm=True):
        # The number of input "data" should be "batch_size".
        # The size of "cls_embeddings" should be [batch_size, 768].
        # The number of "labels" should be "batch_size".

        for param in self.lm.parameters():
            param.requires_grad = False if freeze_lm == True else True

        cls_embeddings = self.lm(ids=data['text'])
        predictions = self.classifier(tensors=cls_embeddings)
        labels = data['label']

        loss = self.criterion(predictions, labels)
        cm = get_comfusion_matrix(preds=predictions, labels=labels, cm=cm)

        return {
            'prediction': predictions
            , 'label': labels
            , 'loss': loss
            , 'cm': cm
        }
