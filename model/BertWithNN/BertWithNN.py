import torch.nn as nn

from model.BertWithNN.Bert import Bert
from model.BertWithNN.Classifier import Classifier


class BertWithNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.lm = Bert()
        self.classifier = Classifier(config=config)
        self.criterion = nn.CrossEntropyLoss()


    # TODO
    def forward(self, data, freeze_lm):
        # The number of input "data" should be "batch_size".
        # The size of "cls_embeddings" should be [batch_size, 768].
        # The number of "labels" should be "batch_size".

        for param in self.lm.parameters():
            param.requires_grad = False if freeze_lm == True else True

        print(data)
        input()

        cls_embeddings = self.lm(ids=data['text'])
        predictions = self.classifier(tensors=cls_embeddings)
        labels = data['label']

        loss = self.criterion(predictions, labels)

        return {
            'prediction': predictions
            , 'label': labels
            , 'loss': loss
        }
