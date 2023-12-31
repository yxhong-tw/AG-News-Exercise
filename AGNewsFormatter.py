import torch

from transformers import BertTokenizer


class AGNewsFormatter:
    def __init__(self, configs):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.model_name = configs['model_name']
        self.max_sequence_len = configs['max_sequence_len']


    def format(self, data):
        texts = []
        labels = []

        for one_data in data:
            title_string = one_data['title']
            description_string = one_data['description']

            text_ids = self.string2ids(
                first_string=title_string
                , second_string=description_string)

            texts.append(text_ids)

            label = None

            if self.model_name == 'BertWithSingleNN':
                label = [0, 0, 0, 0]
                label[one_data['label']-1] = 1
            elif self.model_name == 'BertWithMultiNNs':
                label = []

                for idx in range(4):
                    if idx == (one_data['label'] - 1):
                        label.append([0, 1])
                    else:
                        label.append([1, 0])

            labels.append(label)

        return {
            'text': torch.IntTensor(texts)
            , 'label': torch.FloatTensor(labels)
        }


    def string2ids(self, first_string, second_string):
        string = '[CLS]' + first_string + '[SEP]' + second_string
        tokens = self.tokenizer.tokenize(text=string)

        for _ in range(self.max_sequence_len-len(tokens)):
            tokens.append('[PAD]')

        ids = self.tokenizer.convert_tokens_to_ids(tokens=tokens)

        return ids
