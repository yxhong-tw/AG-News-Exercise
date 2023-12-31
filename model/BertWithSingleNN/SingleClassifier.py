import torch.nn as nn


class SingleClassifier(nn.Module):
    def __init__(self, configs, device):
        super().__init__()

        hidden_size = configs['hidden_size']
        class_number = configs['class_number']
        nn_layer_number = configs['nn_layer_number']
        n_factor = configs['n_factor']

        self.classifier = nn.Sequential()

        for i in range(nn_layer_number):
            if i == nn_layer_number - 1:
                self.classifier.add_module(
                    f'linear_{i}'
                    , nn.Linear(int(hidden_size*(n_factor**i)), class_number)
                )
            else:
                self.classifier.add_module(
                    f'linear_{i}'
                    , nn.Linear(
                        int(hidden_size*(n_factor**i))
                        , int(hidden_size*(n_factor**(i+1)))
                    )
                )

        self.classifier.add_module(f'relu_{i}', nn.ReLU())

        # DO NOT ADD SOFTMAX LAYER THERE!
        # (CrossEntropyLoss() is already implemented with Softmax.
        # If we add another one Softmax there, it may influence the gradients.
        # self.classifier.add_module(f'softmax_0', F.softmax(dim=1))


    def forward(self, tensors):
        # The size of input tensors should be [batch_size, 768].
        # The size of classifier output should be [batch_size, 4].
        return self.classifier(input=tensors)
