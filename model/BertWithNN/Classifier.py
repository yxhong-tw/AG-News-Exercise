import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        nn_layer_number = config['nn_layer_number']
        hidden_size = config['hidden_size']
        n_factor = config['n_factor']
        class_number = config['class_number']

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
