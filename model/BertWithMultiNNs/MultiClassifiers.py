import torch
import torch.nn as nn


class MultiClassifier(nn.Module):
    def __init__(self, configs, device):
        super().__init__()

        nn_layer_number = configs['nn_layer_number']
        hidden_size = configs['hidden_size']
        n_factor = configs['n_factor']
        class_number = configs['class_number']

        self.device = device
        self.classifiers = []
        
        for _ in range(class_number):
            self.classifiers.append(nn.Sequential())

            for i in range(nn_layer_number):
                if i == nn_layer_number - 1:
                    self.classifiers[-1].add_module(
                        f'linear_{i}'
                        , nn.Linear(
                            int(hidden_size*(n_factor**i))
                            , 2
                        )
                    )
                else:
                    self.classifiers[-1].add_module(
                        f'linear_{i}'
                        , nn.Linear(
                            int(hidden_size*(n_factor**i))
                            , int(hidden_size*(n_factor**(i+1)))
                        )
                    )

                self.classifiers[-1].add_module(f'relu_{i}', nn.ReLU())

            self.classifiers[-1].to(device)


    def forward(self, tensors):
        # The size of input tensors should be [batch_size, 768].
        # The size of outputs should be [class_num, batch_size, 2].
        outputs = None

        for classifier in self.classifiers:
            if outputs == None:
                outputs = classifier(input=tensors)
                outputs = torch.unsqueeze(input=outputs, dim=0)
            else:
                # The size of output should be [batch_size, 2].
                output = classifier(input=tensors)
                output = torch.unsqueeze(input=output, dim=0)
                outputs = torch.cat(tensors=(outputs, output), dim=0)

        # The size of return tensor should be [batch_size, class_num, 2].
        return outputs.permute(1, 0, 2)
