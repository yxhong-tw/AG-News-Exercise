import torch.nn as nn

from transformers import BertModel


class Bert(nn.Module):
  def __init__(self):
    super().__init__()

    self.bert = BertModel.from_pretrained('bert-base-uncased')


  def forward(self, ids):
    # The size of input "ids" should be [batch_size, 512].
    # The size of "pooler_output" should be [batch_size, 768]
    # ("pooler_output" is the last layer hidden-state of
    # the first token of the sequence).
    output = self.bert(input_ids=ids)

    return output.pooler_output
