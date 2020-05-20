import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, input_size, activation, output_size, dropout=None, bn=None):
        super(ANN, self).__init__()

        def ann_block(in_neurons, out_neurons, activation=nn.ReLU(), dropout=None, bn=None):
            block = [nn.Linear(in_neurons, out_neurons), activation]

            if dropout is not None:
                block.append(dropout)

            if bn is not None:
                block.append(bn)

            return block

        self.model = nn.Sequential(
            *ann_block(input_size, 512, activation, dropout, bn),
            *ann_block(512, 1024, activation, dropout),
            *ann_block(1024, 2048, activation, dropout),
            *ann_block(2048, 2048, activation, dropout),
            *ann_block(2048, 1024, activation, dropout),
            *ann_block(1024, 512, activation, dropout),
            *ann_block(512, 256, activation, dropout),
            *ann_block(256, output_size, activation, dropout)
        )

    def forward(self, x):
        shape = x.shape[0]
        return self.model(x.view(shape, -1))


