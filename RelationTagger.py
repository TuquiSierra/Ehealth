import torch
from torch import nn
from torch.nn.functional import softmax


class RelationTagger(nn.Module):
    def __init__(self, input_size, output_size):
        super(RelationTagger, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.output_layer = nn.Sequential(
            nn.Linear(input_size*2, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, self.output_size)
        )

    def forward(self, X):
        # Sample (t1, t2)
        # t -> [ tensors ]
        samples = []
        for from_entity, to_entity in X:
            from_tensor = torch.mean(from_entity, dim=0)
            to_tensor = torch.mean(to_entity, dim=0)
            samples.append(torch.cat((from_tensor, to_tensor)))
        
        samples_tensor = torch.stack(samples)

        output = self.output_layer(samples_tensor)
        return output