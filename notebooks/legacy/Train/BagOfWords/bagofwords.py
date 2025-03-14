import torch
import torch.nn as nn


class BagOfWordsModel(nn.Module):
    def __init__(self, input_sizes, output_index, metric):
        super(BagOfWordsModel, self).__init__()

        self.input_sizes = input_sizes
        self.output_index = output_index
        self.metric = metric
        self.output_size = self.input_sizes[self.output_index]
        self.input_fields = 2
        self.model = nn.Sequential(
            nn.Linear(sum(self.input_sizes) * self.input_fields, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_size),
        )
        if self.metric == "rating":
            self.lossfn = self.mse
        elif self.metric in ["watch", "plantowatch"]:
            self.logsoftmax = nn.LogSoftmax(dim=-1)
            self.softmax = nn.Softmax(dim=-1)
            self.lossfn = self.crossentropy
        elif self.metric == "drop":
            self.sigmoid = nn.Sigmoid()
            self.lossfn = self.binarycrossentropy
        else:
            assert False

    def mse(self, x, y, w):
        return (torch.square(x - y) * w).sum()

    def crossentropy(self, x, y, w):
        x = self.logsoftmax(x)
        return (-x * y * w).sum()

    def binarycrossentropy(self, x, y, w):
        return nn.functional.binary_cross_entropy_with_logits(
            input=x,
            target=y,
            weight=w,
            reduction="sum",
        )

    def forward(self, inputs, labels, weights, mask, mode):
        if mask:
            masks = [
                torch.rand(inputs.shape[0], x, device=inputs.device) > mask
                for x in self.input_sizes
            ]
            input_mask = torch.cat([masks[0], masks[1]] * self.input_fields, 1)
            output_mask = ~masks[self.output_index]
            inputs = inputs * input_mask
            weights = weights * output_mask
        if mode == "training":
            return self.lossfn(self.model(inputs), labels, weights), weights.sum()
        elif mode == "evaluation":
            if self.metric == "rating":
                # we return the quadratic so we can meaure correlation
                x = self.model(inputs)
                return (
                    self.lossfn(1 * x, labels, weights),
                    self.lossfn(0 * x, labels, weights),
                    self.lossfn(-1 * x, labels, weights),
                ), weights.sum()
            elif self.metric in ["watch", "plantowatch", "drop"]:
                return self.lossfn(self.model(inputs), labels, weights), weights.sum()
        elif mode == "inference":
            assert not mask
            x = self.model(inputs)
            if self.metric in ["watch", "plantowatch"]:
                x = self.softmax(x)
            elif self.metric == "drop":
                x = self.sigmoid(x)
            return x
        else:
            assert False