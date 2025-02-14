class BagOfWordsModel(nn.Module):
    def __init__(self, medium, metric):
        super(BagOfWordsModel, self).__init__()
        num_items = {
            x: pd.read_csv(f"../../data/training/{y}.csv").matchedid.max() + 1
            for (x, y) in {0: "manga", 1: "anime"}.items()
        }
        self.model = nn.Sequential(
            nn.Linear(sum(num_items.values()) * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(256, num_items[medium])
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        lossfn_map = {
            "rating": self.mse,
            "watch": self.crossentropy,
            "plantowatch": self.crossentropy,
            "drop": self.binarycrossentropy,
        }
        self.lossfn = lossfn_map[metric]
        evaluate_map = {
            "rating": self.moments,
            "watch": self.crossentropy,
            "plantowatch": self.crossentropy,
            "drop": self.binarycrossentropy,
        }
        self.evaluatefn = evaluate_map[metric]

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

    def moments(self, x, y, w):
        return (
            self.mse(1 * x, y, w),
            self.mse(0 * x, y, w),
            self.mse(-1 * x, y, w),
        )

    def forward(self, inputs, labels, weights, mode):
        e = self.model(inputs)
        if mode == "train":
            return self.lossfn(self.classifier(e), labels, weights), weights.sum()
        elif mode == "eval":
            return self.evaluatefn(self.classifier(e), labels, weights), weights.sum()
        elif mode == "embed":
            return e
        else:
            assert False
