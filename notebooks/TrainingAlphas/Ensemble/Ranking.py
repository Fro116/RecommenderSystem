import os

import numpy as np
import torch
import torch.nn as nn


class RankingModel(nn.Module):
    def __init__(self, interaction_weights):
        super(RankingModel, self).__init__()
        self.model = nn.Linear(4, 1)
        self.interaction_weights = interaction_weights

    def preference_relation(self, prios, interaction_weights, layer):
        x = prios[:, :, layer].unsqueeze(2)
        y = x.permute(0, 2, 1)
        return interaction_weights[layer] * (x > y)

    def combine_preferences(self, x, y):
        return x + (x == 0) * (x.permute(0, 2, 1) == 0) * y

    def process_preferences(self, prios, interaction_weights):
        a = self.preference_relation(prios, interaction_weights, 0)
        b = self.preference_relation(prios, interaction_weights, 1)
        c = self.preference_relation(prios, interaction_weights, 2)
        p = self.combine_preferences(a, b)
        p = self.combine_preferences(p, c)
        halflife = np.exp(np.log(0.5) / -50)
        worse_than = p.sum(dim=1)
        better_than = p.sum(dim=2)
        w = (better_than != 0) * (halflife**worse_than)
        w = (w / w.sum(dim=1).unsqueeze(1)).unsqueeze(2)
        return p, w

    def mle_loss(self, m, feats, prios, interaction_weights):
        # position aware list mle loss with modifications to handle a weighted
        # preference relation instead of a total ordering. See [Position-Aware
        # ListMLE: A Sequential Learning Process for Ranking](
        # https://auai.org/uai2014/proceedings/individuals/164.pdf)
        p, w = self.process_preferences(prios, interaction_weights)
        q = m(feats)
        q = q - q.max(dim=1, keepdim=True)[0]
        lognumer = q
        eq = torch.exp(q)
        logdenom = torch.log(torch.matmul(p, eq) + eq)
        return -(w * (lognumer - logdenom)).sum() / w.sum()

    def forward(self, feats, prios):
        return self.mle_loss(self.model, feats, prios, self.interaction_weights)



# I/O
def get_data_path(file):
    path = os.getcwd()
    while os.path.basename(path) != "notebooks":
        path = os.path.dirname(path)
    path = os.path.dirname(path)
    return os.path.join(path, "data", file)


def load_model(outdir, map_location=None):
    fn = os.path.join(outdir, "model.pt")
    state_dict = torch.load(fn, map_location=map_location)
    compile_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(compile_prefix):
            state_dict[k[len(compile_prefix) :]] = state_dict.pop(k)
    return state_dict