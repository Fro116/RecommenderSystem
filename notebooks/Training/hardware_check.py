import os

import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
x = torch.ones(1, device=f"cuda:{rank}")
dist.all_reduce(x)
assert x.item() == dist.get_world_size()
dist.destroy_process_group()
