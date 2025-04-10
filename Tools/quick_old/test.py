import torch,os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn

def example(rank, world_size):
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(11111111)
    model = nn.Linear(2, 1, False).to(rank) 
    print(rank)
    print(222222)
    dist.barrier()
    print("init")
    opt = optim.Adam(model.parameters(), lr=0.0001) 
    opt_stat = torch.load('opt_weight', {'cuda:0':'cuda:%d'%rank}) 
    opt.load_state_dict(opt_stat)
    ddp_model = DDP(model, device_ids=[rank])
    inp = torch.tensor([[1.,2]]).to(rank) 
    labels = torch.tensor([[5.]]).to(rank)
    outp = ddp_model(inp)
    loss = torch.mean((outp - labels)**2)
    opt.zero_grad()
    loss.backward() 

    opt.step()
    if rank == 0:
        torch.save(model.state_dict(), 'model_weight')
        torch.save(opt.state_dict(), 'opt_weight')

if __name__=="__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True) 