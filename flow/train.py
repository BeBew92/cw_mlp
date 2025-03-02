import torch
import torch.distributed as dist
from models import FLOW

torch.cuda.empty_cache()

if __name__ == '__main__':
    opt = FLOW.build_options().parse_args()
    print(opt)

    # Initialize distributed training only if needed
    # if dist.is_available() and not dist.is_initialized():
    #     dist.init_process_group(backend='nccl', init_method='env://')

    # Set up device
    device0 = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    device1 = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device0  # Second GPU if available

    # Set CUDA device
    torch.cuda.set_device(device0)

    # Initialize model
    model = FLOW(opt)  # Pass both GPUs
    model.fit()
    print("✅ Training finished at iteration:", n_iter)

    # 🚀 Force exit after training to release resources
    import sys
    sys.exit(0)

# // 4 gpus
# import torch
# import torch.multiprocessing as mp
# from models import FLOW

# torch.cuda.empty_cache()  # ✅ Clear memory

# def run_training(gpu_group):
#     """Runs a training session on a specific set of GPUs"""
#     opt = FLOW.build_options().parse_args()
#     print(f"Starting training on GPUs: {gpu_group}")

#     # ✅ Set the primary GPU in each group
#     torch.cuda.set_device(gpu_group[0])  

#     # ✅ Initialize model
#     model = FLOW(opt, device_ids=gpu_group)  # Pass specific GPUs
#     model.fit()

# if __name__ == '__main__':
#     # ✅ Define two separate GPU groups
#     gpu_groups = [[0, 1], [2, 3]]

#     # ✅ Use multiprocessing to launch training on two groups
#     processes = []
#     for group in gpu_groups:
#         p = mp.Process(target=run_training, args=(group,))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()  # ✅ Wait for both processes to finish
# import torch
# import torch.distributed as dist

# from models import FLOW

# if __name__ == '__main__':
#     opt = FLOW.build_options().parse_args()
#     print(opt)
#     dist.init_process_group(backend='nccl',
#                             init_method='env://')
#     torch.cuda.set_device(opt.local_rank)
#     model = FLOW(opt)
#     model.fit()