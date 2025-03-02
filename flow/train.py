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

