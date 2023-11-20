import time

import torch

m = 4096
k = 4096
n = 4096
repeats = 1000
dtype = [torch.float16, torch.float32, torch.float64]
devices = ["cuda:0"]

def run_benchmark(m, k, n, repeats, dtype, device):
    
    a = torch.rand((m, k), dtype=dtype, device=device)
    b = torch.rand((k, n), dtype=dtype, device=device)
    c = torch.empty((m, n), dtype=dtype, device=device)

    start_time = time.time()
    for i in range(repeats):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    end_time = time.time()

    time_cost = end_time - start_time

    print("============={}+{}=============".format(device.split(':')[0], d.__repr__().split('.')[1]))

    print(" Time cost: {}s".format(time_cost))

    print(
        " M: {} \n K: {} \n N: {} \n Repeats: {} \n {} TFLOPS \n".format(
            m, n, k, repeats, 2 * m * n * k * repeats / time_cost / 1e12
        )
    )

if __name__ == "__main__":
    for d in dtype:
        for device in devices:
            run_benchmark(m, k, n, repeats, d, device)
    
