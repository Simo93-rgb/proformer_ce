import torch
import time
from tqdm import tqdm

print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
# Funzione per misurare il tempo di esecuzione su CPU
def benchmark_cpu(tensor_size):
    """
    Measures the execution time of tensor addition on the CPU.

    Args:
        tensor_size (tuple): The size of the tensors to be created.

    Returns:
        float: The total time taken to perform the addition 1000 times.
    """
    device = torch.device('cpu')
    x = torch.randn(tensor_size, device=device)
    y = torch.randn(tensor_size, device=device)
    
    start_time = time.time()
    for _ in tqdm(range(1000), desc="CPU Progress"):
        z = x + y
    end_time = time.time()
    
    return end_time - start_time

# Funzione per misurare il tempo di esecuzione su GPU
def benchmark_gpu(tensor_size):
    """
    Measures the execution time of tensor addition on the GPU.

    Args:
        tensor_size (tuple): The size of the tensors to be created.

    Raises:
        RuntimeError: If a GPU is not available.

    Returns:
        float: The total time taken to perform the addition 1000 times.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU non disponibile")
    
    device = torch.device('cuda')
    x = torch.randn(tensor_size, device=device)
    y = torch.randn(tensor_size, device=device)
    
    # Sincronizza e misura il tempo di esecuzione
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in tqdm(range(1000), desc="GPU Progress"):
        z = x + y
    torch.cuda.synchronize()
    end_time = time.time()
    
    return end_time - start_time

# Dimensione del tensore
tensor_size = (10000, 10000)

# Esegui il benchmark su CPU e GPU
gpu_time = benchmark_gpu(tensor_size)
print(f"Tempo di esecuzione su GPU: {gpu_time:.6f} secondi")

cpu_time = benchmark_cpu(tensor_size)
print(f"Tempo di esecuzione su CPU: {cpu_time:.6f} secondi")