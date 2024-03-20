import argparse
import logging
import time
from multiprocessing import Process

import torch

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Parallel GPU Memory and Stress Test')
parser.add_argument('-r', '--runtime', type=int, default=300, help='Duration of the test in seconds')
parser.add_argument('-d', '--delay', type=int, default=0, help='Delay before starting the test in seconds')
parser.add_argument('-g', '--gpus', default='all', help='Specify GPU ids to use (e.g., "0,1,3") or "all" for all GPUs. Default is "all".')
args = parser.parse_args()

def get_target_gpus(gpu_arg):
    if gpu_arg.lower() == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(id) for id in gpu_arg.split(',') if id.isdigit()]

def load_and_stress_gpu(gpu_id, runtime):
    torch.cuda.set_device(gpu_id)
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    target_memory = total_memory * 0.9  # 90% of total memory
    num_elements = int(target_memory / 4)  # Float32, 4 bytes each
    
    try:
        tensor = torch.randn(num_elements, device='cuda', dtype=torch.float32)
        logging.info(f"GPU {gpu_id}: Loaded approximately 90% of GPU memory.")
    except RuntimeError as e:
        logging.error(f"GPU {gpu_id}: Failed to load memory. {e}")
        return

    logging.info(f"GPU {gpu_id}: Starting stress test.")
    start_time = time.time()
    while time.time() - start_time < runtime:
        tensor.mul_(1.0001)

    logging.info(f"GPU {gpu_id}: Stress test completed.")

if __name__ == "__main__":
    # Delay execution if specified
    time.sleep(args.delay)

    target_gpus = get_target_gpus(args.gpus)
    processes = []

    for gpu_id in target_gpus:
        p = Process(target=load_and_stress_gpu, args=(gpu_id, args.runtime))
        p.start()
        processes.append(p)
        logging.info(f"Started process for GPU {gpu_id}.")

    for p in processes:
        p.join()
        logging.info("Process joined.")

    logging.info("All GPU tests completed.")
