import os
import subprocess
import threading
import time
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
import torch
import logging
import datetime

# Transformers logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# logging.getLogger("transformers").setLevel(logging.DEBUG)
# logging.getLogger("accelerate").setLevel(logging.DEBUG)

scratch_dir = os.getenv("SCRATCH")
if scratch_dir is None:
    scratch_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface_scratch_custom")
    print(f"SCRATCH environment variable not set. Using fallback: {scratch_dir}")
else:
    print(f"Using SCRATCH directory: {scratch_dir}")

job_id = os.getenv("SLURM_JOB_ID")
superdate = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
sstat_file = f"sstat-{job_id}-{superdate}.txt"
nvidia_smi_file = f"nvidia-smi-{job_id}-{superdate}.txt"
times_file = f"times-{job_id}-{superdate}.txt"

# Add stop flag for monitor thread
stop_monitoring = threading.Event()

def monitor_resources():
    while not stop_monitoring.is_set():
        result = subprocess.run(f"sstat --format=AveCPU,AvePages,AveRSS,AveVMSize,MaxRSS,AveVMSize,ConsumedEnergy,JobID -j {job_id} | tail -n 1", shell=True, capture_output=True, text=True)
        with open(sstat_file, 'a') as f:
            f.write(result.stdout)

        result = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,clocks.gr,clocks.mem,pcie.link.gen.current,pcie.link.width.current', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        with open(nvidia_smi_file, 'a') as f:
            f.write(result.stdout)

        time.sleep(1)

monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()

# Define model identifier
model_id = "google/gemma-3-12b-it"
safe_model_name = model_id.replace("/", "_")

# Set up Hugging Face Hub cache directory within SCRATCH
hf_hub_cache_dir = os.path.join(scratch_dir, "huggingface_cache", safe_model_name)
os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache_dir
os.makedirs(hf_hub_cache_dir, exist_ok=True)
print(f"HUGGINGFACE_HUB_CACHE set to: {hf_hub_cache_dir}")

# Define paths for model storage (snapshot) and offloading within SCRATCH
model_storage_path = os.path.join(scratch_dir, "model_snapshots", safe_model_name)
offload_path = os.path.join(scratch_dir, "model_offload", safe_model_name)

os.makedirs(model_storage_path, exist_ok=True)
os.makedirs(offload_path, exist_ok=True)

print(f"Attempting to download/load snapshot for {model_id} to/from: {model_storage_path}")

# Start timing model loading
model_load_snapshot_start_time = time.perf_counter_ns() / 1_000_000  # Convert ns to ms

try:
    model_snapshot_path = snapshot_download(
        repo_id=model_id,
        local_dir=model_storage_path,
        local_dir_use_symlinks=False,
    )
    print(f"Model snapshot for {model_id} is located at: {model_snapshot_path}")
except Exception as e:
    print(f"Error downloading snapshot: {e}")
    print(f"Please ensure you have internet connectivity and the model identifier '{model_id}' is correct.")
    print(f"If it's a gated model, ensure you have access and are logged in via huggingface-cli or provide a token.")
    exit()

model_load_snapshot_end_time = time.perf_counter_ns() / 1_000_000
model_load_snapshot_time = model_load_snapshot_end_time - model_load_snapshot_start_time

model_load_config_start_time = time.perf_counter_ns() / 1_000_000

print(f"Loading processor and config from: {model_snapshot_path}")
try:
    processor = AutoProcessor.from_pretrained(model_snapshot_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_snapshot_path)
except Exception as e:
    print(f"Error loading processor or config from snapshot: {e}")
    exit()

print("Initializing empty model weights...")
with init_empty_weights():
    model = AutoModelForImageTextToText.from_config(config)

model.tie_weights() # Important for some models after empty initialization

model_load_config_end_time = time.perf_counter_ns() / 1_000_000
model_load_config_time = model_load_config_end_time - model_load_config_start_time

print(f"Loading checkpoint and dispatching model using Accelerate...")
no_split_modules = ["GemmaDecoderLayer"] # Common for Gemma based models

# Max memory for GPU
max_memory_map = {}
max_memory_map["cpu"] = "10000MiB"
max_memory_map[0] = "21000MiB"
m_mm_copy = max_memory_map.copy()
# if torch.cuda.is_available():
#     num_gpus = torch.cuda.device_count()
#     print(f"Found {num_gpus} GPU(s).")
#     for i in range(num_gpus):
#         total_gpu_memory_bytes = torch.cuda.get_device_properties(i).total_memory
#         half_gpu_memory_bytes = total_gpu_memory_bytes // 2
#         # Express in MiB for good granularity. Accelerate understands units like "MiB", "GiB", "MB", "GB".
#         half_gpu_memory_mib = half_gpu_memory_bytes // (1024 * 1024)
#         max_memory_map[i] = f"{half_gpu_memory_mib}MiB" # Use string key for device index
#     print(f"Calculated max_memory for Accelerate dispatch: {max_memory_map}")
# else:
#     print("CUDA is not available. Model will be loaded on CPU if possible, max_memory for GPU not set.")

model_load_dispatch_start_time = time.perf_counter_ns() / 1_000_000

model = load_checkpoint_and_dispatch(
    model,
    checkpoint=model_snapshot_path, # Path to the downloaded snapshot
    device_map="auto",              # Automatically distribute across GPU/CPU
    no_split_module_classes=no_split_modules,
    # offload_folder=offload_path,    # For offloading parts of the model to CPU RAM/disk
    dtype=None,                      # Use None to infer dtype from the checkpoint
    # max_memory=max_memory_map if max_memory_map else None # Pass the calculated memory map
)

model_load_dispatch_end_time = time.perf_counter_ns() / 1_000_000
model_load_dispatch_time = model_load_dispatch_end_time - model_load_dispatch_start_time

print("Model loaded and dispatched. Device map:")
for k, v in model.hf_device_map.items():
    print(f"- {k}: {v}")

model_tokenize_start_time = time.perf_counter_ns() / 1_000_000

input_text = "Once upon a time in a magical forest"

print(f"\nTokenizing input: '{input_text}'")

inputs = processor(text=input_text, return_tensors="pt").to('cuda')

model_tokenize_end_time = time.perf_counter_ns() / 1_000_000
model_tokenize_time = model_tokenize_end_time - model_tokenize_start_time

print("Generating text...")
try:
    # Start timing first inference
    first_inference_start_time = time.perf_counter_ns() / 1_000_000
    outputs = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    first_inference_end_time = time.perf_counter_ns() / 1_000_000
    print("\nGenerated text:")
    print(generated_text)
    
    # Calculate time to first inference
    time_to_first_inference = first_inference_end_time - first_inference_start_time
    
    # Measure throughput (samples per second)
    num_samples = 10  # Number of samples to measure throughput
    throughput_start_time = time.perf_counter_ns() / 1_000_000
    
    for _ in range(num_samples):
        outputs = model.generate(**inputs, max_new_tokens=100)
        _ = processor.decode(outputs[0], skip_special_tokens=True)
    
    throughput_end_time = time.perf_counter_ns() / 1_000_000
    throughput_time = throughput_end_time - throughput_start_time
    
    # Save measurements to times_file
    with open(times_file, 'w') as f:
        f.write(f"Model Loading Time: {model_load_snapshot_time/1000:.3f} s\n")
        f.write(f"Model Config Loading Time: {model_load_config_time/1000:.3f} s\n")
        f.write(f"Model Dispatch Time: {model_load_dispatch_time/1000:.3f} s\n")
        f.write(f"Model Tokenize Time: {model_tokenize_time/1000:.3f} s\n")
        f.write(f"Time to First Inference: {time_to_first_inference/1000:.3f} s\n")
        f.write(f"Throughput: {throughput_time} miliseconds\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Memory Map: {m_mm_copy}\n")
        f.write("Device Map:\n")
        for k, v in model.hf_device_map.items():
            f.write(f"- {k}: {v}\n")


except Exception as e:
    print(f"Error during text generation: {e}")

print("\nScript finished.")
stop_monitoring.set()
monitor_thread.join(timeout=2) 
