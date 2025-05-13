import torch
import torch.nn as nn
import time
import copy

# Configuration
INPUT_SIZE = 1024
HIDDEN_SIZE = 2048 # Increased hidden size to make activations larger
OUTPUT_SIZE = 512
BATCH_SIZE = 256 # Increased batch size
NUM_ITERATIONS = 100 # Number of iterations for timing
WARMUP_ITERATIONS = 10 # Iterations to warmup GPU

# Check for GPU availability
if not torch.cuda.is_available():
    print("CUDA (GPU) is not available. This demo requires a GPU.")
    exit()

device = torch.device("cuda:0")
cpu_device = torch.device("cpu")

# 1. Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Model for standard GPU execution
model_gpu = SimpleNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
model_gpu.eval() # Set to evaluation mode, we are only timing inference

# Model for activation offloading (deep copy to have separate weights)
model_offload = copy.deepcopy(model_gpu).to(device)
model_offload.eval()

# Generate random input data
dummy_input = torch.randn(BATCH_SIZE, INPUT_SIZE, device=device)

# --- Scenario 1: Standard GPU Execution ---

def run_standard_gpu(model, data, iterations):
    timings = []
    for i in range(iterations + WARMUP_ITERATIONS):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize() # Ensure previous GPU work is done
        start_event.record()

        _ = model(data)

        end_event.record()
        torch.cuda.synchronize() # Wait for the ops to complete

        if i >= WARMUP_ITERATIONS:
            timings.append(start_event.elapsed_time(end_event))
    return sum(timings) / iterations

print("Running Standard GPU Execution...")
avg_time_gpu = run_standard_gpu(model_gpu, dummy_input, NUM_ITERATIONS)
print(f"Average time per iteration (GPU only): {avg_time_gpu:.6f} ms\n")

# --- Scenario 2: Manual Activation Offloading (GPU + CPU) ---

class SimpleNetWithOffloading(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, offload_device):
        super(SimpleNetWithOffloading, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.offload_device = offload_device
        self.activation_on_cpu = None # To store the offloaded activation

    def forward(self, x):
        # Layer 1: Compute on GPU, then offload activation
        x = self.fc1(x)
        x = self.relu1(x)
        self.activation_on_cpu = x.to(self.offload_device) # Offload to CPU
        # print(f"Activation 1 offloaded to: {self.activation_on_cpu.device}")

        # Simulate some other operations or just the delay of having it on CPU
        # For this demo, we immediately bring it back for the next layer

        # Layer 2: Bring activation back to GPU, compute on GPU
        x_gpu = self.activation_on_cpu.to(x.device) # Bring back to original device (GPU)
        # print(f"Activation 1 moved back to: {x_gpu.device}")
        x = self.fc2(x_gpu)
        x = self.relu2(x)
        # Offload activation 2 (optional, for demonstration)
        # self.activation2_on_cpu = x.to(self.offload_device)
        # x_gpu_2 = self.activation2_on_cpu.to(x.device)

        # Layer 3: Compute on GPU
        x = self.fc3(x) # or x = self.fc3(x_gpu_2) if second offload was done
        return x

model_manual_offload = SimpleNetWithOffloading(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, cpu_device).to(device)
model_manual_offload.eval()

# Ensure weights are the same if desired for direct comparison (already handled by previous deepcopy if structure is identical)
# For this custom class, we need to load state dict if we want identical weights
model_manual_offload.fc1.load_state_dict(model_gpu.fc1.state_dict())
model_manual_offload.fc2.load_state_dict(model_gpu.fc2.state_dict())
model_manual_offload.fc3.load_state_dict(model_gpu.fc3.state_dict())


def run_manual_offloading(model, data, iterations):
    timings = []
    for i in range(iterations + WARMUP_ITERATIONS):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        _ = model(data) # Forward pass with manual offloading inside

        end_event.record()
        torch.cuda.synchronize()
        if i >= WARMUP_ITERATIONS:
            timings.append(start_event.elapsed_time(end_event))
    return sum(timings) / iterations

print("Running Manual Activation Offloading (GPU + CPU)...")
avg_time_offload = run_manual_offloading(model_manual_offload, dummy_input, NUM_ITERATIONS)
print(f"Average time per iteration (with manual offloading): {avg_time_offload:.6f} ms\n")

# --- Comparison ---
print("--- Comparison Summary ---")
print(f"Average time per iteration (GPU only):           {avg_time_gpu:.6f} ms")
print(f"Average time per iteration (Manual Offloading):  {avg_time_offload:.6f} ms")

if avg_time_gpu < avg_time_offload:
    print("\nStandard GPU execution was faster.")
    print(f"Offloading introduced an overhead of approximately {avg_time_offload - avg_time_gpu:.6f} ms per iteration.")
else:
    print("\nManual activation offloading was faster (this is unexpected for small models).")

print("\nNote: Manual activation offloading, as implemented here, often introduces significant")
print("overhead due to CPU-GPU data transfers for each offloaded activation.")
print("Benefits are typically seen in memory-constrained scenarios with very large models,")
print("and are often managed by more sophisticated libraries (e.g., FSDP, DeepSpeed).")
print("This demo illustrates the mechanics and potential cost of manual transfer.")


# Running Standard GPU Execution...
# Average time per iteration (GPU only): 0.276884 ms

# Running Manual Activation Offloading (GPU + CPU)...
# Average time per iteration (with manual offloading): 0.739320 ms

# --- Comparison Summary ---
# Average time per iteration (GPU only):           0.276884 ms
# Average time per iteration (Manual Offloading):  0.739320 ms

# Standard GPU execution was faster.
# Offloading introduced an overhead of approximately 0.462436 ms per iteration.

# Note: Manual activation offloading, as implemented here, often introduces significant
# overhead due to CPU-GPU data transfers for each offloaded activation.
# Benefits are typically seen in memory-constrained scenarios with very large models,
# and are often managed by more sophisticated libraries (e.g., FSDP, DeepSpeed).
# This demo illustrates the mechanics and potential cost of manual transfer.