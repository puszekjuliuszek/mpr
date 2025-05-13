import torch
import torch.nn as nn
import time
import copy

# Configuration
INPUT_SIZE = 1024
HIDDEN_SIZE = 4096  # Large hidden size to make activations significant
OUTPUT_SIZE = 512
BATCH_SIZE = 256
N_LAYERS = 3 # Number of linear layers in our custom block
NUM_ITERATIONS = 50
WARMUP_ITERATIONS = 5

# Define the device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(DEVICE)}")
else:
    DEVICE = torch.device("cpu")
    print("CUDA not available, using CPU. Comparison might not be meaningful.")

# --- Model Definition ---
class SimpleBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()
        layers = [SimpleBlock(input_size, hidden_size)]
        for _ in range(n_layers - 1):
            layers.append(SimpleBlock(hidden_size, hidden_size))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def forward_with_checkpointing(self, x, pack_hook=None, unpack_hook=None):
        # We will checkpoint the 'features' part of the model
        # For simplicity, if using torch.utils.checkpoint.checkpoint_sequential,
        # it expects a list of modules.
        # Alternatively, torch.utils.checkpoint.checkpoint can wrap any function.

        # Here, we'll use torch.utils.checkpoint.checkpoint for each module in self.features
        # to demonstrate the hook usage. A more common use would be on larger segments.

        def custom_checkpoint_fwd(module, *inputs):
            return module(*inputs)

        for i, layer_module in enumerate(self.features):
            if pack_hook and unpack_hook: # Apply checkpointing only if hooks are provided
                x = torch.utils.checkpoint.checkpoint(
                    custom_checkpoint_fwd,
                    layer_module,
                    x,
                    use_reentrant=False, # Recommended for custom hooks
                    pack_hook=pack_hook,
                    unpack_hook=unpack_hook
                )
            else: # Standard forward
                x = layer_module(x)
        x = self.classifier(x)
        return x

# --- Activation Offloading Hooks ---
# These hooks are called by torch.utils.checkpoint when saving and loading activations
# for the backward pass.

# Global dictionary to store offloaded activations (for simplicity in this example)
# In a more complex scenario, you might manage this differently.
offloaded_activations_store = {}
activation_id_counter = 0

def pack_hook(tensor):
    """Moves tensor to CPU and stores it."""
    global activation_id_counter, offloaded_activations_store
    if tensor.is_cuda: # Only offload if it's on GPU
        # print(f"Packing (Offloading) tensor ID: {activation_id_counter} to CPU. Shape: {tensor.shape}")
        offloaded_activations_store[activation_id_counter] = tensor.cpu()
        current_id = activation_id_counter
        activation_id_counter += 1
        return current_id # Return an identifier for the tensor
    # print(f"Skipping pack for tensor on {tensor.device}")
    return tensor # If not on CUDA or some other case, return tensor itself (no offload)


def unpack_hook(packed_tensor_id_or_tensor):
    """Moves tensor back to GPU from CPU storage."""
    global offloaded_activations_store
    if isinstance(packed_tensor_id_or_tensor, int): # It's an ID, so it was offloaded
        tensor_id = packed_tensor_id_or_tensor
        # print(f"Unpacking (Loading) tensor ID: {tensor_id} to GPU.")
        tensor_cpu = offloaded_activations_store.pop(tensor_id)
        return tensor_cpu.to(DEVICE, non_blocking=True)

    # print(f"Skipping unpack for tensor not identified by ID (already on device or not offloaded).")
    return packed_tensor_id_or_tensor # If it's the tensor itself, return it


# --- Timing Function ---
def measure_time(model, input_data, criterion, optimizer, use_offloading=False, desc=""):
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize() # Ensure previous CUDA ops are done

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warm-up iterations
    for _ in range(WARMUP_ITERATIONS):
        optimizer.zero_grad()
        if use_offloading:
            global offloaded_activations_store, activation_id_counter
            offloaded_activations_store = {} # Reset store for each fwd/bwd
            activation_id_counter = 0
            output = model.forward_with_checkpointing(input_data, pack_hook, unpack_hook)
        else:
            output = model(input_data)
        target = torch.randn_like(output).to(DEVICE)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    total_time = 0
    for i in range(NUM_ITERATIONS):
        optimizer.zero_grad()

        if DEVICE.type == 'cuda':
            start_event.record()

        # Reset offloading store for each iteration if offloading
        if use_offloading:
            global offloaded_activations_store, activation_id_counter
            offloaded_activations_store = {}
            activation_id_counter = 0
            output = model.forward_with_checkpointing(input_data, pack_hook, unpack_hook)
        else:
            output = model(input_data)

        # Create a dummy target for loss calculation
        target = torch.randn_like(output).to(DEVICE)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if DEVICE.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize() # Wait for all kernels to complete
            iter_time = start_event.elapsed_time(end_event) / 1000.0 # milliseconds to seconds
        else: # CPU timing
            # For CPU, we'd use time.time(), but since we are comparing with GPU,
            # we'll keep the structure similar. However, event timing is for CUDA.
            # This part is tricky if DEVICE is CPU for torch.cuda.Event.
            # The code is primarily designed for GPU comparison.
            # If on CPU, this timing won't be accurate or meaningful for comparison.
            # A simple time.time() wrapper would be better for pure CPU.
            # For this script, we assume meaningful comparison is on GPU.
            if i == 0 and WARMUP_ITERATIONS == 0 : print("Warning: CUDA Event timing is for GPU. CPU timing might not be accurate here.")
            iter_time = 0 # Placeholder if not on CUDA and no proper CPU timer implemented here

        total_time += iter_time
        if (i + 1) % 10 == 0:
            print(f"{desc} - Iteration [{i+1}/{NUM_ITERATIONS}], Avg Time: {total_time / (i+1):.6f} s")


    avg_time = total_time / NUM_ITERATIONS
    print(f"--- Average execution time for {desc}: {avg_time:.6f} seconds ---")
    return avg_time

# --- Main Execution ---
if __name__ == "__main__":
    # Create dummy input data and move to device
    dummy_input = torch.randn(BATCH_SIZE, INPUT_SIZE).to(DEVICE)

    # --- Scenario 1: Without Activation Offloading ---
    print("\nRunning WITHOUT Activation Offloading...")
    model_no_offload = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS).to(DEVICE)
    # Ensure parameters are distinct for fair comparison if model is modified in-place
    # model_no_offload_state = copy.deepcopy(model_no_offload.state_dict())

    optimizer_no_offload = torch.optim.Adam(model_no_offload.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    time_no_offload = measure_time(model_no_offload, dummy_input, criterion, optimizer_no_offload,
                                   use_offloading=False, desc="Without Offloading")

    # --- Scenario 2: With Activation Offloading ---
    print("\nRunning WITH Activation Offloading...")
    # Re-initialize or deepcopy model and optimizer to ensure fresh state and same initial weights
    model_with_offload = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS).to(DEVICE)
    # model_with_offload.load_state_dict(model_no_offload_state) # Start with same weights

    optimizer_with_offload = torch.optim.Adam(model_with_offload.parameters(), lr=1e-3)
    # Criterion remains the same

    # Clear global store before starting
    offloaded_activations_store = {}
    activation_id_counter = 0

    time_with_offload = measure_time(model_with_offload, dummy_input, criterion, optimizer_with_offload,
                                     use_offloading=True, desc="With Offloading")

    # --- Results ---
    print("\n--- Comparison Summary ---")
    print(f"Average time WITHOUT activation offloading: {time_no_offload:.6f} seconds")
    print(f"Average time WITH activation offloading:    {time_with_offload:.6f} seconds")

    if time_no_offload < time_with_offload:
        print("Standard approach (without offloading) was FASTER.")
        print(f"Offloading was {time_with_offload / time_no_offload:.2f}x slower.")
    elif time_with_offload < time_no_offload:
        print("Activation offloading approach was FASTER.")
        print(f"Offloading was {time_no_offload / time_with_offload:.2f}x faster.")
    else:
        print("Both approaches had similar execution times.")

    if DEVICE.type == 'cuda':
        print(f"\nNote: Memory usage not directly measured here, but offloading aims to reduce GPU memory at the cost of speed.")
        print(f"Max memory allocated (Without Offload): {torch.cuda.max_memory_allocated(DEVICE)/1e9:.3f} GB")
        torch.cuda.reset_peak_memory_stats(DEVICE) # Reset for next measurement
        # To get memory for the offloading part accurately, you'd need to run it again
        # or integrate memory profiling into the measure_time function carefully.
        # For simplicity, we'll just show an example of how to get it.
        print(f"After reset, current max memory: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.3f} GB")
        print("To properly compare memory, profile memory within each timed loop or use PyTorch Profiler.")

    print("\nRemember: The benefit of activation offloading is primarily GPU memory savings, which might allow for larger models/batches, potentially at the cost of speed due to CPU-GPU data transfers.")

# Using GPU: NVIDIA GeForce RTX 3090

# Running WITHOUT Activation Offloading...
# Without Offloading - Iteration [10/50], Avg Time: 0.006865 s
# Without Offloading - Iteration [20/50], Avg Time: 0.006802 s
# Without Offloading - Iteration [30/50], Avg Time: 0.006793 s
# Without Offloading - Iteration [40/50], Avg Time: 0.006760 s
# Without Offloading - Iteration [50/50], Avg Time: 0.006763 s
# --- Average execution time for Without Offloading: 0.006763 seconds ---

# Running WITH Activation Offloading...
# With Offloading - Iteration [10/50], Avg Time: 0.007997 s
# With Offloading - Iteration [20/50], Avg Time: 0.008004 s
# With Offloading - Iteration [30/50], Avg Time: 0.008013 s
# With Offloading - Iteration [40/50], Avg Time: 0.008052 s
# With Offloading - Iteration [50/50], Avg Time: 0.008026 s
# --- Average execution time for With Offloading: 0.008026 seconds ---

# --- Comparison Summary ---
# Average time WITHOUT activation offloading: 0.006763 seconds
# Average time WITH activation offloading:    0.008026 seconds
# Standard approach (without offloading) was FASTER.
# Offloading was 1.19x slower.

# Note: Memory usage not directly measured here, but offloading aims to reduce GPU memory at the cost of speed.
# Max memory allocated (Without Offload): 1.455 GB
# After reset, current max memory: 1.294 GB
# To properly compare memory, profile memory within each timed loop or use PyTorch Profiler.

# Remember: The benefit of activation offloading is primarily GPU memory savings, which might allow for larger models/batches, potentially at the cost of speed due to CPU-GPU data transfers.