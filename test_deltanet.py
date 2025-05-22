import torch
import os
import numpy as np
from fla.ops.delta_rule import chunk_delta_rule
import time


filename = "debug_delta_fwd_8x8x128x16.txt"  # adjust as needed
os.makedirs("printouts", exist_ok=True)

with open(filename, "r") as f:
    floats = list(map(float, f.read().split()))

# Known dimensions (update if needed)
B, H, N, D = 8, 8, 128, 16
size = B * H * N * D
ITER = 10

q = torch.tensor(floats[0:size], dtype=torch.bfloat16, device='cuda').view(B, H, N, D)
k = torch.tensor(floats[size:2*size], dtype=torch.bfloat16, device='cuda').view(B, H, N, D)
v = torch.tensor(floats[2*size:3*size], dtype=torch.bfloat16, device='cuda').view(B, H, N, D)
o_ref = torch.tensor(floats[3*size:4*size], dtype=torch.bfloat16, device='cuda').view(B, H, N, D)

# Run FLA Triton-based implementation
# o_fla, _ = chunk_delta_rule(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
#                             beta=torch.full((B, N, H), 0.01, dtype=torch.bfloat16, device='cuda'),
#                             initial_state=None, output_final_state=False,
#                             head_first=False)

# Warmup
for _ in range(ITER):
    _ = chunk_delta_rule(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        beta=torch.full((B, N, H), 0.01, dtype=torch.bfloat16, device='cuda'),
        initial_state=None, output_final_state=False,
        head_first=False
    )

# Benchmark with loop
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(ITER):
    o_fla, _ = chunk_delta_rule(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        beta=torch.full((B, N, H), 0.01, dtype=torch.bfloat16, device='cuda'),
        initial_state=None, output_final_state=False,
        head_first=False
    )
end_event.record()

# Wait for the events to be recorded
torch.cuda.synchronize()

elapsed_time_ms = start_event.elapsed_time(end_event) / ITER
print(f"[Triton] Elapsed time: {elapsed_time_ms:.3f} ms")


# Compare
abs_diff = (o_fla - o_ref.transpose(1, 2)).abs()
print("Max abs diff:", abs_diff.max().item())
print("Mean abs diff:", abs_diff.mean().item())

o_fla_flat = o_fla.flatten().tolist()
o_ref_flat = o_ref.transpose(1, 2).flatten().tolist()

with open("printouts/o.txt", "w") as f:
    for val in o_fla_flat:
        f.write(f"{val} ")

with open("printouts/o_ref.txt", "w") as f:
    for val in o_ref_flat:
        f.write(f"{val} ")

with open("printouts/diff.txt", "w") as f:
    for i, (a, b) in enumerate(zip(o_fla_flat, o_ref_flat)):
        if i % 128 == 0:
            f.write("\n")
        f.write(f"{a - b} ")

print("Saved outputs and diffs to printouts")

assert abs_diff.max() < 1e-4, "Too much divergence!"
print("PyTorch and Triton match within tolerance.")
