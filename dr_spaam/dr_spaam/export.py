import torch
import executorch
from dr_spaam.model.dr_spaam import DrSpaam
import numpy as np
from dr_spaam.utils import utils as u
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export

# Number of scan points from the LD19 LiDAR
NUM_SCAN_POINTS = 504

model = DrSpaam(
                    dropout=0.5,
                num_pts=56,
                embedding_length=128,
                alpha=0.5,
                window_size=17,
                panoramic_scan=True,
                cls_loss=None,
                mixup_alpha=0.0,
                mixup_w=0.0,
            )
path = "/media/nilum/my-stuff/Research/Human_Robot_Interaction/proactive-social-nav/forg_dataset/dr_spaam_5_on_frog.pth"
model.load_state_dict(torch.load(path, map_location="cpu")["model_state"])
model.eval()

# Initialize memory by running one dummy pass first
with torch.no_grad():
    init_scan = np.ones((1, NUM_SCAN_POINTS), dtype=np.float32) * 5.0
    init_phi = np.linspace(-np.pi, np.pi, NUM_SCAN_POINTS, dtype=np.float32)
    init_ct = u.scans_to_cutout(
        init_scan, init_phi, stride=1, centered=True, fixed=True,
        window_width=1.0, window_depth=0.5, num_cutout_pts=56,
        padding_val=29.99, area_mode=True,
    )
    init_tensor = torch.from_numpy(init_ct).float().unsqueeze(0)
    model(init_tensor, inference=True)  # seeds the internal memory

dummy_scan = np.ones((1, NUM_SCAN_POINTS), dtype=np.float32)*5.0
dummy_phi = np.linspace(-np.pi, np.pi, NUM_SCAN_POINTS, dtype=np.float32)
dummy_ct = u.scans_to_cutout(
                dummy_scan,
                dummy_phi,
                stride=1,
                centered=True,
                fixed=True,
                window_width=1.0,
                window_depth=0.5,
                num_cutout_pts=56,
                padding_val=29.99,
                area_mode=True,
            )

dummy_tensor = torch.from_numpy(dummy_ct).float().unsqueeze(0)
print(f"Export input shape: {dummy_tensor.shape}")

exported_program = export(
    model,
    (dummy_tensor,)
)

executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner=[XnnpackPartitioner()],
).to_executorch()

output_path = "model.pte"
with open(output_path, "wb") as file:
    file.write(executorch_program.buffer)
print(f"Exported to {output_path}")