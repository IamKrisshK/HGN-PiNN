pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-geometric==2.5.3
pip install nvidia-physicsnemo==2.0.0
pip install hydra-core==1.3.2 omegaconf==2.3.0 tqdm pyyaml
pip install wandb

python - <<'EOF'
import torch
import torch_geometric
import torch_scatter
import torch_sparse
import torch_cluster
import torch_spline_conv
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyG:", torch_geometric.__version__)
print("torch_scatter OK")
print("torch_sparse OK")
print("torch_cluster OK")
print("torch_spline_conv OK")
print("PhysicsNeMo OK")
print("MeshGraphKAN import OK")

print("========== All Deps Downloaded ==========")
EOF
