pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install torch-geometric==2.5.3
pip install nvidia-physicsnemo==2.0.0
pip install hydra-core==1.3.2 omegaconf==2.3.0 tqdm pyyaml
pip install wandb

python - <<EOF
import torch
import torch_geometric
import physicsnemo
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyG:", torch_geometric.__version__)
print("PhysicsNeMo imported successfully")
EOF
