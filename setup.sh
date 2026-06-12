# conda create -n ccc-grpo python=3.11
# conda activate ccc-grpo

cd src/open-r1-multimodal
pip install -e ".[dev]"

pip install wandb==0.18.3
pip install qwen_vl_utils torchvision
pip install flash_attn==2.7.4.post1
pip install scikit-learn
