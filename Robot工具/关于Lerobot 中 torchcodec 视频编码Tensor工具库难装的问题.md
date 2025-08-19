* torchcodec表格来源链接：（https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec）
![[img_v3_02p3_1aca9b99-bb1c-4a8f-9b30-acdb223f032g.jpg]]
```bash
# 更新lerobot
cd lerobot/
pip3 install -e .

# 先确认对cuda下的torchcodec版本
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 根据上表去找torchcodec对应的torch版本
pip install torchcodec --index-url=https://download.pytorch.org/whl/cu124

# 补全一下ffmpeg
conda install ffmpeg -c conda-forge
```

