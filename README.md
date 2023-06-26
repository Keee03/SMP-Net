# Non-contact Multi-physiological Signals Estimation via Visible and Infrared Facial Features Fusion (SMP-Net)

## Network Structure：
![image](SMP-Net.png)

## How to train:
please check the code in main.py and data.py.
```
# use 2 GPUs to train the model
python -m torch.distributed.run --nproc_per_node=2 main.py
```
