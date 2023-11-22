# SRNet
institution：JMAI
created by：fuchenli
date：2023.11.22
E-mail：lfchen@yeah.net

This work creates a deep learning framework for low-level vision, using super-resolution networks as an example.

It is easy to change all different network for train,test,inference and so on.

![image](https://github.com/andre20000131/SRNet/assets/95755599/236c8fa5-e167-4341-9314-2b4ed38db7fc)

The main network structure is shown in the figure, with the resnet series, unet series, and their improvements all under the codes path.

### Configuration
‘’‘
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
‘’‘

### test
- Modify `dataroot_LQ` and `pretrain_model_G` (you can also use the pretrained model which is provided in the `./pretrained_model`) in `./codes/options/test/test_net.yml`, then run python test.py

### train
- Prepare the data. Modify `input_folder` and `save_folder` in `./scripts/extract_subimgs_single.py`, then run
```
cd scripts
python extract_subimgs_single.py
```
This process is quite resource intensive, please make sure there are enough disks to carry it.


- Modify `dataroot_LQ` and `dataroot_GT` in `./codes/options/train/train_net.yml`, then run
```
cd codes
python train.py -opt options/train/train_net.yml
```
The models and training states will be saved to `./experiments/name`.



## Acknowledgment
The code is inspired by [BasicSR](https://github.com/xinntao/BasicSR).




