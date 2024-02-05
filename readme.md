### ViT (Vision Transformer)

#### Introduction
This project is unofficial implementation for ViT network. 
Based on open source Python 3.7 and PyTorch 1.7.1, Vit network is created. 

All of our training and experiments are achieved at the sever which is equipped with 2 Tesla T4 GPUs.
The datasets to train our model is related to garbage classification, which can be downloaded on [2].
This dataset consist of two classes including organic garbage and recycle garbage.

To balance of our computing resources and predicting performance, our feeding images are resized by 128*128 resolutions.
And then we set up our model parameters being similar with ViT-Base presented in paper [3].
For feeding images, they are divided into patches of 16 * 16 pixels each.
Our Vit network consists of 12 layers of transformer encoder and for each of which contains 12 multi-head attention, MlP with 3072 size and 768 latent vector size.

Through 500 epoch training, our model reaches 0.883 accuracy on this dataset.

#### Folder Structure
- Config: It is a folder used to store ViT model's configuration, including the number of layers of transformer encoder, the size of the image scaling and so on.
- Dataset: It is a folder to store data processing approach which consist of data argumentation and data type conversion from PIL to tensor. 
- Model: It is a folder to store our model structure.
- VITTrainer.py: It is a file to train ViT model.

#### Usage

If you run training model from the command line, see the following.
```python
# --nproc_per_node: the number of Gpu you want to used.
# --ngpus_per_node: same as above.
python -m torch.distributed.launch --nproc_per_node=2   /home/ztp/workspace/semantic_segmentation/VIT-Pytorch/VITTrainer.py --seed=7 --multiprocessing_distributed=1 --ngpus_per_node=2 --config_file='./Config/GarbageCls_VIT_config.yml' 
```

If you run on kind of compilers such as pycharm, you need to set up some scrip parameters and environment variables.
```python
# scrip parameters
--multiprocessing_distributed=0  # it represents to launch multi-gpu in a way of torch.multiprocessing.spawn
--ngpus_per_node=2 
--seed=7

#envrioment variables
MASTER_ADDR=localhost;
MASTER_PORT=29501
```

#### Reference
- [1] https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/models/vit.py
- [2] https://www.cvmart.net/dataSets/detail/242?channel_id=op10&utm_source=cvmartmp&utm_campaign=datasets&utm_medium=article
- [3] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).