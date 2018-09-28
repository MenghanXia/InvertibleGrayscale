## Invertible Grayscale

We run this code under [TensorFlow](https://www.tensorflow.org) 1.6 on Ubuntu16.04 with python pakage IPL installed.

### Network Architecture

TensorFlow Implementation of our paper ["Invertible Grayscale"](https://arxiv.org/abs/1609.04802) accepted to SIGGRAPH ASIA 2018.

<div align="center">
	<img src="img/overview.jpg" width="80%" height="10%"/>
</div>

### Results

<div align="center">
	<img src="img/examples.jpg" width="80%" height="50%"/>
</div>

### Prepare Data

- You can use any color image set as the training set of the network, as it is a self-supervised learning. 
- The patch size is set to 256x256 in the model.py (you can also change it to any other size as you like).
- Download the pretrained VGG19 model in [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs).

### Run
- Set your image folder in `config.py`, if you download [DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/) dataset, you don't need to change it. 
- Other links for DIV2K, in case you can't find it : [test\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_bicubic_X4.zip), [train_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), [train\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip), [valid_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_valid_HR.zip), [valid\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip).

```python
config.TRAIN.img_path = "your_image_folder/"
```

- Start training.

```bash
python main.py
```

- Start evaluation. ([pretrained model](https://github.com/tensorlayer/srgan/releases/tag/1.2.0) for DIV2K)

```bash
python main.py --mode=evaluate 
```