## Invertible Grayscale

We run this code under [TensorFlow](https://www.tensorflow.org) 1.6 on Ubuntu16.04 with python pakage IPL installed.

### Network Architecture

TensorFlow Implementation of our paper ["Invertible Grayscale"](http://menghanxia.github.io/papers/2018_Invertible_Grayscale_siga.pdf) accepted to SIGGRAPH ASIA 2018.

<div align="center">
	<img src="img/overview.jpg" width="90%">
</div>

### Results

<div align="center">
	<img src="img/examples.jpg" width="90%">
</div>

### Notice
- You can use any color image set as the training data of the network, as it is a self-supervised learning scheme. 
- The input image resolution is hard-coded in the Line:7~8 of [`model.py`](model.py), and you need to modify it to match your data resolution (only multiple of 4 is supported).

### Train
- Set the training hyperparameters in [`main.py`](./main.py).
- Download the pretrained VGG19 model in [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs).
- Start training by specifying the training dataset and validation dataset.
```bash
python3 main.py --mode 'train' --train_dir 'your_train_dir' --val_dir 'your_val_dir'
```

### Test
- Download the [pretrained model](https://drive.google.com/open?id=1wUKSzoYijU0dfyp9cl-9gTqyJY20OU2Y) and place it into the folder *'./checkpoints'*.
- Start evaluation by specifying the testing images and the result saving directory.
```bash
python3 main.py --mode 'test' --test_dir 'your_test_dir' --save_dir './results'
```

### Copyright and License
You are granted with the [license](./LICENSE.txt) for both academic and commercial usages.

### Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```
@article{XiaLW18,
  author    = {Menghan Xia and Xueting Liu and Tien-Tsin Wong},
  title     = {Invertible grayscale},
  journal   = {{ACM} Trans. Graph.},
  volume    = {37},
  number    = {6},
  pages     = {246:1--246:10},
  year      = {2018}
}
```
