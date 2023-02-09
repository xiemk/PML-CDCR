# A Deep Model for Partial Multi-label Image Classification with Curriculum Based Disambiguation

The Pytorch implementation for the paper [A Deep Model for Partial Multi-label Image Classification with Curriculum Based Disambiguation](http://www.xiemk.pro/publication/arxiv-cdcr-preprint.pdf) (arXiv 2022)

See much related works in [Awesome Weakly Supervised Multi-label Learning!](https://github.com/milkxie/awesome-weakly-supervised-multi-label-learning)

## Preparing Data

See the `README.md` file in the `data` directory for instructions on downloading and preparing the datasets.

## Training Model
To train and evaluate a model, the next two steps are required:

1. For the first stage, we warm-up the model with the BCE loss on partially labeled images. Run:
```
CUDA_VISIBLE_DEVICES=gpu_ids python run_warmup.py --noise_rate=0.05 --data=./data
```

2. For the second stage, we train the model by adding curriculum based disambiguation and consistency regularization.
```
CUDA_VISIBLE_DEVICES=gpu_ids python run_cdcr.py --noise_rate=0.05 --data=./data
```


## Reference
If you find the code useful in your research, please consider citing our paper:
```
@article{sun2022deep,
  title={A Deep Model for Partial Multi-Label Image Classification with Curriculum Based Disambiguation},
  author={Sun, Feng and Xie, Ming-Kun and Huang, Sheng-Jun},
  journal={arXiv preprint arXiv:2207.02410},
  year={2022}
}
```