# JointSLU: Joint Semantic Parsing for Spoken/Natural Language Understanding


*A Keras implementation of the models described in [Hakkani-Tur et al. (2016)] (https://www.csie.ntu.edu.tw/~yvchen/doc/IS16_MultiJoint.pdf).*

This model learns various RNN architectures (RNN, GRU, LSTM, etc.) for joint semantic parsing, 
where intent prediction and slot filling are performed in a single network model.

## Content
* [Requirements](#requirements)
* [Getting Started](#getting-started)
* [Contact](#contact)
* [Reference](#reference)

## Requirements
1. Python
2. Numpy `pip install numpy`
3. Keras and associated Theano or TensorFlow `pip install keras`
4. H5py `pip install h5py`

## Dataset
1. Train: word sequences with IOB slot tags and the intent label (data/atis.train.w-intent.iob)
2. Test: word sequences with IOB slot tags and the intent label (data/atis.test.w-intent.iob)


## Getting Started
You can train and test JointSLU with the following commands:

```shell
  git clone --recursive https://github.com/yvchen/JointSLU.git
  cd JointSLU
```
You can run a sample tutorial with this command:
```shell
  bash script/run_sample.sh rnn theano 0 | sh
```
Then you can see the predicted result in `sample/rnn+emb_H-50_O-adam_A-tanh_WR-embedding.test.3`.

## Contact
Yun-Nung (Vivian) Chen, y.v.chen@ieee.org

## Reference

Main papers to be cited
```
@Inproceedings{hakkani-tur2016multi,
  author    = {Hakkani-Tur, Dilek and Tur, Gokhan and Celikyilmaz, Asli and Chen, Yun-Nung and Gao, Jianfeng and Wang, Ye-Yi},
  title     = {Multi-Domain Joint Semantic Frame Parsing using Bi-directional RNN-LSTM},
  booktitle = {Proceedings of Interspeech},
  year      = {2016}
}


