## JointSLU: Spoken Language Understanding with Joint Semantic Parsing

Yun-Nung (Vivian) Chen, y.v.chen@ieee.org

This model learns various RNN models (RNN, GRU, LSTM, etc.) for joint semantic parsing.
The intent and slots are tagged in a single network model.

### Requirements
1. Python
2. Numpy
3. Scipy
4. Keras
5. H5py

### Input data
1. Train: word sequences with IOB slot tags and the intent label (data/atis.train.w-intent.iob)
2. Test: word sequences with IOB slot tags and the intent label (data/atis.test.w-intent.iob)


### Reference

Main papers to be cited
```
@Inproceedings{hakkani-tur2016multi,
  author    = {Hakkani-Tur, Dilek and Tur, Gokhan and Celikyilmaz, Asli and Chen, Yun-Nung and Gao, Jianfeng and Wang, Ye-Yi},
  title     = {Multi-Domain Joint Semantic Frame Parsing using Bi-directional RNN-LSTM},
  booktitle = {Proceedings of Interspeech},
  year      = {2016}
}


