# Deepcut

[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/rkcosmos/deepcut/blob/master/LICENSE)

A Thai word tokenization library using Deep Neural Network.

## What's new on version 0.5.1.0

Faster tokenization by code refactorization from our new contributor: Titipat Achakulvisut

## Performance

The Convolutional Neural network is trained from 90% of NECTEC's BEST corpus(consists of 4 sections, article, news, novel and encyclopedia) and test on the rest 10%. It is a binary classification model trying to predict whether a character is the beginning of word or not. The results calculated from only 'true' class are as follow

* f1 score:  98.8%
* precision score:  98.6%
* recall score:  99.1%

## Installation

Install using `pip` for stable release,

```bash
pip install deepcut
```

For latest development release,

```bash
pip install git+git://github.com/rkcosmos/deepcut.git
```

Or clone the repository and install using `setup.py`

```bash
python setup.py install
```

Make sure you are using `tensorflow` backend in `Keras` by making sure `~/.keras/keras.json` is as follows (see also https://keras.io/backend/)

```bash
  {
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_last"
  }
```

## Usage

```python
import deepcut
deepcut.tokenize('ตัดคำได้ดีมาก')
```

Output will be in list format

```bash
['ตัด','คำ','ได้','ดี','มาก']
```

## Notes

Some texts might not be segmented as we would expected (e.g. 'โรงเรียน' -> ['โรง', 'เรียน']), this is because of

* BEST corpus (training data) tokenizes word this way (They use 'Compound words' as a criteria for segmentation)

* They are unseen/new words -> Ideally, this would be cured by having better corpus but it's not very practical so I am thinking of doing semi-supervised learning to incorporate new examples.

Any suggestion and comment are welcome, please post it in issue section.

## Contributors

* Rakpong Kittinaradorn
* Korakot Chaovavanich
* Titipat Achakulvisut

And we are open for contribution and collaboration.
