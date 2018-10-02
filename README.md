# Deepcut

[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/rkcosmos/deepcut/blob/master/LICENSE)

A Thai word tokenization library using Deep Neural Network.

## What's new?

* Deepcut JS, try tokenizing Thai text on browser [here](https://rkcosmos.github.io/deepcut/)
* `v0.6` Add stop words, updated weight with semi-supervised learning, custom dictionary
* `v0.5.2` Better weight matrix
* `v0.5.1` Faster tokenization by code refactorization from our new contributor: Titipat Achakulvisut

## Performance

The Convolutional Neural network is trained from 90 % of NECTEC's BEST corpus
(consists of 4 sections, article, news, novel and encyclopedia) and test on the rest 10 %.
It is a binary classification model trying to predict whether a character is the beginning of word or not.
The results calculated from only 'true' class are as follow

* f1 score:  98.1%
* precision score:  97.8%
* recall score:  98.5%

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

We do not add `tensorflow` in automatic installation process because it has cpu and gpu version.
Installing cpu version to everyone might break those who already have gpu version installed.
So please install `tensorflow` yourself following this guildline https://www.tensorflow.org/install/.


### Docker

Install Docker on your machine

For Linux:
```bash
curl -sSL https://get.docker.com | sudo sh
docker build -t deepcut .
```

For other OS: see https://docs.docker.com/engine/installation/

To run this Docker image:

```bash
docker run --rm -it deepcut
```

It will open a shell for us to play with deepcut.

## Usage

```python
import deepcut
deepcut.tokenize('ตัดคำได้ดีมาก')
```

Output will be in list format

```bash
['ตัดคำ','ได้','ดี','มาก']
```

#### Bag-of-word transformation

We implemented tokenizer which works similar to
[`CountVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) from `scikit-learn`.
Here is an example usage:

```python
from deepcut import DeepcutTokenizer
tokenizer = DeepcutTokenizer(ngram_range=(1,1),
                             max_df=1.0, min_df=0.0)
X = tokenizer.fit_tranform(['ฉันบินได้', 'ฉันกินข้าว', 'ฉันอยากบิน']) # 3 x 4 CSR sparse matrix
print(tokenizer.vocabulary_) # {'กิน': 0, 'ข้าว': 3, 'อยาก': 1, 'ได้': 2}
```


#### Custom Dictionary

User can add custom dictionary by adding path to `.txt` file with one word per line like the following.

```bash
ขี้เกียจ
โรงเรียน
```

The file can be placed as an argument in `tokenize` function e.g.

```python
deepcut.tokenize('ตัดคำได้ดีมาก', custom_dict='/path/to/custom_dict.txt')
deepcut.tokenize('ตัดคำได้ดีมาก', custom_dict=['ดีมาก']) # alternatively, you can provide a list of custom dictionary
```


## Notes

Some texts might not be segmented as we would expected (e.g. 'โรงเรียน' -> ['โรง', 'เรียน']), this is because of

* BEST corpus (training data) tokenizes word this way (They use 'Compound words' as a criteria for segmentation)

* They are unseen/new words -> Ideally, this would be cured by having better corpus but it's not very practical so I am thinking of doing semi-supervised learning to incorporate new examples.

Any suggestion and comment are welcome, please post it in issue section.


## Contributors

* [Rakpong Kittinaradorn](https://github.com/rkcosmos)
* [Korakot Chaovavanich](https://github.com/korakot)
* [Titipat Achakulvisut](https://github.com/titipata)
* [Chanwit Kaewkasi](https://github.com/chanwit)


## Partner Organizations

* True Corporation

And we are open for contribution and collaboration.
