# Deepcut

[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/rkcosmos/deepcut/blob/master/LICENSE) [![DOI](https://zenodo.org/badge/95091660.svg)](https://zenodo.org/badge/latestdoi/95091660)

A Thai word tokenization library using Deep Neural Network.

![model_structure](https://user-images.githubusercontent.com/1214890/58486992-14c1d880-8191-11e9-9122-8385750e06bd.png)

## What's new

* `v0.7.0` Migrate from keras to TensorFlow 2.0
* `v0.6.0` Allow excluding stop words and custom dictionary, updated weight with semi-supervised learning
* `v0.5.2` Better pretrained weight matrix
* `v0.5.1` Faster tokenization by code refactorization
* `examples` folder provide starter script for Thai text classification problem
* `DeepcutJS`, you can try tokenizing Thai text on web browser [here](https://rkcosmos.github.io/deepcut/)

## Performance

The Convolutional Neural network is trained from 90 % of NECTEC's BEST corpus (consists of 4 sections, article, news, novel and encyclopedia) and test on the rest 10 %. It is a binary classification model trying to predict whether a character is the beginning of word or not. The results calculated from only 'true' class are as follow

| Precision | Recall |   F1   |
| --------- | ------ | ------ |
| 97.8%     | 98.5%  | 98.1%  |

## Installation

Install using `pip` for stable release (tensorflow version2.0),

``` bash
pip install deepcut
```

For latest development release (recommended),

``` bash
pip install git+git://github.com/rkcosmos/deepcut.git
```

If you want to use tensorflow version 1.x and standalone keras, you will need

``` bash
pip install deepcut==0.6.1
```

### Docker

First, install and run [`docker`](https://www.docker.com/get-started) on your machine. Then, you can build and run `deepcut` as follows

``` bash
docker build -t deepcut:dev . # build docker image
docker run --rm -it deepcut:dev # run docker, -it flag makes it interactive, --rm for clean up the container and remove file system
```

This will open a shell for us to play with `deepcut`.

## Usage

``` python
import deepcut
deepcut.tokenize('ตัดคำได้ดีมาก')
```

Output will be in list format

``` bash
['ตัดคำ','ได้','ดี','มาก']
```

### Bag-of-word transformation

We implemented a tokenizer which works similar to [`CountVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) from `scikit-learn` . Here is an example usage:

``` python
from deepcut import DeepcutTokenizer
tokenizer = DeepcutTokenizer(ngram_range=(1,1),
                             max_df=1.0, min_df=0.0)
X = tokenizer.fit_tranform(['ฉันบินได้', 'ฉันกินข้าว', 'ฉันอยากบิน']) # 3 x 6 CSR sparse matrix
print(tokenizer.vocabulary_) # {'บิน': 0, 'ได้': 1, 'ฉัน': 2, 'อยาก': 3, 'ข้าว': 4, 'กิน': 5}, column index of sparse matrix

X_test = tokenizer.transform(['ฉันกิน', 'ฉันไม่อยากบิน']) # use built tokenizer vobalurary to transform new text
print(X_test.shape) # 2 x 6 CSR sparse matrix

tokenizer.save_model('tokenizer.pickle') # save the tokenizer to use later
```

You can load the saved tokenizer to use later

``` python
tokenizer = deepcut.load_model('tokenizer.pickle')
X_sample = tokenizer.transform(['ฉันกิน', 'ฉันไม่อยากบิน'])
print(X_sample.shape) # getting the same 2 x 6 CSR sparse matrix as X_test
```

### Custom Dictionary

User can add custom dictionary by adding path to `.txt` file with one word per line like the following.

``` bash
ขี้เกียจ
โรงเรียน
ดีมาก
```

The file can be placed as an `custom_dict` argument in `tokenize` function e.g.

``` python
deepcut.tokenize('ตัดคำได้ดีมาก', custom_dict='/path/to/custom_dict.txt')
deepcut.tokenize('ตัดคำได้ดีมาก', custom_dict=['ดีมาก']) # alternatively, you can provide a list of custom dictionary
```

## Notes

Some texts might not be segmented as we would expected (e.g.'โรงเรียน' -> ['โรง', 'เรียน']), this is because of

* BEST corpus (training data) tokenizes word this way (They use 'Compound words' as a criteria for segmentation)
* They are unseen/new words -> Ideally, this would be cured by having better corpus but it's not very practical so I am thinking of doing semi-supervised learning to incorporate new examples.

Any suggestion and comment are welcome, please post it in issue section.

## Contributors

* [Rakpong Kittinaradorn](https://github.com/rkcosmos)
* [Korakot Chaovavanich](https://github.com/korakot)
* [Titipat Achakulvisut](https://github.com/titipata)
* [Chanwit Kaewkasi](https://github.com/chanwit)

## Citations

If you use `deepcut` in your project or publication, please cite the library as follows

``` bash
Rakpong Kittinaradorn, Titipat Achakulvisut, Korakot Chaovavanich, Kittinan Srithaworn,
Pattarawat Chormai, Chanwit Kaewkasi, Tulakan Ruangrong, Krichkorn Oparad.
(2019, September 23). DeepCut: A Thai word tokenization library using Deep Neural Network. Zenodo. http://doi.org/10.5281/zenodo.3457707
```

or BibTeX entry:

``` bib
@misc{Kittinaradorn2019,
    author       = {Rakpong Kittinaradorn, Titipat Achakulvisut, Korakot Chaovavanich, Kittinan Srithaworn, Pattarawat Chormai, Chanwit Kaewkasi, Tulakan Ruangrong, Krichkorn Oparad},
    title        = {{DeepCut: A Thai word tokenization library using Deep Neural Network}},
    month        = Sep,
    year         = 2019,
    doi          = {10.5281/zenodo.3457707},
    version      = {1.0},
    publisher    = {Zenodo},
    url          = {http://doi.org/10.5281/zenodo.3457707}
}
```

## Partner Organizations

* True Corporation

We are open for contribution and collaboration.
