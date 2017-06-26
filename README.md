# Deepcut
A Thai word tokenization library using Deep Neural Network.

# Performance

The Convolutional Neural network is trained from 90% of NECTEC's BEST corpus(consists of 4 sections, article, news, novel and wikipedia) and test on the rest 10%. It is a binary classification model trying to predict whether a character is the beginning of word or not. The results calculated from only 'true' class are as follow

* f1 score:  98.8%
* precision score:  98.6%
* recall score:  99.1%

# Installation

On terminal, just type
```
  pip install deepcut
```  
Make sure you are using tensorflow backend in keras by making sure ~/.keras/keras.json is as follows (see also https://keras.io/backend/)
```  
  {
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_last"
  }
```

# Usage

```
import deepcut
deepcut.tokenize('ตัดคำได้ดีมาก')
```

output will be in list format

```
['ตัด','คำ','ได้','ดี','มาก']
```

# Notes

Some texts might not be segmented as we would expected (e.g. 'โรงเรียน' -> ['โรง', 'เรียน']), this is because of

* BEST corpus (training data) tokenizes word this way 

* They are unseen/new words -> Ideally, this would be cured by having better corpus but it's not very practical so I am thinking of doing semi supervised learning to incorporate new examples.

# Contributors

* Rakpong Kittinaradorn
* Korakot Chaovavanich
