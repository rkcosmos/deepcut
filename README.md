# Deepcut
A Thai word tokenization library using Deep Neural Network

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

# Contributors

* Rakpong Kittinaradorn
* Korakot Chaovavanich
