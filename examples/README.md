# Examples

In this section, we collect some of the example how to use `deepcut` for classification problem.
This aims to use as an starter-kit for Thai text classification, not state-of-the-art classification.

## True Voice Intent Classification

Example of script for True Voice Intent Classification problem from [`PyThaiNLP/truevoice-intent`](https://github.com/PyThaiNLP/truevoice-intent).
To run the scipt, you have download `mari-intent.zip` and unzip the data first.

``` bash
wget https://github.com/PyThaiNLP/truevoice-intent/raw/master/mari-intent.zip
unzip mari-intent.zip mari-intent
```

Then run the script below.

``` bash
python true_intent_classification.py
```

Below is the classification result on 15 percent leave out on training set

| Method                         | Accuracy | Precision | Recall |   F1   |
| ------------------------------ | -------- | --------- | ------ | ------ |
| Unigram + Logistic Reg         | 79.13    | 84.85     | 82.84  | 83.84  |
| Unigram + Linear SVC           | 78.57    | 84.31     | 84.13  | 83.22  |
| Unigram + TFIDF + Logistic Reg | 76.97    | 85.09     | 79.39  | 82.14  |
| Unigram + TFIDF + Linear SVC   | 80.27    | 86.39     | 83.41  | 84.88  |
| Bigram + Logistic Reg          | 82.84    | 88.35     | 84.75  | 86.51  |
| Bigram + Linear SVC            | 79.95    | 86.04     | 84.13  | 85.07  |
| Bigram + TDIDF + Logistic Reg  | 76.30    | 86.15     | 76.92  | 81.27  |
| Bigram + TFIDF + Linear SVC    | **82.95** | **88.85** | **84.54** | **86.64**  |
