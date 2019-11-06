# Examples

In this section, we collect some of the example how to use `deepcut` for classification problem. This aims to use as an **starter-kit** for Thai text classification, not state-of-the-art classification. Please see the original repository if you want to do a proper comparison.

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

Below is the prediction score on `action` column on 15 percent leave out training set

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

And prediction score on `destination` column on 15 percent leave out training set

| Method                         | Accuracy | Precision | Recall |   F1   |
| ------------------------------ | -------- | --------- | ------ | ------ |
| Unigram + Logistic Reg         | 77.33    | 88.33     | 79.96  | 83.94  |
| Unigram + Linear SVC           | 78.57    | 86.21     | 82.43  | 84.28  |
| Unigram + TFIDF + Logistic Reg | 71.35    | 91.00     | 72.38  | 80.63  |
| Unigram + TFIDF + Linear SVC   | 78.05    | 88.05     | 80.89  | 84.32  |
| Bigram + Logistic Reg          | 79.29    | 88.11     | 81.30  | 84.57  |
| Bigram + Linear SVC            | 76.56    | 82.78     | 82.23  | 82.50  |
| Bigram + TDIDF + Logistic Reg  | 65.48    | 93.53     | 65.53  | 77.07  |
| Bigram + TFIDF + Linear SVC    | 78.21    | 89.84     | 79.75  | 84.50  |

## License

Please refer to the license from the repository for each corpus before usage.