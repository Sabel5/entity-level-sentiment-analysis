# entity-level-sentiment-analysis

## Requirements:
- python 3.6
- fasttext
- nltk
- sklearn

## How to run
If you want to train the classifier:

```bash
python train.py path-to-training-set
```

If you want to predict the sentiment for a given test set:

```bash
python test.py -i path-to-test-set -m path-to-model
```