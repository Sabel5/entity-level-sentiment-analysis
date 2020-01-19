import sys
import os
from ftmodel import FTModel
import pandas as pd


def train(training_set_path):
    dim = 300
    params = {"lr": 0.9, "epoch": 20, "wordNgrams": 3} 
    nfolds = 3
    window = 8

    model = FTModel(dim)
    training_data = pd.read_excel(training_set_path)
    trained_model = model.train(training_data, params, window)
    trained_model.save_model("assets/model.bin")
    scores = model.kfold_scores(nfolds, training_data)
    print(scores)
    

if __name__ == '__main__':
    training_set_path = sys.argv[1]
    train(training_set_path)