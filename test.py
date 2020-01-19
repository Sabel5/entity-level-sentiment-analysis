import sys
import os
import pandas as pd
from ftmodel import FTModel

from argparse import ArgumentParser

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", help="Path to test dataset.", default="../data/Entity_sentiment_testV2.xlsx", type=str)
    parser.add_argument(
        "-m", help="Path to the model.", default="assets/model.txt", type=str)
    return parser


def test(test_set_path, model_path):
    df_test = pd.read_excel(test_set_path)
    model = FTModel(model_path=model_path)
    window = 8

    pred = model.predict(df_test, window)
    
    data_path = os.path.dirname(test_set_path)
    output_path = os.path.join(data_path, "predictions.csv")
    df_output = pd.concat([df_test[["Sentence", "Entity"]], pd.Series(pred, name='Pred')], axis=1)
    df_output.to_csv(output_path, index=False)


if __name__ == '__main__':
    args = build_argparser().parse_args()
    test(args.i, args.m)