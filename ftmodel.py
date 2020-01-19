import sys
import os
import fasttext as ft
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_entity_context(text, word, window):
        after = [""]
        sentence_split = text.split(word)
        before = word_tokenize(sentence_split[0])[-window:]
        if len(sentence_split) > 1:
            after = word_tokenize(sentence_split[1])[:window]
        return " ".join(before+[word]+after)


def save_text_data(data, path):
    with open(path, "w") as output_file:
        for line in data:
            output_file.write("".join(line)+"\n")


def get_scores(y_true, y_pred):
        scoring_map = {'accuracy': accuracy_score, "f1": f1_score,
                    "precision": precision_score, "recall": recall_score}
        score_dict = {k:v(y_true, y_pred) for k,v in scoring_map.items()}
        return score_dict


class FTModel():
    def __init__(self, dim=None, model_path=None):
        if model_path is not None:
            self.model = ft.load_model(model_path)
            self.dim = self.model.get_dimension()
        else:
            self.model = ft
            self.dim = dim

    
    def pre_process(self, df, window=5):
        df["Context"] = df.apply(lambda r: get_entity_context(r["Sentence"], r["Entity"], window), axis=1)
        sentence_list = [line.lower().replace('.', '') for line in df["Context"]]

        df['ft_Sentiment'] = df['Sentiment'].apply(lambda x: '\t'+'__label__'+x)
        sentiment_list = df['ft_Sentiment'].tolist()
        
        text_list = list(zip(sentence_list, sentiment_list))
        return text_list


    def train(self, data, params, window):
        self.params = params
        self.window = window
        processed_data = self.pre_process(data, window)

        output_file_path = "assets/fasttext_input.txt"
        save_text_data(processed_data, output_file_path)

        embeddings = "assets/wiki-news-300d-1M.vec"
        model = self.model.train_supervised(input=output_file_path, **self.params, dim=self.dim, 
                                                pretrainedVectors=embeddings)
        return model


    def kfold_scores(self, nfolds, training_data):
        kf = KFold(n_splits=nfolds, random_state=42)
        scores = {'accuracy': [], "f1": [], "precision": [], "recall": []}
        for train_index, test_index in kf.split(training_data):
            df_train = training_data.iloc[train_index]
            df_test = training_data.iloc[test_index]
            
            X_test = df_test["Context"]
            y_test = df_test["Sentiment"].map({'negative':0, 'positive':1})
            
            model = self.train(df_train, self.params, self.window)
            y_pred = X_test.apply(lambda x: model.predict([x],k=1)[0][0][0]).map({'__label__negative':0,
                                                                                    '__label__positive':1})
            new_scores = get_scores(y_test, y_pred)
            scores = {k: v+[new_scores[k]] for k, v in scores.items()}
        scores = {k:np.mean(v) for k, v in scores.items()}
        return scores


    def predict(self, df, window):
        df["Context"] = df.apply(lambda r: get_entity_context(r["Sentence"], r["Entity"], window), axis=1)
        df["Context"] = df["Context"].apply(lambda x: x.lower().replace('.', ''))
        y_pred = df["Context"].apply(lambda x: self.model.predict([x],k=1)[0][0][0])
        y_pred = y_pred.map({'__label__negative':'negative', '__label__positive':'positive'})
        return y_pred