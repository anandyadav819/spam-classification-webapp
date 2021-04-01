# Load our Spam Classification model and predict on user data.
from joblib import load
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os

MODEL_PICKLE_PATH = "spam_clf_vc.joblib"
TRANSFORMER_PICKLE_PATH  = "spam_clf_tfidf_transformer.joblib"
PUNCT_NORMALIZER_PATH = "spam_clf_punct_transformer.joblib"
CAP_NORMALIZER_PATH = "spam_clf_cap_transformer.joblib"
TEXTLEN_NORMALIZER_PATH = "spam_clf_textlen_transformer.joblib"

class Predict:
    def __init__(self, model_path, transformer_path, mm1_path, mm2_path, mm3_path):
        self.model_path = model_path
        self.transformer_path = transformer_path
        self.mm1_path = mm1_path
        self.mm2_path = mm2_path
        self.mm3_path = mm3_path
        self.label_dict = {1:"SPAM", 0:"HAM"}
        self._load_tfidf()
        self._load_clf()
        self._load_scalers()

        self.exceptionFlag = False
        pass

    def preprocess_text(self, text):
        if len(text)>400:
            text = text[-400:]
        return text
    
    def get_static_features(self, text):
        punct_count = self.count_punctuation(text)
        cap_count = self.count_capitals(text)
        text_len = len(text)
        if punct_count!=0:
            punct_count_norm = self.mm1.transform(np.asarray([punct_count]).reshape((-1, 1)))
        else:
            punct_count_norm = np.asarray([punct_count]).reshape((-1, 1))
        
        if cap_count!=0:
            cap_count_norm = self.mm2.transform(np.asarray([cap_count]).reshape((-1, 1)))
        else:
            cap_count_norm = np.asarray([cap_count]).reshape((-1, 1))

        # cap_count_norm = 1/cap_count
        text_len_norm = self.mm3.transform(np.asarray([text_len]).reshape((-1, 1)))

        return np.concatenate([punct_count_norm, cap_count_norm, text_len_norm], axis=1)
        #return np.asarray([punct_count_norm, cap_count_norm, text_len_norm]).reshape((1, 3))

    def _load_tfidf(self):
        try:
            self.tfidf = load(self.transformer_path)
        except Exception:
            "Print error loading TFIDF from pickle."
            self.exceptionFlag = True
            self.tfidf = TfidfVectorizer()
            

    def _load_clf(self):
        try:
            self.lr = load(self.model_path)
        except Exception:
            "Print error loading LR model from pickle."
            self.exceptionFlag = True
            self.lr = LogisticRegression()

    def _load_scalers(self):
        self.mm1 = load(self.mm1_path)
        self.mm2 = load(self.mm2_path)
        self.mm3 = load(self.mm3_path)

    def count_punctuation(self, x):
        x = re.sub(r" ", "", x)
        lst_punc = re.findall(r'[^A-Za-z0-9.,/]', x)
        return len(lst_punc)

    def count_capitals(self, x):
        x = re.sub(r" ", "", x)
        lst_caps = re.findall(r'^[A-Z][A-Z]+', x)
        #print(lst_caps)
        return len(lst_caps)

    def _aggregate_features(self, text):
        stat_feats = self.get_static_features(text)
        tfidf_feats = self.tfidf.transform([text])
        all_feats = np.concatenate([tfidf_feats.toarray(), stat_feats], axis=1)
        return all_feats
    
    def _get_model_pred(self, feats):
        preds = self.lr.predict(feats)
        return preds

    def run(self, text):
        pre_text = self.preprocess_text(text)
        features = self._aggregate_features(pre_text)
        lr_preds = self._get_model_pred(features)
        #print(lr_preds)
        return self.label_dict[lr_preds[0]]


Predicter = Predict(MODEL_PICKLE_PATH, TRANSFORMER_PICKLE_PATH, PUNCT_NORMALIZER_PATH, CAP_NORMALIZER_PATH, TEXTLEN_NORMALIZER_PATH)

# test_lbl = newObj.run("PRIVATE! Your 2003 Account Statement for 078")
# print(test_lbl)