# indexer.py

import numpy as np
import joblib
from image_matching_final2 import extract_features

class EarlyVisionIndexer:
    def __init__(self, feat_path='db_feats_2.npy', label_path='db_labels_2.pkl'):
        self.db_feats = np.load(feat_path)  # shape (N, D)
        self.db_labels = joblib.load(label_path)

    def extract(self, img_path):
        """Query 이미지에서 피처 벡터 추출"""
        return extract_features(img_path)

    def get_db(self):
        """DB 전체 피처와 레이블 반환"""
        return self.db_feats, self.db_labels
