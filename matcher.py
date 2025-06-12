# matcher.py

import numpy as np

class Matcher:
    def __init__(self, db_feats, db_labels, metric='cosine'):
        self.db = db_feats      # (N, D) numpy array
        self.labels = db_labels
        self.metric = metric

    def _distance(self, q):
        if self.metric == 'euclidean':
            # 브로드캐스트로 빠른 계산
            return np.linalg.norm(self.db - q, axis=1)
        elif self.metric == 'cosine':
            qn = q / (np.linalg.norm(q) + 1e-6)
            dbn = self.db / (np.linalg.norm(self.db, axis=1, keepdims=True) + 1e-6)
            return 1 - dbn.dot(qn)
        else:
            raise ValueError(f'Unknown metric: {self.metric}')

    def classify(self, q_feat):
        """Top-1 분류"""
        dists = self._distance(q_feat)
        idx = np.argmin(dists)
        return self.labels[idx]

    def retrieve(self, q_feat, k=10):
        """Top-k 리트리벌: 가장 가까운 k개 레이블 리스트 반환"""
        dists = self._distance(q_feat)
        idxs = np.argsort(dists)[:k]
        return [self.labels[i] for i in idxs]

    def match(self, q_feat, k=10):
        """(idxs, distances) 반환"""
        dists = self._distance(q_feat)
        idxs = np.argsort(dists)[:k]
        return idxs, dists[idxs]
