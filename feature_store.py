# feature_store.py

import os
import numpy as np
import joblib
from image_matching_final1 import extract_features  # 위에서 짠 extract_features 함수
from sklearn.decomposition import PCA


def build_feature_store(dataset_dir, labels, out_feats='db_feats_1.npy', out_labels='db_labels_1.pkl'):
    feats = []
    labs  = []
    for cls in labels:
        folder = os.path.join(dataset_dir, cls)
        for fname in sorted(os.listdir(folder)):
            path = os.path.join(folder, fname)
            fv = extract_features(path)       # (D,) vector
            feats.append(fv)
            labs.append(cls)
    feats = np.vstack(feats)                # (N, D)
    np.save(out_feats, feats)
    joblib.dump(labs, out_labels)
    
if __name__ == '__main__':
    DATA_DIR = './recaptcha-dataset/Large'
    LABELS   = ['Bicycle','Bridge','Bus','Car','Chimney',
                'Crosswalk','Hydrant','Motorcycle','Palm','Traffic Light']
    build_feature_store(DATA_DIR, LABELS)
