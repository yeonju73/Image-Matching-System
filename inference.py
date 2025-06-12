# inference.py

import os
import csv
import random
import numpy as np
from indexer import EarlyVisionIndexer
from matcher import Matcher

from sklearn.decomposition import PCA
import joblib
db_feats = np.load('db_feats_2.npy')
pca = PCA(n_components=20)        # 예: 50차원으로 줄임
pca.fit(db_feats)
joblib.dump(pca, 'pca_recaptcha.pkl')


# 1) 로드

pca = joblib.load('pca_recaptcha.pkl')

iv_idx = EarlyVisionIndexer('db_feats_2.npy','db_labels_2.pkl')
db_feats, db_labels = iv_idx.get_db()

# 3) DB 피처 차원 축소
db_feats_reduced = pca.transform(db_feats)         # (N, D_red)

matcher = Matcher(db_feats_reduced, db_labels, metric='cosine')

# 2) 테스트셋 경로
test_dir = './query'   # 당일 배포된 100장
out1 = open('c1_t1_a5.csv','w', newline='')
out2 = open('c1_t2_a5.csv','w', newline='')
w1 = csv.writer(out1)
w2 = csv.writer(out2)

# # 여기서부터는 로컬 테스트를 위함
# LABELS = ['Bicycle','Bridge','Bus','Car','Chimney',
#           'Crosswalk','Hydrant','Motorcycle','Palm','Traffic Light']

# total_samples     = 0
# correct_top1      = 0
# correct_in_top10  = 0

# for cls in LABELS:
#     folder = os.path.join(test_dir, cls)
#     all_imgs = os.listdir(folder)
#     # 시드 고정 없이 매번 다른 10장 샘플링
#     sampled = random.sample(all_imgs, 10)
#     for fn in sampled:
#         total_samples += 1
#         path = os.path.join(folder, fn)
#         q_feat = iv_idx.extract(path)

#         # Task1: Top-1 Classification
#         pred = matcher.classify(q_feat)
#         w1.writerow([f'{cls}/{fn}', pred])
#         if pred == cls:
#             correct_top1 += 1

#         # Task2: Top-10 Retrieval
#         top10 = matcher.retrieve(q_feat, k=10)
#         w2.writerow([f'{cls}/{fn}'] + top10)
#         correct_in_top10 += sum(1 for pred_cls in top10 if pred_cls == cls)


# 3) 순회
for fn in sorted(os.listdir(test_dir)):
    path = os.path.join(test_dir, fn)
    q_feat = iv_idx.extract(path)
    q_feat_reduced = pca.transform(q_feat.reshape(1, -1)).squeeze()  # (D_red,)

    # Task1
    pred = matcher.classify(q_feat_reduced)
    w1.writerow([fn, pred])

    # Task2
    top10 = matcher.retrieve(q_feat_reduced, k=10)
    w2.writerow([fn] + top10)


out1.close()
out2.close()
