import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import signal as sg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import csv


# 한 장의 예제 이미지 선택 (경로 수정)
sample_path = './recaptcha-dataset/Large/Bicycle/Bicycle (3).png'
img = cv2.imread(sample_path)
if img is None:
    raise FileNotFoundError(f"경로를 확인하세요: {sample_path}")

# --- 설정 ---------------------------------------------------
dataset_dir = './recaptcha-dataset/Large'
labels = ['Bicycle','Bridge','Bus','Car','Chimney',
          'Crosswalk','Hydrant','Motorcycle','Palm','Traffic Light']
# PCA 축소 차원
pca_dims = 30
# KNN 이웃 개수
k_neighbors = 4
cv_folds    = 5

# --- helper functions -------------------------------------

def preprocess_extended(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq   = cv2.equalizeHist(gray)
    # 1) Bilateral Filter (엣지 보존 스무딩)
    #    d: 이웃 픽셀 탐색 지름, sigmaColor: 강도 유사도, sigmaSpace: 공간 거리 유사도
    bilat = cv2.bilateralFilter(eq, d=9, sigmaColor=75, sigmaSpace=75)
    # 2) Canny 엣지 맵
    edges = cv2.Canny(bilat, 100, 200)
    # 3) 샤프닝 (가우시안 블러를 이용)
    gauss = cv2.GaussianBlur(bilat, (3,3), 1)
    sharp = cv2.addWeighted(bilat, 2, gauss, -1, 0)
    return bilat, edges, sharp


# 엣지 히스토그램 (32‐bin 예시)
def extract_edge_hist(edge):
    hist, _ = np.histogram(edge.ravel(), bins=32, range=(0,256))
    return hist.astype(float) / hist.sum()

# 샤프 영상에 LBP 적용
def extract_lbp_sharp(sharp):
    lbp = local_binary_pattern(sharp, P=8, R=1)
    hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0,256))
    return hist.astype(float) / hist.sum()


def extract_lbp(gray):
    lbp = local_binary_pattern(gray, P=8, R=1)
    hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0, 256))
    hist = hist.astype(float) / hist.sum()
    return hist

def extract_glcm_props(gray):
    # 거리=1, 각도 0°, 90° 두 방향
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/2],
                        levels=256, symmetric=False, normed=True)
    feats = []
    for prop in ['contrast','dissimilarity','homogeneity','energy','correlation']:
        arr = graycoprops(glcm, prop)[0]    # shape (n_dist=1, n_angle=2) → take [0]
        feats.append(arr[0] + arr[1])        # 0° + 90°
    return np.array(feats)

def extract_laws(gray):
    # smooth
    smooth = (1/25)*np.ones((5,5))
    blur = sg.convolve(gray, smooth, mode='same')
    proc = np.abs(gray - blur)
    # 4×5-vector
    v = np.array([[1,4,6,4,1],[-1,-2,0,2,1],[-1,0,2,0,1],[1,-4,6,-4,1]])
    # 16 filters L5*L5 ... R5*R5
    filters = [np.outer(v[i], v[j]) for i in range(4) for j in range(4)]
    convs = np.stack([sg.convolve(proc, f, mode='same') for f in filters], axis=-1)
    # 9 texture energy maps
    combos = [
      (1,4),(2,8),(3,12),(7,13),(6,9),(11,14),
      (10,10),(5,5),(15,15)
    ]
    energies = []
    denom = np.abs(convs[...,0]).sum()
    for (i,j) in combos:
        if i==j:
            energies.append(np.abs(convs[...,i]).sum() / denom)
        else:
            energies.append((np.abs(convs[...,i])+np.abs(convs[...,j])).sum() / denom)
    return np.array(energies)  # shape (9,)


# --- 특징 추출 함수 --------------------------------

def extract_features(path):
    bilat, edges, sharp = preprocess_extended(path)

    # 기존 피처
    f_lbp    = extract_lbp(bilat)                    # 64
    # f_glcm   = extract_glcm_props(bilat)            # 5
    f_laws   = extract_laws(bilat)                  # 9

    # 추가 피처
    f_edge   = extract_edge_hist(edges)           # 32
    f_lbp_sh = extract_lbp_sharp(sharp)           # 64

    # 최종 벡터: 78 + 32 + 64 = 174차원
    return np.hstack([f_lbp, f_laws, f_edge, f_lbp_sh])


# --- 데이터 로드 & 특징 추출 --------------------------------
train_feats, train_labels = [], []
test_feats,  test_labels  = [], []

for lab in labels:
    folder = os.path.join(dataset_dir, lab)
    imgs = sorted(os.listdir(folder))
    for i, name in enumerate(imgs):
        path = os.path.join(folder, name)
        feat = extract_features(path)
        if i < 100:
            train_feats.append(feat); train_labels.append(lab)
        elif i < 150:
            test_feats.append(feat);  test_labels.append(lab)
        else:
            break

X_train = np.array(train_feats)
y_train = np.array(train_labels)
X_test  = np.array(test_feats)
y_test  = np.array(test_labels)

# --- 1) Standard Scaler 적용 --------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- 2) PCA 차원 축소 ------------------------------------------
pca = PCA(n_components=pca_dims, random_state=42)
X_train_p = pca.fit_transform(X_train_s)
X_test_p  = pca.transform(X_test_s)

# --- KNN 학습 & 예측 ---------------------------------------
knn = KNeighborsClassifier(n_neighbors=k_neighbors)

# 교차검증
cv_scores = cross_val_score(knn, X_train_p, y_train,
                            cv=cv_folds, scoring='accuracy', n_jobs=-1)
print("------------------ 교차검증 ----------------------")
print(f"{cv_folds}-fold CV accuracies (1-NN): {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("------------------------------------------------")


knn.fit(X_train_p, train_labels)

# Task1: Classification (Top-1)
pred1 = knn.predict(X_test_p)
print(classification_report(test_labels, pred1))

with open('c1_t1_a1.csv','w', newline='') as f:
    w = csv.writer(f)
    for idx, lab in enumerate(pred1, 1):
        w.writerow([f'query{idx:03}.png', lab])

# Task2: Retrieval (Top-10)
inds = knn.kneighbors(X_test_p, n_neighbors=10, return_distance=False)
# neigh_labels: (100,10)
neigh_labels = np.array(train_labels)[inds]

with open('c1_t2_a1.csv','w', newline='') as f:
    w = csv.writer(f)
    for idx, neigh in enumerate(neigh_labels, 1):
        w.writerow([f'query{idx:03}.png'] + neigh.tolist())
