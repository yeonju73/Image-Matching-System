import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import signal as sg
from sklearn.cluster import MiniBatchKMeans

# --- 설정 ---------------------------------------------------
dataset_dir = './recaptcha-dataset/Large'
labels = ['Bicycle','Bridge','Bus','Car','Chimney',
          'Crosswalk','Hydrant','Motorcycle','Palm','Traffic Light']
# PCA 축소 차원
pca_dims = 60
# KNN 이웃 개수
k_neighbors = 4
cv_folds    = 5

# BoW 설정
n_clusters = 100   # SIFT BoW 코드북 크기
random_state = 42

# SIFT 생성기
sift = cv2.SIFT_create()

# 1) 모든 훈련 디스크립터 수집
all_descriptors = []
image_paths = []
for lab in labels:
    for fname in sorted(os.listdir(os.path.join(dataset_dir, lab))):
        path = os.path.join(dataset_dir, lab, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            all_descriptors.append(des)
        image_paths.append((path, lab))
all_descriptors = np.vstack(all_descriptors)

# 2) KMeans로 코드북 학습
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=10000)
kmeans.fit(all_descriptors)

def bow_histogram(descriptors, kmeans):
    if descriptors is None:
        return np.zeros(n_clusters, dtype=float)
    labels = kmeans.predict(descriptors)
    hist, _ = np.histogram(labels, bins=np.arange(n_clusters+1))
    return hist.astype(float) / hist.sum()  # 정규화

# --- helper functions -------------------------------------

def preprocess_extended(path):
    img = cv2.imread(path)
    
    # 1) 컬러 노이즈 제거
    img = cv2.fastNlMeansDenoisingColored(img, None,
                                          h=10, hColor=10,
                                          templateWindowSize=7,
                                          searchWindowSize=21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq   = cv2.equalizeHist(gray)
    
    # 3) Canny 엣지 맵
    edges = cv2.Canny(eq, 100, 200)
    # 4) 샤프닝 (가우시안 블러를 이용)
    gauss = cv2.GaussianBlur(eq, (3,3), 1)
    sharp = cv2.addWeighted(eq, 2, gauss, -1, 0)
    return img, eq, edges, sharp


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
    return lbp, hist

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

def extract_color_hist(path, bins=32):
    """
    BGR 이미지를 HSV로 변환한 뒤,
    H, S, V 채널 각각에 대해 bins‐구간 히스토그램을 계산하고 정규화.
    총 차원 = bins * 3
    """
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # H: [0,179], S/V: [0,255]
    h_hist, _ = np.histogram(hsv[:,:,0], bins=bins, range=(0,180))
    s_hist, _ = np.histogram(hsv[:,:,1], bins=bins, range=(0,256))
    v_hist, _ = np.histogram(hsv[:,:,2], bins=bins, range=(0,256))
    hist = np.hstack([h_hist, s_hist, v_hist]).astype(float)
    return hist / hist.sum()

# --- Visualization -----------------------------
# Sample image
sample_path = './recaptcha-dataset/Large/Bus/Bus (3).png'
orig, eq, edges, sharp = preprocess_extended(sample_path)
lbp_img, lbp_hist = extract_lbp(eq)
edge_hist = extract_edge_hist(edges)
color_hist = extract_color_hist(sample_path, bins=32)

# Visualization
plt.figure(figsize=(14, 8))

# Original & processed
titles = ['Original','Denoised','eq','Edges','Sharp']
imgs = [orig, cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), eq, edges, sharp]
for i,(im,t) in enumerate(zip(imgs,titles),1):
    ax=plt.subplot(3,5,i); ax.set_title(t); ax.axis('off')
    cmap=None if im.ndim==3 else 'gray'; ax.imshow(im, cmap=cmap)

# LBP & its histogram
ax=plt.subplot(3,5,6); ax.set_title('LBP'); ax.axis('off'); ax.imshow(lbp_img, cmap='gray')
ax=plt.subplot(3,5,7); ax.set_title('LBP Hist'); ax.bar(range(len(lbp_hist)), lbp_hist, width=1)

# Edge histogram
ax=plt.subplot(3,5,8); ax.set_title('Edge Hist'); ax.bar(range(len(edge_hist)), edge_hist, width=1)

# Color histogram
ax=plt.subplot(3,5,9); ax.set_title('HSV Color Hist'); ax.bar(range(len(color_hist)), color_hist, width=1)

plt.tight_layout()
plt.show()