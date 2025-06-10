import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import signal as sg
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
import csv

# --- 설정 ---------------------------------------------------
dataset_dir = './recaptcha-dataset/Large'
labels = ['Bicycle','Bridge','Bus','Car','Chimney',
          'Crosswalk','Hydrant','Motorcycle','Palm','Traffic Light']
cv_folds    = 5
random_state = 42

# BoW 설정
n_clusters = 100  # SIFT BoW 코드북 크기
sift = cv2.SIFT_create()

# 1) SIFT-BoW 코드북 생성
des_list = []
for cls in labels:
    folder = os.path.join(dataset_dir, cls)
    for img_name in sorted(os.listdir(folder)):
        gray_img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
        _, des = sift.detectAndCompute(gray_img, None)
        if des is not None:
            des_list.append(des)
all_descriptors = np.vstack(des_list)
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=10000)
kmeans.fit(all_descriptors)

# BoW histogram
def bow_histogram(descriptors):
    if descriptors is None:
        return np.zeros(n_clusters)
    lbls = kmeans.predict(descriptors)
    hist, _ = np.histogram(lbls, bins=np.arange(n_clusters+1))
    return hist.astype(float)/hist.sum()

# 전처리 및 피처 함수
def preprocess(path):
    img = cv2.imread(path)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def extract_lbp(gray):
    lbp = local_binary_pattern(gray, P=8, R=1)
    hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0,256))
    return hist.astype(float)/hist.sum()

def extract_grad_orient_hist(gray, bins=8):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist, _ = np.histogram(ang.ravel(), bins=bins, range=(0,180), weights=mag.ravel())
    return hist/(hist.sum()+1e-6)

def extract_laws(gray):
    smooth = np.ones((5,5))/25
    blur = sg.convolve(gray, smooth, mode='same')
    proc = np.abs(gray-blur)
    v = np.array([[1,4,6,4,1],[-1,-2,0,2,1],[-1,0,2,0,1],[1,-4,6,-4,1]])
    filters=[np.outer(v[i],v[j]) for i in range(4) for j in range(4)]
    convs=np.stack([sg.convolve(proc,f,mode='same') for f in filters],axis=-1)
    combos=[(1,4),(2,8),(3,12),(7,13),(6,9),(11,14),(10,10),(5,5),(15,15)]
    denom=np.abs(convs[...,0]).sum(); energies=[]
    for i,j in combos:
        if i==j: energies.append(np.abs(convs[...,i]).sum()/denom)
        else: energies.append((np.abs(convs[...,i])+np.abs(convs[...,j])).sum()/denom)
    return np.array(energies)

def extract_color_hist(path, bins=32):
    img=cv2.imread(path); hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    h_hist,_=np.histogram(h,bins=bins,range=(0,180))
    s_hist,_=np.histogram(s,bins=bins,range=(0,256))
    v_hist,_=np.histogram(v,bins=bins,range=(0,256))
    hist=np.hstack([h_hist,s_hist,v_hist]).astype(float)
    return hist/hist.sum()

def extract_sift(path):
    gray_img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    _,descr=sift.detectAndCompute(gray_img,None)
    return bow_histogram(descr)

# 모든 피처 결합 함수
def extract_features(path):
    gray=preprocess(path)
    return np.hstack([
        extract_lbp(gray),
        extract_grad_orient_hist(gray),
        extract_laws(gray),
        extract_color_hist(path),
        extract_sift(path)
    ])

# 데이터 로드
train_feats, train_labels, test_feats, test_labels = [],[],[],[]
for cls in labels:
    fld=os.path.join(dataset_dir,cls)
    for i,fn in enumerate(sorted(os.listdir(fld))):
        feat=extract_features(os.path.join(fld,fn))
        if i<80:
            train_feats.append(feat); train_labels.append(cls)
        elif i<100:
            test_feats.append(feat);  test_labels.append(cls)
        else: break
X_train,y_train=np.array(train_feats),np.array(train_labels)
X_test, y_test =np.array(test_feats), np.array(test_labels)

# Pipeline + GridSearchCV
tube=Pipeline([
    ('scaler', StandardScaler()),
    # ('skb',    SelectKBest(f_classif)),
    ('pca',    PCA(random_state=random_state)),
    ('knn',    KNeighborsClassifier())
])
param_grid={
    # 'skb__k':[100,200],
    'pca__n_components':[20,30,40,50,60],
    'knn__n_neighbors':[1,3,4,5,6,7],
    'knn__weights':['uniform','distance']
}
if __name__=='__main__':
    search=GridSearchCV(tube,param_grid,cv=cv_folds,scoring='accuracy',n_jobs=-1)
    search.fit(X_train,y_train)
    print('Best params:',search.best_params_)
    print('Best CV:',search.best_score_)

    best=search.best_estimator_
    pred=best.predict(X_test)
    print(classification_report(y_test,pred))

    with open('c1_t1_a1.csv','w',newline='') as f:
        w=csv.writer(f)
        for i,l in enumerate(pred,1): w.writerow([f'query{i:03}.png',l])

    # Retrieval
    X_test_proc=best.named_steps['pca'].transform(best.named_steps['scaler'].transform(X_test))
    inds=best.named_steps['knn'].kneighbors(X_test_proc,n_neighbors=10,return_distance=False)
    neighs=np.array(y_train)[inds]
    with open('c1_t2_a1.csv','w',newline='') as f:
        w=csv.writer(f)
        for i,row in enumerate(neighs,1): w.writerow([f'query{i:03}.png']+row.tolist())
