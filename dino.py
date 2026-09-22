import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from skimage.feature import hog, local_binary_pattern
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import joblib

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
BASE_DIR = r"C:\Users\omgaw\Downloads\Pareidolia 2.0\lunar dataset final"
CACHE_DIR = os.path.join(BASE_DIR, "cache_dinov2") # Separate cache for DINOv2 features
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(BASE_DIR, "train_metadata_split.csv")
VAL_CSV = os.path.join(BASE_DIR, "val_metadata_split.csv")

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val")

# Save paths for final model package
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "final_model_weights.pkl")
THRESHOLD_SAVE_PATH = os.path.join(BASE_DIR, "optimal_threshold.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# LOAD DINOv2 BACKBONE
# ==========================================
print("Loading DINOv2-S/14 backbone from PyTorch Hub...")
# Loading Meta's DINOv2 small model (embedding dimension: 384)
dinov2_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_backbone = dinov2_backbone.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# FRANKOT-CHELLAPPA & HESSIAN CURVATURE
# ==========================================
def frankot_chellappa_integration(grad_x, grad_y):
    H, W = grad_x.shape
    wx = np.fft.fftfreq(W) * 2 * np.pi
    wy = np.fft.fftfreq(H) * 2 * np.pi
    WX, WY = np.meshgrid(wx, wy)
    
    denom = WX**2 + WY**2
    denom[0, 0] = 1.0  
    
    F_gx = np.fft.fft2(grad_x)
    F_gy = np.fft.fft2(grad_y)
    
    F_z = -1j * (WX * F_gx + WY * F_gy) / denom
    F_z[0, 0] = 0.0
    
    return np.real(np.fft.ifft2(F_z))

def extract_hessian_curvature_features(height_map):
    dz_dx, dz_dy = np.gradient(height_map)
    dz_dxx, dz_dxy = np.gradient(dz_dx)
    _, dz_dyy = np.gradient(dz_dy)
    
    trace = dz_dxx + dz_dyy
    det = (dz_dxx * dz_dyy) - (dz_dxy ** 2)
    
    discriminant = np.maximum(0.0, (trace ** 2) - (4 * det))
    sqrt_disc = np.sqrt(discriminant)
    
    lambda1 = (trace + sqrt_disc) / 2.0
    lambda2 = (trace - sqrt_disc) / 2.0
    total_pixels = float(height_map.size)
    
    return np.array([
        np.mean(lambda1), np.std(lambda1),
        np.mean(lambda2), np.std(lambda2),
        np.sum((lambda1 > 0) & (lambda2 > 0)) / total_pixels,  # Bowls
        np.sum((lambda1 < 0) & (lambda2 < 0)) / total_pixels,  # Domes
        np.sum(lambda1 * lambda2 < 0) / total_pixels,          # Saddles
        np.mean(det), np.mean(trace)
    ])

# ==========================================
# RAW FEATURE EXTRACTION + DINOv2 EMBEDDINGS
# ==========================================
def extract_raw_features(img_path, sun_azimuth=0.0):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None, None
    img_gray = cv2.resize(img_gray, (128, 128))
    
    # Hand-crafted + Curvature
    hog_features = hog(img_gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    lbp = local_binary_pattern(img_gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    
    mean_val, std_val = np.mean(img_gray), np.std(img_gray)
    skew_val, kurt_val = skew(img_gray.ravel()), kurtosis(img_gray.ravel())
    
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    az_rad = np.radians(sun_azimuth)
    directional_grad = grad_x * np.cos(az_rad) + grad_y * np.sin(az_rad)
    
    sfs_fc_features = [
        np.mean(grad_x), np.std(grad_x), np.mean(grad_y), np.std(grad_y),
        np.mean(directional_grad), np.std(directional_grad),
        pd.Series(directional_grad.ravel()).skew(), np.mean(np.abs(grad_x) + np.abs(grad_y))
    ]
    
    _, t_low = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    _, t_mid = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
    _, t_high = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    topology_features = [
        np.mean(gradient_magnitude), np.std(gradient_magnitude),
        cv2.connectedComponents(t_low)[0], cv2.connectedComponents(t_mid)[0], cv2.connectedComponents(t_high)[0]
    ]
    
    height_map = frankot_chellappa_integration(grad_x, grad_y)
    curvature_features = extract_hessian_curvature_features(height_map)
    
    handcrafted = np.hstack([
        hog_features, lbp_hist, [mean_val, std_val, skew_val, kurt_val],
        sfs_fc_features, topology_features, curvature_features
    ])

    # DINOv2 Embeddings (384-d output vector)
    try:
        pil_img = Image.open(img_path).convert('RGB')
        tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            dinov2_emb = dinov2_backbone(tensor).cpu().numpy().flatten()
    except Exception:
        dinov2_emb = np.zeros(384)

    return handcrafted, dinov2_emb

def load_or_extract_dataset(csv_path, img_dir, dataset_type="train"):
    cache_path = os.path.join(CACHE_DIR, f"{dataset_type}_cache.npz")
    
    if os.path.exists(cache_path):
        print(f"Loading cached DINOv2 features for {dataset_type} from {cache_path}...")
        data = np.load(cache_path)
        return data['X'], data['y']
        
    df = pd.read_csv(csv_path)
    id_col = next((col for col in ['image_id', 'id', 'filename'] if col in df.columns), df.columns[0])
    target_col = next((col for col in ['label', 'target', 'class'] if col in df.columns), df.columns[-1])
    azimuth_col = next((col for col in ['sun_azimuth', 'azimuth', 'sun_azi'] if col in df.columns), None)
    
    print(f"Extracting features & DINOv2 embeddings for {img_dir}...")
    X_hc, X_dino, y = [], [], []
    
    for _, row in df.iterrows():
        img_id = str(row[id_col])
        label = row[target_col]
        azimuth = float(row[azimuth_col]) if azimuth_col and not pd.isna(row[azimuth_col]) else 0.0
        img_path = os.path.join(img_dir, os.path.basename(img_id))
        
        hc, dino = extract_raw_features(img_path, sun_azimuth=azimuth)
        if hc is not None:
            X_hc.append(hc)
            X_dino.append(dino)
            y.append(label)
            
    X_hc = np.array(X_hc)
    X_dino = np.array(X_dino)
    y = np.array(y)
    
    print(f"Fitting PCA on DINOv2 embeddings ({dataset_type})...")
    pca_dino = PCA(n_components=192, random_state=42)
    dino_pca = pca_dino.fit_transform(X_dino)
    
    if dataset_type == "train":
        np.save(os.path.join(CACHE_DIR, "pca_dino_components.npy"), pca_dino.components_)
        np.save(os.path.join(CACHE_DIR, "pca_dino_mean.npy"), pca_dino.mean_)

    X_combined = np.hstack([X_hc, dino_pca])
    np.savez_compressed(cache_path, X=X_combined, y=y)
    return X_combined, y

def load_val_dataset(csv_path, img_dir):
    cache_path = os.path.join(CACHE_DIR, "val_cache.npz")
    if os.path.exists(cache_path):
        print("Loading cached validation DINOv2 features...")
        data = np.load(cache_path)
        return data['X'], data['y']
        
    df = pd.read_csv(csv_path)
    id_col = next((col for col in ['image_id', 'id', 'filename'] if col in df.columns), df.columns[0])
    target_col = next((col for col in ['label', 'target', 'class'] if col in df.columns), df.columns[-1])
    azimuth_col = next((col for col in ['sun_azimuth', 'azimuth', 'sun_azi'] if col in df.columns), None)
    
    X_hc, X_dino, y = [], [], []
    for _, row in df.iterrows():
        img_id = str(row[id_col])
        label = row[target_col]
        azimuth = float(row[azimuth_col]) if azimuth_col and not pd.isna(row[azimuth_col]) else 0.0
        img_path = os.path.join(img_dir, os.path.basename(img_id))
        hc, dino = extract_raw_features(img_path, sun_azimuth=azimuth)
        if hc is not None:
            X_hc.append(hc)
            X_dino.append(dino)
            y.append(label)
            
    X_hc, X_dino, y = np.array(X_hc), np.array(X_dino), np.array(y)
    
    dino_comp = np.load(os.path.join(CACHE_DIR, "pca_dino_components.npy"))
    dino_mean = np.load(os.path.join(CACHE_DIR, "pca_dino_mean.npy"))
    
    dino_pca = (X_dino - dino_mean) @ dino_comp.T
    
    X_combined = np.hstack([X_hc, dino_pca])
    np.savez_compressed(cache_path, X=X_combined, y=y)
    return X_combined, y

# ==========================================
# EXECUTION
# ==========================================
X_train, y_train = load_or_extract_dataset(TRAIN_CSV, TRAIN_IMG_DIR, dataset_type="train")
X_val, y_val = load_val_dataset(VAL_CSV, VAL_IMG_DIR)

print(f"\nFinal Feature Shape with DINOv2 (Train): {X_train.shape}")
print(f"Final Feature Shape with DINOv2 (Val): {X_val.shape}")

neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

# ==========================================
# MODEL STACKING
# ==========================================
print("\nBuilding Stacking Ensemble...")
estimators = [
    ('xgb', XGBClassifier(
        n_estimators=400, learning_rate=0.02, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8, 
        scale_pos_weight=scale_weight, random_state=42, n_jobs=-1
    )),
    ('hgb', HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.02, max_depth=6, 
        random_state=42
    )),
    ('rf', RandomForestClassifier(
        n_estimators=400, max_depth=20, 
        class_weight='balanced_subsample', random_state=42, n_jobs=-1
    ))
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
    cv=3,
    n_jobs=-1
)

ensemble_pipeline = make_pipeline(
    SimpleImputer(strategy='constant', fill_value=0.0),
    StandardScaler(),
    stacking_model
)

print("Training Ensemble Pipeline with DINOv2 Features...")
ensemble_pipeline.fit(X_train, y_train)

# ==========================================
# EVALUATION & THRESHOLD SWEEP
# ==========================================
print("\nEvaluating Model & Sweeping Decision Thresholds...")
val_probs = ensemble_pipeline.predict_proba(X_val)[:, 1]

best_thresh = 0.5
best_score = 0.0

for thresh in np.linspace(0.2, 0.8, 61):
    preds = (val_probs >= thresh).astype(int)
    score = balanced_accuracy_score(y_val, preds)
    if score > best_score:
        best_score = score
        best_thresh = thresh

print(f"\nOptimal Decision Threshold: {best_thresh:.2f}")
print(f"Tuned Validation Balanced Accuracy (DINOv2): {best_score:.4f}")

# Final evaluation report using the optimal threshold
y_pred_tuned = (val_probs >= best_thresh).astype(int)
print("\nDetailed Classification Report (DINOv2 + Tuned Threshold):")
print(classification_report(y_val, y_pred_tuned))

# ==========================================
# SAVE FINAL MODEL WEIGHTS & THRESHOLD
# ==========================================
print("\nSaving final model weights and configuration...")
joblib.dump(ensemble_pipeline, MODEL_SAVE_PATH)
joblib.dump(best_thresh, THRESHOLD_SAVE_PATH)

print(f"Successfully saved final model weights to: {MODEL_SAVE_PATH}")
print(f"Successfully saved optimal threshold ({best_thresh:.2f}) to: {THRESHOLD_SAVE_PATH}")