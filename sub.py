import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from skimage.feature import hog, local_binary_pattern
from scipy.stats import skew, kurtosis
import joblib

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
BASE_DIR = r"C:\Users\omgaw\Downloads\Pareidolia 2.0"
EVAL_IMG_DIR = os.path.join(BASE_DIR, "eval_images")
CACHE_DIR = os.path.join(BASE_DIR, "cache_dinov2")

# Automatically detect test metadata format
test_files = [f for f in os.listdir(BASE_DIR) if f.startswith("test_metadata")]
if not test_files:
    raise FileNotFoundError("Could not find test_metadata file in the base directory.")

test_file_path = os.path.join(BASE_DIR, test_files[0])
print(f"Loading test metadata from: {test_files[0]}")

if test_file_path.endswith('.csv') or '.' not in test_file_path:
    try:
        df_test = pd.read_csv(test_file_path)
    except Exception:
        df_test = pd.read_excel(test_file_path)
else:
    df_test = pd.read_excel(test_file_path)

MODEL_PATH = os.path.join(BASE_DIR, "final_model_weights.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "optimal_threshold.pkl")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# LOAD DINOv2 BACKBONE & SAVED ARTIFACTS
# ==========================================
print("Loading DINOv2 backbone and saved model artifacts...")
dinov2_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_backbone = dinov2_backbone.to(device).eval()

model = joblib.load(MODEL_PATH)
best_thresh = joblib.load(THRESHOLD_PATH)

pca_components = np.load(os.path.join(CACHE_DIR, "pca_dino_components.npy"))
pca_mean = np.load(os.path.join(CACHE_DIR, "pca_dino_mean.npy"))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# BICUBIC ROTATION HELPER
# ==========================================
def rotate_image_bicubic(image, angle):
    """Rotates an image by a given angle using high-quality bicubic interpolation."""
    if angle == 0.0 or pd.isna(angle):
        return image
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    # OpenCV rotation matrix; cv2.INTER_CUBIC ensures bicubic interpolation
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return rotated

# ==========================================
# FEATURE EXTRACTION FUNCTIONS
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
        np.sum((lambda1 > 0) & (lambda2 > 0)) / total_pixels,
        np.sum((lambda1 < 0) & (lambda2 < 0)) / total_pixels,
        np.sum(lambda1 * lambda2 < 0) / total_pixels,
        np.mean(det), np.mean(trace)
    ])

def extract_raw_features(img_path, sun_azimuth=0.0):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None, None
    img_gray = cv2.resize(img_gray, (128, 128))
    
    # Apply bicubic rotation matching the sun azimuth angle
    img_gray = rotate_image_bicubic(img_gray, sun_azimuth)
    
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

    # For DINOv2 embedding extraction, load RGB, rotate using bicubic, then transform
    try:
        pil_color = cv2.imread(img_path)
        pil_color = cv2.resize(pil_color, (224, 224))
        pil_color = rotate_image_bicubic(pil_color, sun_azimuth)
        pil_rgb = cv2.cvtColor(pil_color, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pil_rgb)
        
        tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            dinov2_emb = dinov2_backbone(tensor).cpu().numpy().flatten()
    except Exception:
        dinov2_emb = np.zeros(384)

    return handcrafted, dinov2_emb

# ==========================================
# RUN INFERENCE ON TEST SET
# ==========================================
id_col = next((col for col in ['image_id', 'id', 'filename'] if col in df_test.columns), df_test.columns[0])
azimuth_col = next((col for col in ['sun_azimuth', 'azimuth', 'sun_azi'] if col in df_test.columns), None)

print(f"Extracting features and applying bicubic azimuth rotation for test images from {EVAL_IMG_DIR}...")
X_hc_list, X_dino_list, image_ids = [], [], []

for _, row in df_test.iterrows():
    img_id = str(row[id_col])
    azimuth = float(row[azimuth_col]) if azimuth_col and not pd.isna(row[azimuth_col]) else 0.0
    img_path = os.path.join(EVAL_IMG_DIR, os.path.basename(img_id))
    
    hc, dino = extract_raw_features(img_path, sun_azimuth=azimuth)
    if hc is not None:
        X_hc_list.append(hc)
        X_dino_list.append(dino)
        image_ids.append(img_id)

X_hc = np.array(X_hc_list)
X_dino = np.array(X_dino_list)

# Project DINOv2 embeddings using saved PCA components
dino_pca = (X_dino - pca_mean) @ pca_components.T
X_test = np.hstack([X_hc, dino_pca])

print(f"Running predictions using threshold {best_thresh:.2f}...")
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_thresh).astype(int)

# Create submission file
sub_df = pd.DataFrame({
    id_col: image_ids,
    'label': test_preds
})

sub_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission successfully created and saved to: {SUBMISSION_PATH}")