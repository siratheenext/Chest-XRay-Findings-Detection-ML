import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import random
import pandas as pd # 🌟 เพิ่ม pandas เพื่อโหลดไฟล์เฉลย

# --- ✅ CONFIG V7 ULTIMATE ---
MODELS_DIR = '../../models_v7'
TEST_DIR = '../../data/train_images_14_hybrid' 
CSV_PATH = '../../data/csv/train_labels_14_hybrid.csv' #  ระบุที่อยู่ไฟล์ข้อมูลจริง
OUTPUT_DIR = '../../outputs/predictions_v7'

IMG_SIZE = (384, 384) 
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# โมเดลหลักสำหรับทำ Heatmap
HEATMAP_MODEL_FILE = 'best_DenseNet121_v7.keras'

# น้ำหนักการโหวตแบบเท่ากัน 3 โมเดล
MODEL_WEIGHTS = {
    'best_DenseNet121_v7.keras': 1.0,
    'best_ResNet50V2_v7.keras': 1.0,
    'best_MobileNetV2_v7.keras': 1.0
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🌟 โหลดไฟล์เฉลย (Ground Truth)
print("⏳ Loading Ground Truth Data...")
df_labels = pd.read_csv(CSV_PATH)

# ==========================================
# 🧠 ฟังก์ชัน Grad-CAM
# ==========================================
def get_gradcam_heatmap(img_array, model, class_index):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer = layer
            break
            
    if last_conv_layer is None: return None

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_smart_heatmap(original_img, heatmap_raw):
    heatmap = cv2.resize(heatmap_raw, (original_img.shape[1], original_img.shape[0]))
    
    # Threshold ตัดสีฟ้าทิ้ง เอาเฉพาะจุดที่ AI โฟกัสจริงๆ (สีแดง/เหลือง)
    threshold = 0.3 
    heatmap[heatmap < threshold] = 0
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # ทำให้จุดที่ไม่โฟกัสโปร่งใส
    mask = heatmap_raw > threshold
    mask = cv2.resize(mask.astype(np.uint8), (original_img.shape[1], original_img.shape[0]))
    
    superimposed_img = original_img.copy()
    for c in range(3):
        superimposed_img[:, :, c] = np.where(mask == 1, 
                                           (original_img[:, :, c] * 0.5 + heatmap[:, :, c] * 0.5), 
                                           original_img[:, :, c])
    return superimposed_img

# ==========================================
# 🚀 PREDICT ภาพแบบสุ่ม
# ==========================================
all_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random_image_file = random.choice(all_images)
img_path = os.path.join(TEST_DIR, random_image_file)

# 🌟 ดึงเฉลยจากไฟล์ CSV ตามชื่อไฟล์รูปภาพ
row = df_labels[df_labels['Filename'] == random_image_file]
actual_labels = []
if not row.empty:
    for cls in CLASSES:
        if row.iloc[0][cls] == 1:
            actual_labels.append(cls)
if not actual_labels:
    actual_labels = ["Normal (No Finding)"]
actual_text = " + ".join(actual_labels)

print(f"\n🎯 Selected Image: {random_image_file}")
print(f"✅ Actual Labels from CSV: {actual_text}")

img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

original = cv2.imread(img_path)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
original = cv2.resize(original, IMG_SIZE)

ensemble_preds = np.zeros(len(CLASSES))
total_weight = 0

for m_name, weight in MODEL_WEIGHTS.items():
    model_path = os.path.join(MODELS_DIR, m_name)
    if os.path.exists(model_path):
        print(f"⏳ Loading & Predicting: {m_name} (Weight: {weight})")
        temp_model = tf.keras.models.load_model(model_path, compile=False)
        preds = temp_model.predict(img_array, verbose=0)[0]
        ensemble_preds += (preds * weight)
        total_weight += weight
        
        # วาด Heatmap จาก DenseNet
        if m_name == HEATMAP_MODEL_FILE:
            top_idx = np.argmax(preds)
            heatmap_raw = get_gradcam_heatmap(img_array, temp_model, top_idx)
            if heatmap_raw is not None:
                final_img = apply_smart_heatmap(original, heatmap_raw)
            else:
                final_img = original.copy()
            
        del temp_model
        tf.keras.backend.clear_session()

if total_weight > 0:
    ensemble_preds = ensemble_preds / total_weight

top_idx_ensemble = np.argmax(ensemble_preds)
predicted_label = CLASSES[top_idx_ensemble]
ai_confidence = ensemble_preds[top_idx_ensemble] * 100

# ==========================================
# 📊 PLOT RESULTS (โชว์ความจริง เทียบกับ AI)
# ==========================================
plt.figure(figsize=(18, 8))

# 🌟 รูปที่ 1: รูปต้นฉบับ พร้อมเฉลยของหมอ
plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title(f"Original X-ray\nActual (ความจริง): {actual_text}", color='darkblue', fontsize=14, fontweight='bold')
plt.axis('off')

# รูปที่ 2: รอยโรคที่ AI มองเห็น (Heatmap)
plt.subplot(1, 3, 2)
if 'final_img' in locals():
    plt.imshow(final_img)
    plt.title(f"AI Focus Area\nPredicted: {predicted_label} ({ai_confidence:.1f}%)", color='darkred', fontsize=14, fontweight='bold')
else:
    plt.imshow(original)
    plt.title(f"AI Focus (Error generating)\nPredicted: {predicted_label}", color='red')
plt.axis('off')

# รูปที่ 3: กราฟแท่งความมั่นใจ 14 โรค
plt.subplot(1, 3, 3)
y_pos = np.arange(len(CLASSES))
colors = ['red' if score > 0.5 else 'gray' for score in ensemble_preds]
plt.barh(y_pos, ensemble_preds, color=colors)
plt.yticks(y_pos, CLASSES, fontsize=10)
plt.xlabel('Confidence Score (0-1)')
plt.title(f"Ensemble Model Confidence\nTop Guess: {predicted_label}")
plt.axvline(x=0.5, color='blue', linestyle='--', linewidth=1, label='Threshold 0.5')
plt.legend()
plt.xlim(0, 1)

save_path = os.path.join(OUTPUT_DIR, f"predict_{random_image_file}")
plt.savefig(save_path, bbox_inches='tight', dpi=200)
plt.show()

print(f"\n🎉 ภาพถูกวิเคราะห์และเซฟไว้ที่: {save_path}")