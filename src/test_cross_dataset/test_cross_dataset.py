import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import os

# ================= ⚙️ 1. CONFIGURATION =================
# เปลี่ยนจาก ../../ เป็น ../ (ถอยหลังแค่ 1 ชั้นเพื่อออกมาที่โปรเจกต์หลัก)
CSV_PATH = '../data/external_test/pneumonia_AP_report.csv'
IMG_DIR = '../data/external_test/pneumonia_ap_images'
MODELS_DIR = '../models_v7'

# โมเดลที่จะใช้ (ใช้ DenseNet121 เป็นตัวแทนก่อนเพื่อความรวดเร็ว)
MODEL_PATH = os.path.join(MODELS_DIR, 'best_DenseNet121_v7.keras')

# โรคที่เราสนใจ (เป้าหมายคือ Pneumonia)
TARGET_DISEASE = 'Pneumonia'
TARGET_INDEX = 6  # Pneumonia อยู่ตำแหน่งที่ 6 ในลิสต์ CLASSES ของเรา

# ================= 🔧 2. ฟังก์ชันปรับแสง CLAHE =================
def apply_clahe(img_array):
    # ภาพที่โหลดมาเป็น float32 (0-1) ต้องแปลงเป็น uint8 (0-255) ก่อนทำ CLAHE
    img_uint8 = (img_array * 255).astype(np.uint8)
    
    # ถ้าเป็นภาพสี ให้แปลงเป็นขาวดำก่อน
    if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_uint8
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    
    # แปลงกลับเป็น 3 Channel และสเกลกลับเป็น 0-1
    clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    return clahe_rgb.astype(np.float32) / 255.0

# ================= 🚀 3. เริ่มกระบวนการทดสอบ =================
print("⏳ Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("⏳ Loading CSV...")
df = pd.read_csv(CSV_PATH)
# กรองเอาเฉพาะเคสที่เป็น Pneumonia จริงๆ (ป้องกัน Noise)
df = df[df['Labels'].str.contains('pneumonia', case=False, na=False)].reset_index(drop=True)
print(f"✅ พบผู้ป่วย {TARGET_DISEASE} จำนวน {len(df)} เคส")

results = []

for idx, row in df.iterrows():
    img_name = row['ImageID']
    img_path = os.path.join(IMG_DIR, img_name)
    
    if not os.path.exists(img_path):
        continue  # ถ้ารูปไหนไม่มีในโฟลเดอร์ให้ข้ามไป
        
    # โหลดรูปภาพขนาด 384x384
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))
    img_normalized = img.astype(np.float32) / 255.0
    
    # 🌟 แบบที่ 1: Baseline (ไม่ปรับอะไรเลย)
    img_batch_base = np.expand_dims(img_normalized, axis=0)
    pred_base = model.predict(img_batch_base, verbose=0)[0]
    score_base = pred_base[TARGET_INDEX]
    
    # 🌟 แบบที่ 2: ใช้ CLAHE (ปรับแสงให้ชัดขึ้น)
    img_clahe = apply_clahe(img_normalized)
    img_batch_clahe = np.expand_dims(img_clahe, axis=0)
    pred_clahe = model.predict(img_batch_clahe, verbose=0)[0]
    score_clahe = pred_clahe[TARGET_INDEX]
    
    results.append({
        'Image': img_name,
        'Baseline_Score': score_base,
        'CLAHE_Score': score_clahe
    })

# ================= 📊 4. สรุปผลลัพธ์ =================
results_df = pd.DataFrame(results)

if len(results_df) > 0:
    mean_base = results_df['Baseline_Score'].mean()
    mean_clahe = results_df['CLAHE_Score'].mean()
    
    print("\n" + "="*40)
    print(f"📊 ผลทดสอบ CROSS-DATASET ({TARGET_DISEASE} ท่า AP)")
    print("="*40)
    print(f"จำนวนภาพที่ทดสอบ : {len(results_df)} ภาพ")
    print(f"คะแนนความมั่นใจเฉลี่ย (ภาพดิบ)     : {mean_base:.4f} ({mean_base*100:.1f}%)")
    print(f"คะแนนความมั่นใจเฉลี่ย (ใช้ CLAHE) : {mean_clahe:.4f} ({mean_clahe*100:.1f}%)")
    print("="*40)
    
    # เซฟตารางเทียบผล
    results_df.to_csv('cross_dataset_pneumonia_ap.csv', index=False)
    print("💾 เซฟผลลัพธ์ลงไฟล์ cross_dataset_pneumonia_ap.csv แล้ว")
else:
    print("❌ ไม่พบรูปภาพในโฟลเดอร์ โปรดเช็ค Path อีกครั้ง")