import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import os

# =========================================================================
# CONFIG: รันจาก SeniorProject_BaseModel/ (root) ตามตำแหน่งที่พบไฟล์โมเดล
# ปรับ path ตรงนี้ถ้ารันจากตำแหน่งอื่น
# =========================================================================
CSV_PATH = 'data/csv/train_labels_14_hybrid.csv'   # ไฟล์ CSV ต้นฉบับของ Mixed AP+PA (19,470 ภาพ)
IMG_DIR = 'data/train_images_14_hybrid'
MODEL_DIR = 'models_v7'

MODEL_FILES = {
    'DenseNet121': 'best_DenseNet121_v7.keras',
    'ResNet50V2': 'best_ResNet50V2_v7.keras',
    'MobileNetV2': 'best_MobileNetV2_v7.keras',
}

IMG_SIZE = (384, 384)
BATCH_SIZE = 8

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

OUTPUT_DIR = 'outputs/graphs_v7_Mixed_V1_per_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =========================================================================

for name, path in [('CSV', CSV_PATH), ('IMG_DIR', IMG_DIR), ('MODEL_DIR', MODEL_DIR)]:
    if not os.path.exists(path):
        print(f"ERROR: ไม่เจอ {name} ที่ {path} - เช็ค path ก่อนรันใหม่")
        exit()

print("กำลังสร้าง Validation split เดิม (random_state=42, frac=0.8) ให้ตรงกับตอนเทรนจริง...")
df = pd.read_csv(CSV_PATH)
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)
print(f"Validation set: {len(val_df)} ภาพ (ต้องตรงกับ 3,894 ที่เคยบันทึกไว้)")

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df, directory=IMG_DIR, x_col='Filename', y_col=CLASSES,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', shuffle=False
)

y_true = val_generator.labels[:val_generator.samples]

individual_preds = {}
for m_name, fname in MODEL_FILES.items():
    model_path = os.path.join(MODEL_DIR, fname)
    print(f"\nกำลัง predict ด้วย: {m_name}")
    model = tf.keras.models.load_model(model_path, compile=False)
    preds = model.predict(val_generator, steps=len(val_generator), verbose=1)
    individual_preds[m_name] = preds[:val_generator.samples]
    del model
    tf.keras.backend.clear_session()
    val_generator.reset()

# คำนวณ ensemble ด้วย เพื่อเช็คว่าตรงกับ 0.8140 เดิมมั้ย (sanity check)
ensemble_preds = sum(individual_preds.values()) / len(individual_preds)
individual_preds['Ensemble (sanity check)'] = ensemble_preds

print("\n" + "="*60)
print("V7 MIXED AP+PA V1 - AUC รายโมเดล (คำนวณจากโมเดลจริงที่เทรนไว้)")
print("="*60)

summary_rows = []
for name, preds in individual_preds.items():
    aucs = []
    for i, disease in enumerate(CLASSES):
        try:
            auc = roc_auc_score(y_true[:, i], preds[:, i])
        except ValueError:
            auc = np.nan
        aucs.append(auc)
    macro_auc = np.nanmean(aucs)
    print(f"{name:25} Macro AUC: {macro_auc:.4f}")
    summary_rows.append([name] + aucs + [macro_auc])

summary_df = pd.DataFrame(summary_rows, columns=['Model'] + CLASSES + ['Macro_AUC'])
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'Mixed_V1_per_model_AUC.csv'), index=False)

print(f"\nบันทึกผลไว้ที่: {OUTPUT_DIR}/Mixed_V1_per_model_AUC.csv")
print("\nหมายเหตุ: ถ้า 'Ensemble (sanity check)' ได้ ~0.8140 แปลว่าคำนวณถูกต้องตรงกับผลเดิมที่เคยบันทึกไว้")