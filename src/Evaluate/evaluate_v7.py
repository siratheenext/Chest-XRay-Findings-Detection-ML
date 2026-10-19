# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os

# # --- ✅ SETUP: Mixed Precision ---
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')
# # ------------------------------------

# # ================= ⚙️ CONFIG V7 =================
# # Path ถอยหลัง 2 ก้าวให้ตรงกับตอน Train
# MODELS_DIR = '../../models_v7'
# OUTPUT_BASE_DIR = '../../outputs/reports_v7'
# CSV_PATH = '../../data/csv/train_labels_14_hybrid.csv'
# IMG_DIR = '../../data/train_images_14_hybrid'

# CLASSES = [
#     'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
#     'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
#     'Pleural_Thickening', 'Hernia'
# ]
# IMG_SIZE = (384, 384)   
# BATCH_SIZE = 16       

# # 🌟 การโหวตแบบให้น้ำหนัก V7
# MODEL_WEIGHTS = {
#     'best_DenseNet121_v7.keras': 1.0,
#     'best_ResNet50V2_v7.keras': 1.0,
#     'best_MobileNetV2_v7.keras': 1.0
# }

# os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# # ---------------------------------------------------------
# # 1. โหลดข้อมูล Validation (ให้ตรงกับตอน Train)
# # ---------------------------------------------------------
# print("⏳ Loading Validation Data...")
# df = pd.read_csv(CSV_PATH)
# train_df = df.sample(frac=0.8, random_state=42)
# val_df = df.drop(train_df.index)

# val_datagen = ImageDataGenerator(rescale=1./255)
# val_generator = val_datagen.flow_from_dataframe(
#     dataframe=val_df,
#     directory=IMG_DIR,
#     x_col='Filename',
#     y_col=CLASSES,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='raw',
#     shuffle=False 
# )

# # ---------------------------------------------------------
# # 2. โหลดโมเดล และทำ Ensemble Prediction
# # ---------------------------------------------------------
# y_true = val_generator.labels[:val_generator.samples]
# ensemble_preds = np.zeros((val_generator.samples, len(CLASSES)))
# total_weight_used = 0

# print(f"\n🚀 เริ่มทำการทดสอบโมเดล V7 Ensemble (รวมร่าง)...")

# for m_name, weight in MODEL_WEIGHTS.items():
#     model_path = os.path.join(MODELS_DIR, m_name)
#     if os.path.exists(model_path):
#         print(f"⏳ Running Inference: {m_name} (Weight: {weight})")
#         model = tf.keras.models.load_model(model_path, compile=False)
#         preds = model.predict(val_generator, steps=len(val_generator), verbose=1)
#         ensemble_preds += (preds[:val_generator.samples] * weight)
#         total_weight_used += weight
        
#         del model
#         tf.keras.backend.clear_session()
#     else:
#         print(f"❌ ไม่พบโมเดล: {model_path} (ข้าม...)")

# if total_weight_used > 0:
#     ensemble_preds = ensemble_preds / total_weight_used

# # ---------------------------------------------------------
# # 3. คำนวณ Metrics (เทียบ Actual vs Predicted)
# # ---------------------------------------------------------
# print("\n📊 Calculating V7 Evaluation Metrics...")

# individual_aucs = {}
# for i, disease in enumerate(CLASSES):
#     try:
#         auc = roc_auc_score(y_true[:, i], ensemble_preds[:, i])
#         individual_aucs[disease] = auc
#     except ValueError:
#         individual_aucs[disease] = 0.0
# macro_auc = np.mean(list(individual_aucs.values()))

# y_pred_binary = (ensemble_preds > 0.5).astype(int)

# # ---------------------------------------------------------
# # 4. SAVE ผลลัพธ์ทั้งหมดลงโฟลเดอร์ reports_v7
# # ---------------------------------------------------------
# # 4.1 เซฟ Report เป็น Text
# report_txt = classification_report(y_true, y_pred_binary, target_names=CLASSES, zero_division=0)
# with open(os.path.join(OUTPUT_BASE_DIR, "v7_ensemble_classification_report.txt"), "w") as f:
#     f.write(f"V7 THESIS ENSEMBLE MODEL REPORT (14 Classes)\n")
#     f.write(f"Total Weight Used: {total_weight_used}\n")
#     f.write(f"Macro AUC: {macro_auc:.4f}\n")
#     f.write("="*80 + "\n")
#     f.write(report_txt)

# # 4.2 เซฟ CSV Report (F1, Precision, Recall)
# report_dict = classification_report(y_true, y_pred_binary, target_names=CLASSES, zero_division=0, output_dict=True)
# pd.DataFrame(report_dict).transpose().to_csv(os.path.join(OUTPUT_BASE_DIR, "v7_ensemble_metrics_report.csv"))

# # 4.3 เซฟ AUC Scores เป็น CSV
# pd.DataFrame([individual_aucs]).transpose().to_csv(os.path.join(OUTPUT_BASE_DIR, "v7_ensemble_auc_scores.csv"))

# # 4.4 วาด Confusion Matrices 14 โรค
# print("📊 Generating 14 Confusion Matrices for V7...")
# cm_dir = os.path.join(OUTPUT_BASE_DIR, "confusion_matrices")
# os.makedirs(cm_dir, exist_ok=True)

# for i, disease in enumerate(CLASSES):
#     cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
#     plt.figure(figsize=(5, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
#     plt.title(f'{disease} (V7 Ensemble)')
#     plt.ylabel('Actual (หมอระบุ)')
#     plt.xlabel('Predicted (AI ทาย)')
#     plt.savefig(os.path.join(cm_dir, f'cm_{disease}.png'), bbox_inches='tight', dpi=150)
#     plt.close()

# print(f"\n🎉 การประเมินผล V7 เสร็จสมบูรณ์! ข้อมูลทั้งหมดอยู่ใน: {OUTPUT_BASE_DIR}")
# print(f"🏆 V7 MACRO AUC: {macro_auc:.4f}")

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- ✅ SETUP: Mixed Precision ---
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
# ------------------------------------

# ================= ⚙️ CONFIG V7 =================
MODELS_DIR = '../../models_v7'
OUTPUT_BASE_DIR = '../../outputs/reports_v7'
CSV_PATH = '../../data/csv/train_labels_14_hybrid.csv'
IMG_DIR = '../../data/train_images_14_hybrid'

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]
IMG_SIZE = (384, 384)   
BATCH_SIZE = 16       

# 🌟 การโหวตแบบให้น้ำหนัก V7
MODEL_WEIGHTS = {
    'best_DenseNet121_v7.keras': 1.0,
    'best_ResNet50V2_v7.keras': 1.0,
    'best_MobileNetV2_v7.keras': 1.0
}

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
ROC_DIR = os.path.join(OUTPUT_BASE_DIR, "roc_curves")
os.makedirs(ROC_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. โหลดข้อมูล Validation (ให้ตรงกับตอน Train)
# ---------------------------------------------------------
print("⏳ Loading Validation Data...")
df = pd.read_csv(CSV_PATH)
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=IMG_DIR,
    x_col='Filename',
    y_col=CLASSES,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=False 
)

# ---------------------------------------------------------
# 2. โหลดโมเดล และทำ Ensemble Prediction
# ---------------------------------------------------------
y_true = val_generator.labels[:val_generator.samples]
ensemble_preds = np.zeros((val_generator.samples, len(CLASSES)))
total_weight_used = 0

print(f"\n🚀 เริ่มทำการทดสอบโมเดล V7 Ensemble...")

for m_name, weight in MODEL_WEIGHTS.items():
    model_path = os.path.join(MODELS_DIR, m_name)
    if os.path.exists(model_path):
        print(f"⏳ Running Inference: {m_name} (Weight: {weight})")
        model = tf.keras.models.load_model(model_path, compile=False)
        preds = model.predict(val_generator, steps=len(val_generator), verbose=1)
        ensemble_preds += (preds[:val_generator.samples] * weight)
        total_weight_used += weight
        
        del model
        tf.keras.backend.clear_session()
    else:
        print(f"❌ ไม่พบโมเดล: {model_path} (ข้าม...)")

if total_weight_used > 0:
    ensemble_preds = ensemble_preds / total_weight_used

# ---------------------------------------------------------
# 3. 🌟 สร้าง ROC Curve และหา Optimal Thresholds (Youden's Index)
# ---------------------------------------------------------
print("\n📈 Calculating ROC Curves & Optimal Thresholds...")

individual_aucs = {}
optimal_thresholds = {}
thresholds_log = "V7 OPTIMAL THRESHOLDS (Youden's Index):\n" + "-"*40 + "\n"

for i, disease in enumerate(CLASSES):
    try:
        # คำนวณ ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true[:, i], ensemble_preds[:, i])
        auc_score = roc_auc_score(y_true[:, i], ensemble_preds[:, i])
        individual_aucs[disease] = auc_score
        
        # คำนวณ Youden's Index เพื่อหาจุดที่ดีที่สุด (J = TPR - FPR)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        opt_th = thresholds[optimal_idx]
        optimal_thresholds[disease] = opt_th
        
        # 🎨 วาดกราฟ ROC 
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # จุดแดงมาร์คค่า Threshold ที่ดีที่สุด
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100, zorder=5, 
                    label=f'Optimal Th = {opt_th:.3f}\n(Recall={tpr[optimal_idx]:.2f}, FPR={fpr[optimal_idx]:.2f})')
        
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC Curve: {disease}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(ROC_DIR, f'ROC_{disease}.png'), bbox_inches='tight', dpi=150)
        plt.close()
        
        thresholds_log += f"{disease:20} : {opt_th:.4f}\n"
        
    except ValueError:
        individual_aucs[disease] = 0.0
        optimal_thresholds[disease] = 0.5
        thresholds_log += f"{disease:20} : N/A (No positive cases)\n"

macro_auc = np.mean(list(individual_aucs.values()))

# เซฟไฟล์ Thresholds
with open(os.path.join(OUTPUT_BASE_DIR, "v7_optimal_thresholds.txt"), "w") as f:
    f.write(thresholds_log)

# ---------------------------------------------------------
# 4. 🌟 แปลงผลลัพธ์ Binary โดยใช้ Optimal Threshold ของแต่ละโรค
# ---------------------------------------------------------
y_pred_binary_optimal = np.zeros_like(ensemble_preds)
for i, disease in enumerate(CLASSES):
    th = optimal_thresholds[disease]
    y_pred_binary_optimal[:, i] = (ensemble_preds[:, i] > th).astype(int)

# ---------------------------------------------------------
# 5. SAVE ผลลัพธ์ทั้งหมดลงโฟลเดอร์ reports_v7
# ---------------------------------------------------------
print("\n💾 Saving Evaluation Reports...")

# 5.1 เซฟ Report เป็น Text (ใช้ค่า Threshold ใหม่)
report_txt = classification_report(y_true, y_pred_binary_optimal, target_names=CLASSES, zero_division=0)
with open(os.path.join(OUTPUT_BASE_DIR, "v7_ensemble_classification_report.txt"), "w") as f:
    f.write(f"V7 THESIS ENSEMBLE MODEL REPORT (Optimized Thresholds via Youden's Index)\n")
    f.write(f"Total Weight Used: {total_weight_used}\n")
    f.write(f"Macro AUC: {macro_auc:.4f}\n")
    f.write("="*80 + "\n")
    f.write(report_txt)

# 5.2 เซฟ CSV Report
report_dict = classification_report(y_true, y_pred_binary_optimal, target_names=CLASSES, zero_division=0, output_dict=True)
pd.DataFrame(report_dict).transpose().to_csv(os.path.join(OUTPUT_BASE_DIR, "v7_ensemble_metrics_report.csv"))

# 5.3 เซฟ AUC Scores เป็น CSV
pd.DataFrame([individual_aucs]).transpose().to_csv(os.path.join(OUTPUT_BASE_DIR, "v7_ensemble_auc_scores.csv"))

# 5.4 วาด Confusion Matrices 14 โรค (ใช้ค่า Threshold ใหม่)
cm_dir = os.path.join(OUTPUT_BASE_DIR, "confusion_matrices")
os.makedirs(cm_dir, exist_ok=True)

for i, disease in enumerate(CLASSES):
    cm = confusion_matrix(y_true[:, i], y_pred_binary_optimal[:, i])
    opt_th = optimal_thresholds[disease]
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{disease}\n(Optimized Th: {opt_th:.3f})')
    plt.ylabel('Actual (หมอระบุ)')
    plt.xlabel('Predicted (AI ทาย)')
    plt.savefig(os.path.join(cm_dir, f'cm_{disease}.png'), bbox_inches='tight', dpi=150)
    plt.close()

print(f"\n🎉 การประเมินผล V7 เสร็จสมบูรณ์! ข้อมูลทั้งหมดอยู่ใน: {OUTPUT_BASE_DIR}")
print(f"🏆 V7 MACRO AUC: {macro_auc:.4f}")
print("💡 อย่าลืมนำค่าในไฟล์ 'v7_optimal_thresholds.txt' ไปอัปเดตในไฟล์ Viewer ด้วยนะครับ!")