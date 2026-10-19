import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import os

# ไม่จอง VRAM ทั้งก้อนตั้งแต่เริ่ม (เหมือนสคริปต์เทรน) กัน cuDNN init ไม่ผ่านเมื่อมี process อื่นใช้ GPU อยู่
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

# =========================================================================
# CONFIG: รันจากโฟลเดอร์ src/ (ตามโครงสร้างที่เห็นใน Explorer)
# =========================================================================
MODEL_SETS = {
    'Mixed_AP_PA': '../models_v7',
    'PA_only':     '../models_v7_PA_only',
    'AP_only':     '../models_v7_AP_only',
}

MODEL_FILES = {
    'DenseNet121': 'best_DenseNet121_v7.keras',
    'ResNet50V2': 'best_ResNet50V2_v7.keras',
    'MobileNetV2': 'best_MobileNetV2_v7.keras',
}

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

IMG_SIZE = (384, 384)
BATCH_SIZE = 8

OUTPUT_DIR = 'outputs/full_matrix_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================================
# Matrix การทดสอบ: model -> dataset ไหนต้อง test บ้าง (ตามแผนที่วางไว้)
# =========================================================================
MODEL_TEST_MAP = {
    # ครบ 3x3: ทุกโมเดล ทดสอบกับ AP / PA / AP+PA ทั้ง NIH และ Padchest
    # (เพิ่ม cross-view: AP-only บน PA, PA-only บน AP)
    'AP_only':     ['NIH_test_AP', 'NIH_test_PA', 'NIH_test_All',
                    'Padchest_AP', 'Padchest_PA', 'Padchest_AP_PA'],
    'PA_only':     ['NIH_test_PA', 'NIH_test_AP', 'NIH_test_All',
                    'Padchest_PA', 'Padchest_AP', 'Padchest_AP_PA'],
    'Mixed_AP_PA': ['NIH_test_PA', 'NIH_test_AP', 'NIH_test_All',
                    'Padchest_PA', 'Padchest_AP', 'Padchest_AP_PA'],
}

# =========================================================================
# NIH test set (แยก AP / PA / All)
# =========================================================================
NIH_TEST_CSV = '../data/csv/test_labels_14_hybrid.csv'
NIH_META_CSV = '../data/csv/Data_Entry_2017.csv'
NIH_IMG_DIR = '../data/all_images_raw'

def build_nih_test_sets():
    if not os.path.exists(NIH_TEST_CSV) or not os.path.exists(NIH_META_CSV):
        print("SKIP NIH test set: ไม่เจอไฟล์ CSV ที่ต้องใช้")
        return {}
    if not os.path.exists(NIH_IMG_DIR):
        print(f"SKIP NIH test set: ไม่เจอโฟลเดอร์ {NIH_IMG_DIR}")
        return {}

    labels_df = pd.read_csv(NIH_TEST_CSV)
    meta_df = pd.read_csv(NIH_META_CSV, usecols=['Image Index', 'View Position'])
    meta_df = meta_df.rename(columns={'Image Index': 'Filename', 'View Position': 'ViewPosition'})

    merged = labels_df.merge(meta_df, on='Filename', how='left')
    missing = merged['ViewPosition'].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} ภาพหา View Position ไม่เจอ (ตัดออก)")
        merged = merged.dropna(subset=['ViewPosition'])

    result = {}
    result['NIH_test_All'] = (merged[['Filename'] + CLASSES].reset_index(drop=True), NIH_IMG_DIR)
    print(f"  NIH_test_All: พร้อมทดสอบ {len(merged)} ภาพ")

    for view in ['AP', 'PA']:
        subset = merged[merged['ViewPosition'] == view].reset_index(drop=True)
        if len(subset) == 0:
            print(f"  NIH_test_{view}: ไม่มีภาพ")
            continue
        result[f'NIH_test_{view}'] = (subset[['Filename'] + CLASSES], NIH_IMG_DIR)
        print(f"  NIH_test_{view}: พร้อมทดสอบ {len(subset)} ภาพ")

    return result

# =========================================================================
# Padchest test set (รวม AP+AP_horizontal เป็นกลุ่ม AP, แยก PA, และ All = PA+AP+AP_horizontal)
# Lateral (L) ไม่รวมใน matrix นี้ เพราะเป็นคนละ plane กับ AP/PA
# =========================================================================
PADCHEST_LABELS_CSV = '../data/csv/padchest_test_v7_format.csv'
PADCHEST_FULL_META_CSV = '../data/csv/PadChest/PADCHEST_chest_x_ray_images_labels.csv'

PADCHEST_IMG_DIRS = {
    'PA':            '../data/Padchest/100_padchest_PA_chest',
    'AP':            '../data/Padchest/100_padchest_AP_chest',
    'AP_horizontal': '../data/Padchest/100_padchest_AP_horizontal_chest',
}

def build_padchest_test_sets():
    if not os.path.exists(PADCHEST_LABELS_CSV) or not os.path.exists(PADCHEST_FULL_META_CSV):
        print("SKIP Padchest: ไม่เจอไฟล์ CSV ที่ต้องใช้")
        return {}

    labels_df = pd.read_csv(PADCHEST_LABELS_CSV)
    labels_df['basename'] = labels_df['Filename'].apply(lambda x: os.path.basename(x))

    meta_df = pd.read_csv(PADCHEST_FULL_META_CSV, usecols=['ImageID', 'Projection'])
    merged = labels_df.merge(meta_df, left_on='basename', right_on='ImageID', how='left')

    # เก็บ (basename, img_dir, proj_group) ของทุกภาพที่มีอยู่จริง แยกตาม PA / AP / AP_horizontal
    collected = {}  # proj_group -> dataframe rows with Filename+CLASSES, ต้องมี img_dir คู่กัน เพราะรูปอยู่คนละโฟลเดอร์
    per_group_rows = {'PA': [], 'AP': [], 'AP_horizontal': []}

    for proj_key, img_dir in PADCHEST_IMG_DIRS.items():
        if not os.path.exists(img_dir):
            print(f"SKIP Padchest_{proj_key}: ไม่เจอโฟลเดอร์ {img_dir}")
            continue
        available_files = set(os.listdir(img_dir))
        subset = merged[(merged['Projection'] == proj_key) & (merged['basename'].isin(available_files))].copy()
        if len(subset) == 0:
            print(f"SKIP Padchest_{proj_key}: ไม่มีภาพที่ label ตรงกับไฟล์ที่มีอยู่จริง")
            continue
        subset['Filename'] = subset['basename']
        subset['__img_dir__'] = img_dir
        per_group_rows[proj_key] = subset[['Filename', '__img_dir__'] + CLASSES]
        print(f"  Padchest_{proj_key}: พบ {len(subset)} ภาพ")

    result = {}

    # เพราะแต่ละ projection อยู่คนละโฟลเดอร์จริง เราจึง evaluate แยกทีละโฟลเดอร์ แล้วค่อยรวม prediction
    # เก็บ raw data ไว้ก่อน ฟังก์ชันด้านล่างจะจัดการรวมตอน predict จริง
    if len(per_group_rows['PA']) > 0:
        result['Padchest_PA'] = {'PA': per_group_rows['PA']}

    ap_parts = {k: v for k, v in per_group_rows.items() if k in ['AP', 'AP_horizontal'] and len(v) > 0}
    if ap_parts:
        result['Padchest_AP'] = ap_parts

    all_parts = {k: v for k, v in per_group_rows.items() if len(v) > 0}
    if all_parts:
        result['Padchest_AP_PA'] = all_parts

    return result

# =========================================================================
# ฟังก์ชันหลัก
# =========================================================================
def make_generator_from_df(df, img_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_dataframe(
        dataframe=df, directory=img_dir, x_col='Filename', y_col=CLASSES,
        target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', shuffle=False
    )
    return generator

def load_ensemble_models(model_dir):
    models = {}
    for m_name, fname in MODEL_FILES.items():
        model_path = os.path.join(model_dir, fname)
        if not os.path.exists(model_path):
            print(f"    SKIP: ไม่เจอไฟล์โมเดล {model_path}")
            continue
        models[m_name] = tf.keras.models.load_model(model_path, compile=False)
    return models

def predict_ensemble_on_df(models, df, img_dir):
    generator = make_generator_from_df(df, img_dir)
    individual_preds = {}
    for m_name, model in models.items():
        preds = model.predict(generator, steps=len(generator), verbose=0)
        individual_preds[m_name] = preds[:generator.samples]
        generator.reset()
    if not individual_preds:
        return None, None
    ensemble_preds = sum(individual_preds.values()) / len(individual_preds)
    y_true = generator.labels[:generator.samples]
    return ensemble_preds, y_true

def calc_auc_per_disease(y_true, preds):
    aucs = {}
    for i, disease in enumerate(CLASSES):
        col_true = y_true[:, i]
        if len(np.unique(col_true)) < 2:
            aucs[disease] = np.nan
            continue
        try:
            aucs[disease] = roc_auc_score(col_true, preds[:, i])
        except ValueError:
            aucs[disease] = np.nan
    return aucs

# =========================================================================
# เตรียม dataset ทั้งหมด
# =========================================================================
print("="*60)
print("Preparing NIH test sets (by view position)...")
print("="*60)
nih_sets = build_nih_test_sets()

print("\n" + "="*60)
print("Preparing Padchest test sets (by projection)...")
print("="*60)
padchest_sets = build_padchest_test_sets()

# =========================================================================
# MAIN LOOP: ตาม MODEL_TEST_MAP เท่านั้น (ไม่ใช่ทุก combo)
# =========================================================================
all_results = []
macro_summary = []

for model_name, model_dir in MODEL_SETS.items():
    if not os.path.exists(model_dir):
        print(f"\nSKIP model '{model_name}': ไม่เจอโฟลเดอร์ {model_dir}")
        continue

    print("\n" + "="*60)
    print(f"MODEL: {model_name}")
    print("="*60)
    models = load_ensemble_models(model_dir)
    if not models:
        continue

    dataset_keys = MODEL_TEST_MAP.get(model_name, [])
    for dataset_name in dataset_keys:

        # --- NIH datasets: ใช้ df, img_dir ตรงๆ ---
        if dataset_name in nih_sets:
            df, img_dir = nih_sets[dataset_name]
            print(f"\n  Testing on: {dataset_name} ({len(df)} images)")
            preds, y_true = predict_ensemble_on_df(models, df, img_dir)

        # --- Padchest datasets: อาจต้องรวมหลายโฟลเดอร์ (AP+AP_horizontal, หรือ All) ---
        elif dataset_name in padchest_sets:
            parts = padchest_sets[dataset_name]
            preds_parts = []
            ytrue_parts = []
            total_n = sum(len(v) for v in parts.values())
            print(f"\n  Testing on: {dataset_name} ({total_n} images across {len(parts)} folder(s))")
            for proj_key, df_part in parts.items():
                img_dir = PADCHEST_IMG_DIRS[proj_key]
                df_clean = df_part.drop(columns=['__img_dir__'])
                p, y = predict_ensemble_on_df(models, df_clean, img_dir)
                if p is not None:
                    preds_parts.append(p)
                    ytrue_parts.append(y)
            if preds_parts:
                preds = np.concatenate(preds_parts, axis=0)
                y_true = np.concatenate(ytrue_parts, axis=0)
            else:
                preds, y_true = None, None
        else:
            print(f"\n  SKIP: ไม่เจอ dataset '{dataset_name}' ที่เตรียมไว้")
            continue

        if preds is None:
            continue

        aucs = calc_auc_per_disease(y_true, preds)
        macro_auc = np.nanmean(list(aucs.values()))

        for disease, auc_val in aucs.items():
            all_results.append({'Model': model_name, 'Dataset': dataset_name, 'Disease': disease, 'AUC': auc_val})

        macro_summary.append({'Model': model_name, 'Dataset': dataset_name, 'MacroAUC': macro_auc, 'N_images': len(y_true)})
        print(f"    Macro AUC: {macro_auc:.4f}  (N={len(y_true)})")

    # เคลียร์โมเดลออกจาก memory ก่อนไปโมเดลถัดไป
    del models
    tf.keras.backend.clear_session()

# =========================================================================
# SAVE RESULTS
# =========================================================================
results_df = pd.DataFrame(all_results)
macro_df = pd.DataFrame(macro_summary)

results_df.to_csv(os.path.join(OUTPUT_DIR, 'per_disease_AUC_full_matrix.csv'), index=False)
macro_df.to_csv(os.path.join(OUTPUT_DIR, 'macro_AUC_full_matrix.csv'), index=False)

print("\n" + "="*60)
print("MACRO AUC SUMMARY (FULL MATRIX)")
print("="*60)
print(macro_df.to_string(index=False))

# GRAPH 1: Macro AUC grouped bar chart (Dataset x Model)
if not macro_df.empty:
    pivot = macro_df.pivot(index='Dataset', columns='Model', values='MacroAUC')
    fig, ax = plt.subplots(figsize=(max(10, len(pivot)*1.8), 6))
    pivot.plot(kind='bar', ax=ax, colormap='Set2')
    ax.set_ylabel('Macro AUC', fontsize=12)
    ax.set_xlabel('Test Dataset', fontsize=12)
    ax.set_title('Macro AUC: Model x Dataset (Full Matrix)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    ax.legend(title='Model')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Macro_AUC_full_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

# GRAPH 2: Heatmap per model
if not results_df.empty:
    for model_name in results_df['Model'].unique():
        sub = results_df[results_df['Model'] == model_name]
        heat_pivot = sub.pivot(index='Disease', columns='Dataset', values='AUC')
        fig, ax = plt.subplots(figsize=(max(6, len(heat_pivot.columns)*1.8), 8))
        sns.heatmap(heat_pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0, ax=ax)
        ax.set_title(f'AUC Heatmap: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'Heatmap_{model_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

print(f"\nSaved all results to: {OUTPUT_DIR}")
print("  - per_disease_AUC_full_matrix.csv")
print("  - macro_AUC_full_matrix.csv")
print("  - Macro_AUC_full_matrix.png")
print("  - Heatmap_<model_name>.png (per model)")
print("="*60)