import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import os

# =========================================================================
# CONFIG 1: โมเดลที่จะทดสอบ (ใส่ทุกชุดที่มี เทียบกันได้ในกราฟเดียว)
# =========================================================================
MODEL_SETS = {
    'Mixed_AP_PA': '../../models_v7',
    'PA_only':     '../../models_v7_PA_only',
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

OUTPUT_DIR = 'outputs/external_test_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================================
# CONFIG 2: เตรียม Padchest CSV แยกตาม Projection (PA / AP / AP_horizontal / L)
# padchest_test_v7_format.csv  -> Filename (มี prefix เช่น '0/xxx.png') + 14 disease columns
# PADCHEST_chest_x_ray_images_labels.csv -> ใช้ดึง Projection (PA/AP/AP_horizontal/L) ผ่าน ImageID
# =========================================================================
PADCHEST_LABELS_CSV = '../../data/csv/padchest_test_v7_format.csv'
PADCHEST_FULL_META_CSV = '../../data/csv/PadChest/PADCHEST_chest_x_ray_images_labels.csv'

# โฟลเดอร์รูปจริงที่ดาวน์โหลดมาแล้ว (ตามชื่อที่เห็นใน Explorer)
PADCHEST_IMG_DIRS = {
    'Padchest_PA':            '../../data/Padchest/100_padchest_PA_chest',
    'Padchest_AP':            '../../data/Padchest/100_padchest_AP_chest',
    'Padchest_AP_horizontal': '../../data/Padchest/100_padchest_AP_horizontal_chest',
    'Padchest_L':             '../../data/Padchest/100_padchest_L_chest',
}

def build_padchest_test_sets():
    """
    คืนค่า dict {dataset_name: (dataframe, img_dir)} โดยกรองเฉพาะภาพที่:
    1. มี label ครบใน padchest_test_v7_format.csv
    2. Projection ตรงกับกลุ่มนั้น (PA/AP/AP_horizontal/L)
    3. ไฟล์รูปมีอยู่จริงในโฟลเดอร์ (เพราะโหลดมาแค่ ~100 รูปต่อกลุ่ม ไม่ใช่ทั้งหมด)
    """
    if not os.path.exists(PADCHEST_LABELS_CSV) or not os.path.exists(PADCHEST_FULL_META_CSV):
        print("SKIP Padchest: ไม่เจอไฟล์ CSV ที่ต้องใช้")
        return {}

    labels_df = pd.read_csv(PADCHEST_LABELS_CSV)
    labels_df['basename'] = labels_df['Filename'].apply(lambda x: os.path.basename(x))

    meta_df = pd.read_csv(PADCHEST_FULL_META_CSV, usecols=['ImageID', 'Projection'])
    merged = labels_df.merge(meta_df, left_on='basename', right_on='ImageID', how='left')

    result = {}
    for dataset_name, img_dir in PADCHEST_IMG_DIRS.items():
        if not os.path.exists(img_dir):
            print(f"SKIP {dataset_name}: ไม่เจอโฟลเดอร์ {img_dir}")
            continue

        proj_key = dataset_name.replace('Padchest_', '')  # PA / AP / AP_horizontal / L
        subset = merged[merged['Projection'] == proj_key].copy()

        # เช็คว่าไฟล์มีอยู่จริงในโฟลเดอร์ (เพราะโหลดมาแค่บางส่วน)
        available_files = set(os.listdir(img_dir))
        subset = subset[subset['basename'].isin(available_files)].reset_index(drop=True)
        subset['Filename'] = subset['basename']  # ใช้ basename ตรงๆ (รูปอยู่ flat ในโฟลเดอร์)

        if len(subset) == 0:
            print(f"SKIP {dataset_name}: ไม่มีภาพที่ label ตรงกับไฟล์ที่มีอยู่จริง")
            continue

        result[dataset_name] = (subset[['Filename'] + CLASSES], img_dir)
        print(f"  {dataset_name}: พร้อมทดสอบ {len(subset)} ภาพ")

    return result

# =========================================================================
# CONFIG 3: NIH test set แยกตาม View Position (AP / PA / All)
# test_labels_14_hybrid.csv -> Filename + 14 disease columns (unseen ตอน train เพราะเป็น test split แยก)
# Data_Entry_2017.csv -> ใช้ดึง View Position (AP/PA) ผ่าน Image Index
# =========================================================================
NIH_TEST_CSV = '../../data/csv/test_labels_14_hybrid.csv'
NIH_META_CSV = '../../data/csv/Data_Entry_2017.csv'
NIH_IMG_DIR = '../../data/all_images_raw'

def build_nih_test_sets():
    """
    คืนค่า dict {dataset_name: (dataframe, img_dir)} แยกเป็น NIH_test_AP, NIH_test_PA, NIH_test_All
    """
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
# CONFIG 4: Dataset อื่นๆ ที่ CSV พร้อมใช้แล้ว (format Filename + 14 disease columns)
# =========================================================================
OTHER_TEST_SETS = {
    # ใส่ dataset อื่นเพิ่มได้ตรงนี้ตามรูปแบบ 'name': (csv_path, img_dir)
}

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

def get_ensemble_preds(model_dir, generator):
    individual_preds = {}
    for m_name, fname in MODEL_FILES.items():
        model_path = os.path.join(model_dir, fname)
        if not os.path.exists(model_path):
            print(f"    SKIP: ไม่เจอไฟล์โมเดล {model_path}")
            continue
        model = tf.keras.models.load_model(model_path, compile=False)
        preds = model.predict(generator, steps=len(generator), verbose=0)
        individual_preds[m_name] = preds[:generator.samples]
        del model
        tf.keras.backend.clear_session()
        generator.reset()
    if not individual_preds:
        return None
    return sum(individual_preds.values()) / len(individual_preds)

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
# รวม test sets ทั้งหมด: Padchest (แยกตาม projection) + อื่นๆ
# =========================================================================
print("="*60)
print("Preparing Padchest test sets (by projection)...")
print("="*60)
padchest_sets = build_padchest_test_sets()

print("\n" + "="*60)
print("Preparing NIH test sets (by view position)...")
print("="*60)
nih_sets = build_nih_test_sets()

all_test_sets = {}
for name, (df, img_dir) in padchest_sets.items():
    all_test_sets[name] = (df, img_dir)

for name, (df, img_dir) in nih_sets.items():
    all_test_sets[name] = (df, img_dir)

for name, (csv_path, img_dir) in OTHER_TEST_SETS.items():
    if os.path.exists(csv_path) and os.path.exists(img_dir):
        df = pd.read_csv(csv_path)
        all_test_sets[name] = (df, img_dir)
    else:
        print(f"SKIP {name}: ไม่เจอ CSV หรือโฟลเดอร์รูป")

# =========================================================================
# MAIN LOOP: ทุก dataset x ทุกโมเดล
# =========================================================================
all_results = []
macro_summary = []

for dataset_name, (df, img_dir) in all_test_sets.items():
    print("\n" + "="*60)
    print(f"DATASET: {dataset_name}")
    print("="*60)
    generator = make_generator_from_df(df, img_dir)
    y_true = generator.labels[:generator.samples]
    print(f"  จำนวนภาพ: {generator.samples}")

    for model_name, model_dir in MODEL_SETS.items():
        if not os.path.exists(model_dir):
            print(f"  SKIP model '{model_name}': ไม่เจอโฟลเดอร์ {model_dir}")
            continue

        print(f"\n  Testing model set: {model_name}")
        preds = get_ensemble_preds(model_dir, generator)
        if preds is None:
            continue

        aucs = calc_auc_per_disease(y_true, preds)
        macro_auc = np.nanmean(list(aucs.values()))

        for disease, auc_val in aucs.items():
            all_results.append({'Dataset': dataset_name, 'Model': model_name, 'Disease': disease, 'AUC': auc_val})

        macro_summary.append({'Dataset': dataset_name, 'Model': model_name, 'MacroAUC': macro_auc, 'N_images': generator.samples})
        print(f"    Macro AUC: {macro_auc:.4f}")

# =========================================================================
# SAVE RESULTS
# =========================================================================
results_df = pd.DataFrame(all_results)
macro_df = pd.DataFrame(macro_summary)

results_df.to_csv(os.path.join(OUTPUT_DIR, 'per_disease_AUC_all_datasets.csv'), index=False)
macro_df.to_csv(os.path.join(OUTPUT_DIR, 'macro_AUC_summary_all_datasets.csv'), index=False)

print("\n" + "="*60)
print("MACRO AUC SUMMARY (ALL DATASETS x ALL MODELS)")
print("="*60)
print(macro_df.to_string(index=False))

# GRAPH 1: Macro AUC grouped bar chart
if not macro_df.empty:
    pivot = macro_df.pivot(index='Dataset', columns='Model', values='MacroAUC')
    fig, ax = plt.subplots(figsize=(max(8, len(pivot)*2), 6))
    pivot.plot(kind='bar', ax=ax, colormap='Set2')
    ax.set_ylabel('Macro AUC', fontsize=12)
    ax.set_xlabel('Test Dataset', fontsize=12)
    ax.set_title('Macro AUC by Model across External Test Datasets', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    ax.legend(title='Model')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Macro_AUC_by_dataset_and_model.png'), dpi=300, bbox_inches='tight')
    plt.close()

# GRAPH 2: Heatmap per model
if not results_df.empty:
    for model_name in results_df['Model'].unique():
        sub = results_df[results_df['Model'] == model_name]
        heat_pivot = sub.pivot(index='Disease', columns='Dataset', values='AUC')
        fig, ax = plt.subplots(figsize=(max(6, len(heat_pivot.columns)*1.5), 8))
        sns.heatmap(heat_pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0, ax=ax)
        ax.set_title(f'AUC Heatmap: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'Heatmap_{model_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

print(f"\nSaved all results to: {OUTPUT_DIR}")
print("="*60)