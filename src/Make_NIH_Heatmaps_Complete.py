import os
import re
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from sklearn.metrics import roc_auc_score

# =========================================================================
# CONFIG: รันจากโฟลเดอร์ src/
# =========================================================================
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]
IMG_SIZE = (384, 384)
BATCH_SIZE = 8
LOAD_WORKERS = 6

MODEL_FILES = {
    'DenseNet121': 'best_DenseNet121_v7.keras',
    'ResNet50V2': 'best_ResNet50V2_v7.keras',
    'MobileNetV2': 'best_MobileNetV2_v7.keras',
}

# ชื่อบนภาพ (ต้องตรงกับชื่อโมเดลในไฟล์ผลทดสอบ) -> (family, โฟลเดอร์โมเดล)
MODEL_SETS = {
    # ชื่อแบบ version (ผลจาก Evaluate_All_Versions.py) -- ตัวไหนไม่มีในไฟล์ผลหรือไม่มีโฟลเดอร์โมเดลจะถูกข้ามเอง
    'Mixed_V1': ('Mixed', '../models_v7_Mixed_V1'),
    'Mixed_V2': ('Mixed', '../models_v7_Mixed_V2'),
    'Mixed_V3': ('Mixed', '../models_v7'),
    'AP_V1':    ('AP',    '../models_v7_AP_V2'),
    'AP_V2':    ('AP',    '../models_v7_AP_only'),
    'AP_V3':    ('AP',    '../models_v7_AP_V1'),
    'PA_V1':    ('PA',    '../models_v7_PA_only'),
    'PA_V2':    ('PA',    '../models_v7_PA_V3'),
    'PA_V3':    ('PA',    '../models_v7_PA_V1'),
    # ชื่อแบบเดิม (ผลจาก Evaluate_Full_Matrix.py 3 โมเดล)
    'Mixed_AP_PA': ('Mixed', '../models_v7'),
    'AP_only':     ('AP',    '../models_v7_AP_only'),
    'PA_only':     ('PA',    '../models_v7_PA_only'),
}

FAMILY_CSV = {
    'Mixed': '../data/csv/train_labels_14_hybrid.csv',
    'AP':    '../data/csv/train_labels_14_hybrid_AP.csv',
    'PA':    '../data/csv/train_labels_14_hybrid_PA.csv',
}
NIH_META_CSV = '../data/csv/Data_Entry_2017.csv'
NIH_IMG_DIR = '../data/all_images_raw'

RESULT_CANDIDATES = [
    'outputs/all_versions_results/per_disease_AUC_all_versions.csv',
    'outputs/full_matrix_results/per_disease_AUC_full_matrix.csv',
]
NIH_COLS = ['NIH_test_AP', 'NIH_test_PA', 'NIH_test_AP_PA']

OUT_DIR = 'outputs/nih_heatmaps'
CACHE_DIR = 'outputs/eval_cache'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# =========================================================================
# 1) อ่านผลทดสอบ NIH เดิม
# =========================================================================
def load_results():
    for p in RESULT_CANDIDATES:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"ใช้ผลการทดสอบจาก: {p}")
            is_new = 'ModelSet' in df.columns          # ผลจาก Evaluate_All_Versions.py (แก้ภาพ Padchest 16-bit แล้ว)
            if is_new:
                df = df[df['Network'] == 'Ensemble'].rename(columns={'ModelSet': 'Key', 'TestSet': 'Dataset'})
            else:
                df = df.rename(columns={'Model': 'Key'})
            # ไฟล์ผลรุ่นเก่าใช้ชื่อ NIH_test_All -> แปลงเป็นชื่อใหม่
            df['Dataset'] = df['Dataset'].replace({'NIH_test_All': 'NIH_test_AP_PA'})
            return df[['Key', 'Dataset', 'Disease', 'AUC']], is_new
    raise SystemExit("ไม่เจอไฟล์ผลการทดสอบ (per_disease_AUC_*.csv) ต้องรัน evaluate ก่อน")


# =========================================================================
# 2) สร้างชุดภาพ validation (split เดิม random_state=42) แยกตามท่า
# =========================================================================
def val_split(family, views):
    df = pd.read_csv(FAMILY_CSV[family])
    train = df.sample(frac=0.8, random_state=42)
    val = df.drop(train.index).copy()
    val = val.merge(views, on='Filename', how='left')
    return val


def build_eval_sets(family, views):
    """คืน dict: คอลัมน์ NIH -> DataFrame ของภาพที่ 'โมเดลนี้ไม่เคยใช้ปรับน้ำหนัก'
    - Mixed : validation ของ Mixed แยกตามท่า
    - AP    : AP = validation ของ AP-only, PA = ภาพ PA จาก validation ของ Mixed (AP-only ไม่เคยเห็นภาพ PA เลย)
    - PA    : PA = validation ของ PA-only, AP = ภาพ AP จาก validation ของ Mixed (PA-only ไม่เคยเห็นภาพ AP เลย)"""
    if family == 'Mixed':
        v = val_split('Mixed', views)
        sets = {'NIH_test_AP': v[v.View == 'AP'], 'NIH_test_PA': v[v.View == 'PA'], 'NIH_test_AP_PA': v}
    else:
        own = val_split(family, views)
        mixed = val_split('Mixed', views)
        if family == 'AP':
            a, p = own[own.View == 'AP'], mixed[mixed.View == 'PA']
        else:
            a, p = mixed[mixed.View == 'AP'], own[own.View == 'PA']
        sets = {'NIH_test_AP': a, 'NIH_test_PA': p, 'NIH_test_AP_PA': pd.concat([a, p], ignore_index=True)}
    return {k: v.reset_index(drop=True) for k, v in sets.items()}


# =========================================================================
# 3) โหลดภาพ (ครั้งเดียว มี cache) และทำนาย
# =========================================================================
def load_one(path):
    try:
        img = Image.open(path)
        if img.mode.startswith('I'):
            arr = np.asarray(img).astype(np.float32)
            lo, hi = float(arr.min()), float(arr.max())
            img = Image.fromarray(((arr - lo) / (hi - lo + 1e-6) * 255.0).astype(np.uint8), mode='L')
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST)
        return np.asarray(img, dtype=np.uint8), True
    except Exception as e:
        print(f"   ERROR โหลดภาพไม่ได้: {path} ({e})")
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8), False


def load_images(files):
    key = hashlib.md5(('nihfill-v1|' + '|'.join(files)).encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f'nihfill_images_{key}.npy')
    if os.path.exists(path):
        print(f"โหลดภาพจาก cache: {path}")
        return np.load(path), key
    print(f"กำลังโหลดภาพ {len(files)} ภาพ (ครั้งเดียว)...")
    paths = [os.path.join(NIH_IMG_DIR, f) for f in files]
    images = np.empty((len(paths), IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
    bad = 0
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for i, (arr, ok) in enumerate(ex.map(load_one, paths)):
            images[i] = arr
            bad += (not ok)
            if (i + 1) % 1000 == 0:
                print(f"   โหลดแล้ว {i + 1}/{len(paths)}")
    if bad:
        raise SystemExit(f"โหลดภาพไม่ได้ {bad} ภาพ เช็ค NIH_IMG_DIR = {NIH_IMG_DIR}")
    np.save(path, images)
    return images, key


def predict_ensemble(name, model_dir, images, needed, data_key):
    """ทำนายเฉพาะแถวใน needed ด้วยทั้ง 3 เครือข่าย แล้วเฉลี่ย -> array (N,14) (แถวที่ไม่ต้องใช้เป็น NaN)"""
    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    paths = {n: os.path.join(model_dir, f) for n, f in MODEL_FILES.items()}
    miss = [p for p in paths.values() if not os.path.exists(p)]
    if miss:
        print(f"SKIP {name}: ไม่เจอไฟล์โมเดล {miss[0]}")
        return None
    sig = hashlib.md5(''.join(f"{os.path.getsize(p)}-{int(os.path.getmtime(p))}" for p in paths.values())
                      .encode()).hexdigest()[:8]
    ndx_key = hashlib.md5(','.join(map(str, needed)).encode()).hexdigest()[:8]
    cache = os.path.join(CACHE_DIR, f'nihfill_preds_{os.path.basename(os.path.normpath(model_dir))}_{data_key}_{ndx_key}_{sig}.npy')
    if os.path.exists(cache):
        print(f"[{name}] ใช้ผลทำนายจาก cache")
        return np.load(cache)

    acc = np.zeros((len(needed), len(CLASSES)), dtype=np.float64)
    for net, p in paths.items():
        print(f"[{name}] ทำนายด้วย {net} ({len(needed)} ภาพ)...")
        model = tf.keras.models.load_model(p, compile=False)
        out = []
        for i in range(0, len(needed), BATCH_SIZE):
            x = images[needed[i:i + BATCH_SIZE]].astype(np.float32) / 255.0
            out.append(np.asarray(model.predict_on_batch(x)))
        acc += np.concatenate(out, axis=0)
        del model
        tf.keras.backend.clear_session()
    full = np.full((len(images), len(CLASSES)), np.nan)
    full[needed] = acc / len(MODEL_FILES)
    np.save(cache, full)
    return full


def auc_or_nan(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s)


# =========================================================================
# 4) วาด heatmap แบบเดิม
# =========================================================================
def draw_heatmap(key, table, path):
    hp = table.reindex(sorted(table.index))
    hp = hp[sorted(hp.columns)]                     # เรียงคอลัมน์แบบเดียวกับภาพเดิม
    hp.index.name = 'Disease'
    hp.columns.name = 'Dataset'
    fig, ax = plt.subplots(figsize=(max(6, len(hp.columns) * 1.8), 8))
    sns.heatmap(hp, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title(f'AUC Heatmap: {key}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


# =========================================================================
def compute_fill(res):
    """เติมช่อง NIH ที่ว่างด้วย AUC จากภาพ validation
    res: ตารางยาว ['Key','Dataset','Disease','AUC'] (Ensemble)
    คืน (filled_df, tables, keys) โดย tables[k] คือตาราง NIH (โรค x คอลัมน์) ที่เติมแล้ว"""
    views = pd.read_csv(NIH_META_CSV, usecols=['Image Index', 'View Position'])
    views = views.rename(columns={'Image Index': 'Filename', 'View Position': 'View'})

    keys = [k for k in MODEL_SETS if k in set(res['Key'])]
    missing_keys = [k for k in MODEL_SETS if k not in keys]
    if missing_keys:
        print(f"ข้าม (ไม่มีในไฟล์ผลทดสอบ): {missing_keys}")

    # ตารางจาก NIH test เดิม
    tables = {}
    for k in keys:
        sub = res[(res['Key'] == k) & (res['Dataset'].isin(NIH_COLS))]
        tables[k] = sub.pivot(index='Disease', columns='Dataset', values='AUC').reindex(index=CLASSES, columns=NIH_COLS)

    # หาเซลล์ที่ว่าง -> ต้องเติมจาก validation
    todo = {k: [(d, c) for d in CLASSES for c in NIH_COLS if np.isnan(tables[k].loc[d, c])] for k in keys}
    n_empty = sum(len(v) for v in todo.values())
    print(f"เซลล์ว่างทั้งหมด {n_empty} ช่อง " + str({k: len(v) for k, v in todo.items()}))

    filled_rows = []
    if n_empty:
        eval_sets = {k: build_eval_sets(MODEL_SETS[k][0], views) for k in keys if todo[k]}
        files = sorted({f for sets in eval_sets.values() for df in sets.values() for f in df['Filename']})
        images, data_key = load_images(files)
        row_of = {f: i for i, f in enumerate(files)}

        for k in eval_sets:
            needed = np.array(sorted({row_of[f] for df in eval_sets[k].values() for f in df['Filename']}))
            preds = predict_ensemble(k, MODEL_SETS[k][1], images, needed, data_key)
            if preds is None:
                continue
            for col, df in eval_sets[k].items():
                idx = np.array([row_of[f] for f in df['Filename']])
                for d in [d for d, c in todo[k] if c == col]:
                    j = CLASSES.index(d)
                    y = df[d].values.astype(int)
                    v = auc_or_nan(y, preds[idx, j])
                    tables[k].loc[d, col] = v
                    filled_rows.append({'Model': k, 'Column': col, 'Disease': d, 'AUC': v,
                                        'Positives': int(y.sum()), 'Images': len(y)})
    filled = pd.DataFrame(filled_rows, columns=['Model', 'Column', 'Disease', 'AUC', 'Positives', 'Images'])
    return filled, tables, keys


def complete_results(res_all):
    """ใช้จาก Evaluate_All_Versions.py: รับผลดิบ (คอลัมน์ ModelSet, TestSet, Network, Disease, AUC)
    คืน (complete, filled): complete = เฉพาะแถว Ensemble ที่เติมช่อง NIH ว่างแล้ว + คอลัมน์ Source
    ('test' หรือ 'validation') ใช้แยกที่มาของค่า"""
    ens = res_all[res_all['Network'] == 'Ensemble'].copy()
    ens['Source'] = 'test'
    r = ens.rename(columns={'ModelSet': 'Key', 'TestSet': 'Dataset'})[['Key', 'Dataset', 'Disease', 'AUC']]
    filled, _, _ = compute_fill(r)
    for row in filled.itertuples():
        m = ((ens['ModelSet'] == row.Model) & (ens['TestSet'] == row.Column) & (ens['Disease'] == row.Disease))
        ens.loc[m, 'AUC'] = row.AUC
        ens.loc[m, 'Source'] = 'validation'
    return ens, filled


if __name__ == "__main__":
    res, is_new = load_results()
    filled, tables, keys = compute_fill(res)
    filled_rows = filled.to_dict('records')

    for k in keys:
        draw_heatmap(k, tables[k], os.path.join(OUT_DIR, f'Heatmap_NIH_{k}.png'))
        tables[k].to_csv(os.path.join(OUT_DIR, f'Heatmap_NIH_{k}_values.csv'))
        left = int(np.isnan(tables[k].values).sum())
        print(f"สร้างภาพ NIH: {OUT_DIR}/Heatmap_NIH_{k}.png  (ช่องที่ยังว่าง: {left})")

    # ภาพรวม NIH + Padchest (6 คอลัมน์) -- ใช้เฉพาะผลรุ่นใหม่ที่แก้ภาพ Padchest 16-bit แล้ว
    if is_new:
        PAD_COLS = ['Padchest_AP', 'Padchest_PA', 'Padchest_AP_PA']
        for k in keys:
            sub = res[(res['Key'] == k) & (res['Dataset'].isin(PAD_COLS))]
            pad = sub.pivot(index='Disease', columns='Dataset', values='AUC').reindex(index=CLASSES, columns=PAD_COLS)
            full = pd.concat([tables[k], pad], axis=1)
            draw_heatmap(k, full, os.path.join(OUT_DIR, f'Heatmap_All_{k}.png'))
            full.to_csv(os.path.join(OUT_DIR, f'Heatmap_All_{k}_values.csv'))
            blanks = [(d, c) for d in CLASSES for c in full.columns if np.isnan(full.loc[d, c])]
            print(f"สร้างภาพรวม: {OUT_DIR}/Heatmap_All_{k}.png  (ช่องที่ยังว่าง: {len(blanks)})")
            if blanks:
                print("   ช่องว่าง:", blanks)
    else:
        print("\nข้ามภาพรวมที่มี Padchest: ไฟล์ผลที่ใช้เป็นรุ่นเก่า (คำนวณก่อนแก้ภาพ 16-bit) "
              "ให้รัน Evaluate_All_Versions.py ก่อน แล้วรันสคริปต์นี้อีกครั้ง")

    if filled_rows:
        fr = pd.DataFrame(filled_rows)
        fr.to_csv(os.path.join(OUT_DIR, 'filled_cells_from_validation.csv'), index=False)
        print("\nช่องที่เติมจากภาพ validation (ไว้ใช้เขียนหมายเหตุ):")
        print(fr.round(3).to_string(index=False))
        thin = fr[fr['Positives'] < 20]
        if len(thin):
            print("\nคำเตือน: ช่องเหล่านี้มี positive น้อยกว่า 20 ภาพ ค่าไม่เสถียร:")
            print(thin[['Model', 'Column', 'Disease', 'Positives']].to_string(index=False))