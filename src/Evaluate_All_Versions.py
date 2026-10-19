import os
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from sklearn.metrics import roc_auc_score

# ไม่จอง VRAM ทั้งก้อนตั้งแต่เริ่ม (เหมือนสคริปต์เทรน)
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

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

# นิยาม version: V1 = Fixed-120 + seed | V2 = 28% + seed | V3 = Fixed-120 ไม่ fix seed
# ชื่อโฟลเดอร์ใน disk ไม่ตรงกับเลข version จึงแมปด้วย config จริงตรงนี้
MODEL_SETS = {
    'Mixed_V1': '../models_v7_Mixed_V1',   # ก็อปมาจากเครื่อง lab: models_v7_original
    'Mixed_V2': '../models_v7_Mixed_V2',   # ก็อปมาจากเครื่อง lab: models_v7_original_freezefix
    'Mixed_V3': '../models_v7',
    'AP_V1':    '../models_v7_AP_V2',
    'AP_V2':    '../models_v7_AP_only',
    'AP_V3':    '../models_v7_AP_V1',
    'PA_V1':    '../models_v7_PA_only',
    'PA_V2':    '../models_v7_PA_V3',
    'PA_V3':    '../models_v7_PA_V1',
}

NIH_TEST_CSV = '../data/csv/test_labels_14_hybrid.csv'
NIH_META_CSV = '../data/csv/Data_Entry_2017.csv'
NIH_IMG_DIR = '../data/all_images_raw'

PADCHEST_LABELS_CSV = '../data/csv/padchest_test_v7_format.csv'
PADCHEST_FULL_META_CSV = '../data/csv/PadChest/PADCHEST_chest_x_ray_images_labels.csv'
PADCHEST_IMG_DIRS = {
    'PA':            '../data/Padchest/100_padchest_PA_chest',
    'AP':            '../data/Padchest/100_padchest_AP_chest',
    'AP_horizontal': '../data/Padchest/100_padchest_AP_horizontal_chest',
}

# เติมช่อง NIH ที่ไม่มี positive (Pneumonia, Fibrosis, Hernia) ด้วย AUC จากภาพ validation ให้ตารางและ heatmap ครบ
# ต้องมีไฟล์ Make_NIH_Heatmaps_Complete.py อยู่โฟลเดอร์เดียวกัน (ปิดได้ด้วย False ถ้าไม่ต้องการ)
FILL_NIH_BLANKS = True

OUTPUT_DIR = 'outputs/all_versions_results'
CACHE_DIR = 'outputs/eval_cache'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
PREPROC_TAG = 'v2-16bit-aware'   # เปลี่ยนค่านี้ถ้าแก้วิธีโหลดภาพ เพื่อบังคับสร้าง cache ใหม่


# =========================================================================
# 1) เตรียมรายการภาพทั้งหมด (โหลดทุกภาพครั้งเดียว ทุก test set เป็นแค่ subset ของรายการนี้)
# =========================================================================
def build_meta():
    parts = []

    labels_df = pd.read_csv(NIH_TEST_CSV)
    view_df = pd.read_csv(NIH_META_CSV, usecols=['Image Index', 'View Position'])
    view_df = view_df.rename(columns={'Image Index': 'Filename', 'View Position': 'Group'})
    nih = labels_df.merge(view_df, on='Filename', how='left')
    nih = nih[nih['Group'].isin(['AP', 'PA'])].copy()
    nih['Source'] = 'NIH'
    nih['Dir'] = NIH_IMG_DIR
    nih = nih[nih.apply(lambda r: os.path.exists(os.path.join(r['Dir'], r['Filename'])), axis=1)]
    parts.append(nih[['Source', 'Group', 'Dir', 'Filename'] + CLASSES])

    pl = pd.read_csv(PADCHEST_LABELS_CSV)
    pl['basename'] = pl['Filename'].apply(os.path.basename)
    pm = pd.read_csv(PADCHEST_FULL_META_CSV, usecols=['ImageID', 'Projection'])
    pad = pl.merge(pm, left_on='basename', right_on='ImageID', how='left')
    for proj, d in PADCHEST_IMG_DIRS.items():
        if not os.path.isdir(d):
            print(f"SKIP Padchest {proj}: ไม่เจอโฟลเดอร์ {d}")
            continue
        avail = set(os.listdir(d))
        sub = pad[(pad['Projection'] == proj) & (pad['basename'].isin(avail))].copy()
        sub['Filename'] = sub['basename']
        sub['Source'] = 'Padchest'
        sub['Group'] = proj
        sub['Dir'] = d
        parts.append(sub[['Source', 'Group', 'Dir', 'Filename'] + CLASSES])

    meta = pd.concat(parts, ignore_index=True)
    meta = meta.drop_duplicates(subset=['Source', 'Group', 'Filename']).reset_index(drop=True)
    return meta


def load_one(path):
    """โหลดภาพเป็น uint8 384x384x3
    - ภาพ 8-bit: ทำเหมือน Keras load_img เดิมเป๊ะ (convert RGB แล้ว resize แบบ nearest)
    - ภาพ 16/32-bit grayscale (mode ขึ้นต้นด้วย 'I'): ปรับช่วงค่าเป็น 0-255 ต่อภาพก่อน
      เพราะ PIL convert ตรงๆ จะตัดค่าเกิน 255 ทิ้งทำให้ภาพขาวจัด"""
    try:
        img = Image.open(path)
        mode = img.mode
        if mode.startswith('I'):
            arr = np.asarray(img).astype(np.float32)
            lo, hi = float(arr.min()), float(arr.max())
            arr = (arr - lo) / (hi - lo + 1e-6) * 255.0
            img = Image.fromarray(arr.astype(np.uint8), mode='L')
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST)
        return np.asarray(img, dtype=np.uint8), mode
    except Exception as e:
        print(f"   ERROR โหลดภาพไม่ได้: {path} ({e})")
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8), 'ERROR'


def load_all_images():
    meta = build_meta()
    ids = (meta['Source'] + '/' + meta['Group'] + '/' + meta['Filename']).tolist()
    key = hashlib.md5((PREPROC_TAG + '|' + '|'.join(ids)).encode()).hexdigest()[:10]
    img_path = os.path.join(CACHE_DIR, f'images_{key}.npy')
    meta_path = os.path.join(CACHE_DIR, f'meta_{key}.csv')

    if os.path.exists(img_path) and os.path.exists(meta_path):
        print(f"โหลดภาพจาก cache: {img_path}")
        meta = pd.read_csv(meta_path)
        images = np.load(img_path)
    else:
        print(f"กำลังโหลดภาพ {len(meta)} ภาพ (ครั้งเดียว, {LOAD_WORKERS} threads)...")
        paths = [os.path.join(d, f) for d, f in zip(meta['Dir'], meta['Filename'])]
        images = np.empty((len(paths), IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
        modes = []
        with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
            for i, (arr, mode) in enumerate(ex.map(load_one, paths)):
                images[i] = arr
                modes.append(mode)
                if (i + 1) % 500 == 0:
                    print(f"   โหลดแล้ว {i + 1}/{len(paths)}")
        meta['RawMode'] = modes
        ok = (meta['RawMode'] != 'ERROR').values
        if not ok.all():
            print(f"WARNING: ตัดภาพที่โหลดไม่ได้ออก {(~ok).sum()} ภาพ")
            meta = meta[ok].reset_index(drop=True)
            images = images[ok]
        np.save(img_path, images)
        meta.to_csv(meta_path, index=False)

    print("\nชนิดไฟล์ภาพดิบ (PIL mode) แยกตามแหล่งข้อมูล — ถ้ามี I;16 แปลว่าเป็นภาพ 16-bit:")
    print(meta.groupby(['Source', 'Group', 'RawMode']).size().to_string())
    print()
    return meta, images


# =========================================================================
# 2) ทำนาย: ทุกเครือข่ายทำนายทุกภาพครั้งเดียว แล้วเก็บ cache
# =========================================================================
def predict_all(model, images):
    out = []
    for i in range(0, len(images), BATCH_SIZE):
        x = images[i:i + BATCH_SIZE].astype(np.float32) / 255.0
        out.append(np.asarray(model.predict_on_batch(x)))
    return np.concatenate(out, axis=0)


def get_preds(set_name, model_dir, images, data_key):
    paths = {net: os.path.join(model_dir, f) for net, f in MODEL_FILES.items()}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        print(f"SKIP {set_name}: ไม่เจอไฟล์โมเดล {missing[0]} (และอาจมีอื่นอีก)")
        return None
    sig = hashlib.md5(''.join(f"{os.path.getsize(p)}-{int(os.path.getmtime(p))}"
                              for p in paths.values()).encode()).hexdigest()[:8]
    cache = os.path.join(CACHE_DIR, f'preds_{set_name}_{data_key}_{sig}.npz')
    if os.path.exists(cache):
        print(f"[{set_name}] ใช้ผลทำนายจาก cache")
        return dict(np.load(cache))

    preds = {}
    for net, p in paths.items():
        print(f"[{set_name}] ทำนายด้วย {net} ...")
        model = tf.keras.models.load_model(p, compile=False)
        preds[net] = predict_all(model, images)
        del model
        tf.keras.backend.clear_session()
    np.savez(cache, **preds)
    return preds


def auc_per_disease(y_true, scores):
    out = {}
    for i, d in enumerate(CLASSES):
        col = y_true[:, i]
        if len(np.unique(col)) < 2:
            out[d] = np.nan
            continue
        try:
            out[d] = roc_auc_score(col, scores[:, i])
        except ValueError:
            out[d] = np.nan
    return out


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    meta, images = load_all_images()
    data_key = hashlib.md5(('|'.join((meta['Source'] + '/' + meta['Group'] + '/' + meta['Filename']).tolist())
                            + PREPROC_TAG).encode()).hexdigest()[:10]
    y_true = meta[CLASSES].values.astype(int)

    is_nih = (meta['Source'] == 'NIH').values
    is_pad = (meta['Source'] == 'Padchest').values
    grp = meta['Group'].values
    TEST_SETS = {
        'NIH_test_AP':    is_nih & (grp == 'AP'),
        'NIH_test_PA':    is_nih & (grp == 'PA'),
        'NIH_test_AP_PA':   is_nih,
        'Padchest_AP':    is_pad & np.isin(grp, ['AP', 'AP_horizontal']),
        'Padchest_PA':    is_pad & (grp == 'PA'),
        'Padchest_AP_PA': is_pad,
    }
    print("จำนวนภาพต่อ test set:")
    for k, m in TEST_SETS.items():
        print(f"   {k:16} {int(m.sum())}")

    rows = []
    for set_name, model_dir in MODEL_SETS.items():
        preds = get_preds(set_name, model_dir, images, data_key)
        if preds is None:
            continue
        family, version = set_name.split('_')
        for ds_name, mask in TEST_SETS.items():
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            yt = y_true[idx]
            scores = {net: p[idx] for net, p in preds.items()}
            scores['Ensemble'] = sum(scores.values()) / len(MODEL_FILES)
            for net, s in scores.items():
                aucs = auc_per_disease(yt, s)
                for d, v in aucs.items():
                    rows.append({'ModelSet': set_name, 'Family': family, 'Version': version,
                                 'TestSet': ds_name, 'Network': net, 'Disease': d,
                                 'AUC': v, 'N': len(idx)})

    if not rows:
        print("ไม่มีผลลัพธ์เลย เช็ค path โมเดลใน MODEL_SETS")
        raise SystemExit

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTPUT_DIR, 'per_disease_AUC_all_versions.csv'), index=False)

    # เติมช่อง NIH ที่ว่าง -> ได้ตารางที่ครบทุกโรค (แยกไฟล์ต่างหาก ไฟล์ดิบข้างบนไม่ถูกแตะ)
    complete = None
    if FILL_NIH_BLANKS:
        try:
            import Make_NIH_Heatmaps_Complete as filler
        except ImportError:
            print("ข้ามการเติมช่องว่าง NIH: ไม่เจอ Make_NIH_Heatmaps_Complete.py ในโฟลเดอร์เดียวกัน")
        else:
            print("\nกำลังเติมช่อง NIH ที่ว่างจากภาพ validation (ขั้นนี้โหลดภาพและทำนายเพิ่ม ใช้เวลาอีกพักใหญ่)...")
            complete, filled = filler.complete_results(res)
            complete.to_csv(os.path.join(OUTPUT_DIR, 'per_disease_AUC_all_versions_complete.csv'), index=False)
            filled.to_csv(os.path.join(OUTPUT_DIR, 'filled_cells_from_validation.csv'), index=False)
            print(f"เติมแล้ว {len(filled)} ช่อง -> filled_cells_from_validation.csv (บอกจำนวน positive ของแต่ละช่อง)")

    macro = (res.groupby(['ModelSet', 'Family', 'Version', 'TestSet', 'Network', 'N'])['AUC']
                .mean().reset_index().rename(columns={'AUC': 'MacroAUC'}))
    macro.to_csv(os.path.join(OUTPUT_DIR, 'macro_AUC_all_versions.csv'), index=False)

    ens = macro[macro['Network'] == 'Ensemble']
    pivot = ens.pivot(index='ModelSet', columns='TestSet', values='MacroAUC')
    pivot = pivot[[c for c in TEST_SETS if c in pivot.columns]]
    pivot.to_csv(os.path.join(OUTPUT_DIR, 'ensemble_macro_pivot.csv'))
    print("\n" + "=" * 70)
    print("ENSEMBLE MACRO AUC (แถว = model set, คอลัมน์ = test set)")
    print("=" * 70)
    print(pivot.round(4).to_string())

    # กราฟ: แยกตาม family เทียบ V1/V2/V3 ในแต่ละ test set
    for family in ens['Family'].unique():
        sub = ens[ens['Family'] == family].pivot(index='TestSet', columns='Version', values='MacroAUC')
        sub = sub.reindex([c for c in TEST_SETS if c in sub.index])
        fig, ax = plt.subplots(figsize=(11, 6))
        sub.plot(kind='bar', ax=ax, colormap='Set2')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
        ax.set_ylabel('Ensemble Macro AUC')
        ax.set_title(f'{family}: V1 vs V2 vs V3 across test sets', fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'Macro_AUC_{family}_versions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Heatmap รายโรคของแต่ละ model set (Ensemble) -- ถ้าเติมช่องว่างแล้วใช้ตารางที่ครบ
    heat_src = complete if complete is not None else res[res['Network'] == 'Ensemble']
    for set_name in heat_src['ModelSet'].unique():
        sub = heat_src[heat_src['ModelSet'] == set_name]
        hp = sub.pivot(index='Disease', columns='TestSet', values='AUC')
        hp = hp[sorted(hp.columns)]
        hp.index.name, hp.columns.name = 'Disease', 'Dataset'
        fig, ax = plt.subplots(figsize=(max(6, len(hp.columns) * 1.8), 8))
        sns.heatmap(hp, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0, ax=ax)
        ax.set_title(f'AUC Heatmap: {set_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'Heatmap_{set_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        blanks = int(hp.isna().sum().sum())
        if blanks:
            print(f"   Heatmap_{set_name}.png ยังมีช่องว่าง {blanks} ช่อง")

    print(f"\nบันทึกผลทั้งหมดที่: {OUTPUT_DIR}")
    print("  - macro_AUC_all_versions.csv / per_disease_AUC_all_versions.csv / ensemble_macro_pivot.csv")
    print("  - Macro_AUC_<Family>_versions.png และ Heatmap_<ModelSet>.png")
    if complete is not None:
        print("  - per_disease_AUC_all_versions_complete.csv (ครบทุกโรค มีคอลัมน์ Source = test/validation)")
        print("หมายเหตุ: Macro AUC และกราฟเทียบ V1-V3 คำนวณจากผล test จริงเท่านั้น (NIH เฉลี่ย 11 โรค) ไม่รวมค่าที่เติมจาก validation")