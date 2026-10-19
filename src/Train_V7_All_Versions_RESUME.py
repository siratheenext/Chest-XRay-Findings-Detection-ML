import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, ResNet50V2, MobileNetV2
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras import mixed_precision
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import os
import numpy as np
import random

mixed_precision.set_global_policy('mixed_float16')

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

IMG_SIZE = (384, 384)
BATCH_SIZE = 8
EPOCHS = 40
INITIAL_LR = 1e-4

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ============================================================================
# CONFIG: 4 version ที่จะรันเรียงกัน (ทีละตัว ไม่ขนาน กัน VRAM ไม่พอ)
# แก้ path ตรงนี้ให้ตรงกับตำแหน่งที่รันสคริปต์นี้จริง
# ============================================================================
VERSIONS = [
    {
        'run_name': 'AP_V1',
        'csv_path': '../data/csv/train_labels_14_hybrid_AP.csv',
        'img_dir': '../data/train_images_14_hybrid_AP',
        'seed_on': False,
        'freeze_new': False,
    },
    {
        'run_name': 'AP_V2',
        'csv_path': '../data/csv/train_labels_14_hybrid_AP.csv',
        'img_dir': '../data/train_images_14_hybrid_AP',
        'seed_on': True,
        'freeze_new': False,
    },
    {
        'run_name': 'PA_V1',
        'csv_path': '../data/csv/train_labels_14_hybrid_PA.csv',
        'img_dir': '../data/train_images_14_hybrid_PA',
        'seed_on': False,
        'freeze_new': False,
    },
    {
        'run_name': 'PA_V3',
        'csv_path': '../data/csv/train_labels_14_hybrid_PA.csv',
        'img_dir': '../data/train_images_14_hybrid_PA',
        'seed_on': True,
        'freeze_new': True,
    },
]
# ============================================================================


def set_seed_if_needed(seed_on):
    if seed_on:
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        print("   Seed fixed: 42")
    else:
        print("   Seed NOT fixed (intentional for this version)")


def weighted_focal_loss(pos_weights, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_factor = y_true * pos_weights + (1 - y_true) * 1.0
        loss = -alpha_factor * modulating_factor * tf.math.log(p_t)
        return tf.reduce_mean(loss)
    return loss_fn


def build_model(model_name, pos_weights_tensor, freeze_new):
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    if model_name == 'DenseNet121':
        base = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    elif model_name == 'ResNet50V2':
        base = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    elif model_name == 'MobileNetV2':
        base = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

    base.trainable = True
    if freeze_new:
        UNFREEZE_RATIO = 0.28
        total_layers = len(base.layers)
        num_unfreeze = max(1, int(total_layers * UNFREEZE_RATIO))
        print(f"   {model_name}: total_layers={total_layers}, unfreezing last {num_unfreeze} ({UNFREEZE_RATIO*100:.0f}%)")
        for layer in base.layers[:-num_unfreeze]:
            layer.trainable = False
    else:
        for layer in base.layers[:-120]:
            layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(CLASSES), activation='sigmoid', dtype='float32')(x)

    model = models.Model(inputs=base.input, outputs=outputs, name=model_name)
    optimizer = optimizers.AdamW(learning_rate=INITIAL_LR, weight_decay=1e-4)
    model.compile(optimizer=optimizer,
                  loss=weighted_focal_loss(pos_weights_tensor, gamma=2.0),
                  metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc')])
    return model


def evaluate_and_save(name, preds, y_true, save_root, run_name):
    save_dir = os.path.join(save_root, name)
    cm_dir = os.path.join(save_dir, 'Confusion_Matrices')
    os.makedirs(cm_dir, exist_ok=True)

    individual_aucs = []
    auc_log_text = f"V7 [{name}] AUC SCORES (14 DISEASES) - RUN: {run_name}:\n" + "-"*40 + "\n"
    for i, disease in enumerate(CLASSES):
        try:
            auc = roc_auc_score(y_true[:, i], preds[:, i])
            individual_aucs.append(auc)
            line = f"   {disease:20} : {auc:.4f}"
        except ValueError:
            individual_aucs.append(0.0)
            line = f"   {disease:20} : N/A"
        auc_log_text += line + "\n"

    macro_auc = np.mean(individual_aucs)
    auc_log_text += "-"*40 + f"\n [{name}] MACRO AUC: {macro_auc:.4f}\n"
    with open(os.path.join(save_dir, 'AUC_Scores.txt'), 'w', encoding='utf-8') as f:
        f.write(auc_log_text)

    y_pred_binary = (preds > 0.5).astype(int)
    report = classification_report(y_true, y_pred_binary, target_names=CLASSES, zero_division=0)
    with open(os.path.join(save_dir, 'Classification_Report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"V7 [{name}] - CLASSIFICATION REPORT (Threshold 0.5) - RUN: {run_name}\n\n")
        f.write(report)

    for i, disease in enumerate(CLASSES):
        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'[{name}] Confusion Matrix: {disease}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(cm_dir, f'CM_{disease}.png'), bbox_inches='tight', dpi=150)
        plt.close()

    return macro_auc


def run_version(cfg):
    run_name = cfg['run_name']
    csv_path = cfg['csv_path']
    img_dir = cfg['img_dir']
    seed_on = cfg['seed_on']
    freeze_new = cfg['freeze_new']

    model_save_dir = f"../models_v7_{run_name}"
    graph_save_dir = f"outputs/graphs_v7_{run_name}"
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(graph_save_dir, exist_ok=True)

    # [RESUME] ถ้า version นี้ประเมินผลเสร็จสมบูรณ์แล้ว (มีไฟล์สรุปผล) ข้ามทั้งก้อนเลย
    summary_check = os.path.join(graph_save_dir, 'V7_Macro_AUC_Summary.txt')
    if os.path.exists(summary_check):
        print(f"\n[SKIP] {run_name}: มีผลประเมินเสร็จแล้วที่ {summary_check} - ข้าม version นี้ทั้งหมด")
        with open(summary_check, encoding='utf-8') as f:
            return f.read()

    print("\n" + "#" * 60)
    print(f"# STARTING VERSION: {run_name}  (seed_on={seed_on}, freeze_new={freeze_new})")
    print("#" * 60)

    set_seed_if_needed(seed_on)

    print(f" Loading Data... (RUN: {run_name})")
    df = pd.read_csv(csv_path)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    class_counts = train_df[CLASSES].sum()
    minority_classes = class_counts[class_counts < 2000].index.tolist()
    minority_df = train_df[train_df[minority_classes].sum(axis=1) > 0]
    train_df_balanced = pd.concat([train_df, minority_df], ignore_index=True)
    train_df = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"    Train size: {len(df)*0.8:.0f} -> {len(train_df)} after oversampling")

    pos_counts = train_df[CLASSES].sum().values
    total = len(train_df)
    weight_values = (total / (pos_counts + 1e-5)) / 2.0
    pos_weights_tensor = tf.constant(weight_values, dtype=tf.float32)

    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=10, width_shift_range=0.05, height_shift_range=0.05,
        zoom_range=0.1, fill_mode='constant', cval=0, horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df, directory=img_dir, x_col='Filename', y_col=CLASSES,
        target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', shuffle=True
    )
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df, directory=img_dir, x_col='Filename', y_col=CLASSES,
        target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='raw', shuffle=False
    )

    models_to_train = ['DenseNet121', 'ResNet50V2', 'MobileNetV2']
    saved_model_paths = []
    history_dict = {}

    for m_name in models_to_train:
        save_path = os.path.join(model_save_dir, f'best_{m_name}_v7.keras')
        saved_model_paths.append(save_path)

        # [RESUME] ถ้าโมเดลนี้มี checkpoint เซฟไว้แล้ว ข้ามการเทรนใหม่ ใช้ของเดิมเลย
        if os.path.exists(save_path):
            print(f"\n[SKIP] {m_name} (RUN: {run_name}): มี checkpoint อยู่แล้วที่ {save_path} - ไม่เทรนซ้ำ")
            history_dict[m_name] = []  # ไม่มี history เก่าให้พล็อต แต่โมเดลใช้ได้ปกติ
            continue

        print(f"\n Training: {m_name}  (RUN: {run_name})")
        model = build_model(m_name, pos_weights_tensor, freeze_new)

        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.1, patience=2, min_lr=1e-6, verbose=1),
            callbacks.ModelCheckpoint(save_path, monitor='val_auc', save_best_only=True, mode='max', verbose=1)
        ]
        history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=callbacks_list)
        history_dict[m_name] = history.history['val_auc']

        del model
        tf.keras.backend.clear_session()
        print(f" Finished: {m_name}")

    # กราฟ training curve
    plt.figure(figsize=(12, 8))
    for m_name, scores in history_dict.items():
        plt.plot(scores, label=m_name, linewidth=2.5, marker='o')
    plt.title(f'V7 Validation AUC - {run_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Validation AUC')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(graph_save_dir, 'training_comparison_v7.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Evaluation รายโมเดล + ensemble
    y_true = val_generator.labels[:val_generator.samples]
    individual_preds = {}
    for path in saved_model_paths:
        m_name = os.path.basename(path).replace('best_', '').replace('_v7.keras', '')
        model = tf.keras.models.load_model(path, compile=False)
        preds = model.predict(val_generator, steps=len(val_generator), verbose=1)
        individual_preds[m_name] = preds[:val_generator.samples]
        del model
        tf.keras.backend.clear_session()

    ensemble_preds = sum(individual_preds.values()) / len(individual_preds)
    all_results = dict(individual_preds)
    all_results['Ensemble'] = ensemble_preds

    macro_auc_summary = {}
    for name, preds in all_results.items():
        macro_auc_summary[name] = evaluate_and_save(name, preds, y_true, graph_save_dir, run_name)

    summary_text = f"V7 MACRO AUC SUMMARY - RUN: {run_name}:\n" + "="*50 + "\n"
    for name, macro_auc in macro_auc_summary.items():
        summary_text += f"   {name:15} : {macro_auc:.4f}\n"
    with open(os.path.join(graph_save_dir, 'V7_Macro_AUC_Summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print(f" เสร็จสิ้น RUN: {run_name} -> {graph_save_dir}")

    # เคลียร์ generator ทิ้งก่อนไป version ถัดไป
    del train_generator, val_generator
    tf.keras.backend.clear_session()

    return macro_auc_summary


# ============================================================================
# MAIN: รันทุก version ทีละตัวในไฟล์เดียว
# ============================================================================
if __name__ == "__main__":
    all_version_results = {}

    for cfg in VERSIONS:
        try:
            result = run_version(cfg)
            all_version_results[cfg['run_name']] = result
        except Exception as e:
            print(f"\n!!! ERROR ใน version {cfg['run_name']}: {e}")
            all_version_results[cfg['run_name']] = f"FAILED: {e}"
            continue

    print("\n" + "=" * 60)
    print("สรุปผลทุก version")
    print("=" * 60)
    for run_name, result in all_version_results.items():
        print(f"\n--- {run_name} ---")
        print(result)