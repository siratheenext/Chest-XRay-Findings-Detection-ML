import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from tensorflow.keras import mixed_precision

# --- SETUP: Mixed Precision ---
mixed_precision.set_global_policy('mixed_float16')

# =========================================================================
# CONFIG: เลือก Model และ Dataset ที่จะดู แค่เปลี่ยน 2 ตัวแปรนี้พอ
# =========================================================================
MODEL_KEY = 'Mixed_AP_PA'     # เปลี่ยนเป็น 'PA_only' เพื่อดูอีกโมเดล
DATASET_KEY = 'Padchest_PA'   # เปลี่ยนเป็น NIH_test / Padchest_AP / Padchest_AP_horizontal / Padchest_L

# --- โมเดลที่มีให้เลือก --- 
MODEL_SETS = {
    'Mixed_AP_PA': '../models_v7',
    'PA_only':     '../models_v7_PA_only',
}

# --- Dataset ที่มีให้เลือก (csv_path, test_dir) ---
DATASET_SETS = {
    'NIH_test':              ('../data/csv/test_labels_14_hybrid.csv', '../data/all_images_raw'),
    'Padchest_PA':            ('../data/csv/padchest_test_v7_format.csv', '../data/Padchest/100_padchest_PA_chest'),
    'Padchest_AP':            ('../data/csv/padchest_test_v7_format.csv', '../data/Padchest/100_padchest_AP_chest'),
    'Padchest_AP_horizontal': ('../data/csv/padchest_test_v7_format.csv', '../data/Padchest/100_padchest_AP_horizontal_chest'),
    'Padchest_L':              ('../data/csv/padchest_test_v7_format.csv', '../data/Padchest/100_padchest_L_chest'),
}

MODELS_DIR = MODEL_SETS[MODEL_KEY]
CSV_PATH, TEST_DIR = DATASET_SETS[DATASET_KEY]

# ขนาดภาพ 384 ตามโครงสร้าง DenseNet121 V7
IMG_SIZE = 384

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# Optimal Thresholds (คำนวณจาก Youden's Index ตอน evaluate)
THRESHOLDS = {
    'Atelectasis': 0.4353,
    'Cardiomegaly': 0.4698,
    'Effusion': 0.5165,
    'Infiltration': 0.4444,
    'Mass': 0.4581,
    'Nodule': 0.4176,
    'Pneumonia': 0.4217,
    'Pneumothorax': 0.5005,
    'Consolidation': 0.4546,
    'Edema': 0.4351,
    'Emphysema': 0.4005,
    'Fibrosis': 0.3912,
    'Pleural_Thickening': 0.4802,
    'Hernia': 0.2657
}
# =========================================================================


class ThesisViewer:
    def __init__(self, test_dir, csv_path, model_dir):
        self.test_dir = test_dir
        self.csv_path = csv_path
        self.model_dir = model_dir

        print(f"Model: {MODEL_KEY} ({model_dir})")
        print(f"Dataset: {DATASET_KEY} ({test_dir})")

        print("Loading CSV Ground Truth...")
        self.df_labels = pd.read_csv(csv_path)
        self.df_labels['Filename'] = self.df_labels['Filename'].str.strip()

        # เอาแค่ basename ของ Filename ในกรณีที่ CSV มี prefix เช่น '0/xxx.png' (Padchest)
        self.df_labels['basename'] = self.df_labels['Filename'].apply(lambda x: os.path.basename(x))

        # หาไฟล์รูปที่มีอยู่จริงในโฟลเดอร์ แล้วกรองเอาเฉพาะที่มี label ด้วย (สำคัญมาก
        # กันกรณีโฟลเดอร์มีรูปเป็นแสนไฟล์ เช่น all_images_raw จะได้ไม่ต้อง list ทุกไฟล์)
        files_in_dir = set(f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        labeled_basenames = set(self.df_labels['basename'].tolist())
        self.all_images = sorted(files_in_dir & labeled_basenames)

        self.total_images = len(self.all_images)
        print(f"Found {self.total_images} images with matching labels.")

        if self.total_images == 0:
            raise ValueError(
                "ไม่มีภาพที่ match กันระหว่าง CSV กับโฟลเดอร์เลย "
                "เช็ค path ของ CSV_PATH / TEST_DIR อีกครั้ง"
            )

        self.current_index = 0
        self.verdict_text_obj = None
        self.models = self.load_v7_models()
        self.last_conv_layer_names = {
            'DenseNet121': 'conv5_block16_concat',
            'ResNet50V2': 'post_relu',
            'MobileNetV2': 'out_relu'
        }

        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 7), dpi=100)
        self.fig.canvas.manager.set_window_title(f'Viewer - {MODEL_KEY} on {DATASET_KEY}')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def load_v7_models(self):
        configs = [('best_DenseNet121_v7.keras', 0.5, 'DenseNet121'),
                   ('best_ResNet50V2_v7.keras', 0.3, 'ResNet50V2'),
                   ('best_MobileNetV2_v7.keras', 0.2, 'MobileNetV2')]
        loaded = []
        for f, w, t in configs:
            p = os.path.join(self.model_dir, f)
            if os.path.exists(p):
                print(f"   - Loading {f}")
                loaded.append((tf.keras.models.load_model(p, compile=False), w, t))
            else:
                print(f"   - SKIP (not found): {p}")
        if not loaded:
            raise ValueError(f"ไม่เจอโมเดลเลยใน {self.model_dir}")
        return loaded

    def generate_spot_heatmap(self, img_batch, ensemble_preds):
        max_idx = np.argmax(ensemble_preds)
        is_disease = ensemble_preds[max_idx] > THRESHOLDS.get(CLASSES[max_idx], 0.5)
        top_disease = CLASSES[max_idx] if is_disease else "Normal"

        ensemble_heatmap = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        for m, w, m_type in self.models:
            layer_name = self.last_conv_layer_names.get(m_type)
            grad_model = tf.keras.models.Model([m.inputs], [m.get_layer(layer_name).output, m.output])

            with tf.GradientTape() as tape:
                inputs = tf.cast(img_batch, tf.float32)
                conv_outs, preds = grad_model(inputs)
                loss = preds[:, max_idx]

            grads = tape.gradient(loss, conv_outs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = conv_outs[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(heatmap, 0).numpy().astype(np.float32)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)

            ensemble_heatmap += cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE)) * w

        ensemble_heatmap = np.maximum(ensemble_heatmap, 0)
        if np.max(ensemble_heatmap) > 0:
            ensemble_heatmap /= np.max(ensemble_heatmap)
            ensemble_heatmap[ensemble_heatmap < 0.5] = 0

        return ensemble_heatmap, top_disease

    def analyze_current_image(self):
        filename = self.all_images[self.current_index]
        img_path = os.path.join(self.test_dir, filename)

        # ค้นหาเฉลย (Actual) จาก basename
        mask = self.df_labels['basename'] == filename
        row = self.df_labels[mask]
        actual_list = [CLASSES[i] for i, v in enumerate(row.iloc[0][CLASSES].values) if v == 1] if not row.empty else []
        actual_text = "Actual: " + (", ".join(actual_list) if actual_list else "Normal")

        orig_img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_batch = np.expand_dims(img_resized / 255.0, axis=0)

        ensemble_preds = np.zeros(len(CLASSES))
        for m, w, _ in self.models:
            ensemble_preds += m.predict(img_batch, verbose=0)[0] * w

        heatmap, top_disease = self.generate_spot_heatmap(img_batch, ensemble_preds)

        h_uint8 = np.uint8(255 * heatmap)
        h_color = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
        h_color_rgb = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)

        mask_overlay = (heatmap > 0).astype(np.float32)[..., np.newaxis]
        overlay_img = (img_resized * (1 - mask_overlay * 0.4) + h_color_rgb * (mask_overlay * 0.4)).astype(np.uint8)

        pred_list = [CLASSES[i] for i, s in enumerate(ensemble_preds) if s > THRESHOLDS.get(CLASSES[i], 0.5)]
        pred_summary = "Prediction: " + (", ".join(pred_list) if pred_list else "Normal")

        # ==== สรุปผลว่าทายถูกหรือผิด (เทียบ Actual vs Prediction) ====
        actual_set = set(actual_list)   # ว่างถ้า Normal
        pred_set = set(pred_list)       # ว่างถ้า Normal

        correct_diseases = actual_set & pred_set          # True Positive: เฉลยมี AI ทายมี
        missed_diseases = actual_set - pred_set            # False Negative: เฉลยมี AI ทายไม่มี (พลาดโรค)
        false_alarm_diseases = pred_set - actual_set        # False Positive: เฉลยไม่มี AI ทายมี (ตื่นตูม)

        if not actual_set and not pred_set:
            verdict = "CORRECT — ถูกต้อง (Normal ทั้งเฉลยและ AI ทาย)"
            verdict_color = 'green'
        elif actual_set == pred_set:
            verdict = f"CORRECT — ถูกต้องครบทุกโรค ({', '.join(sorted(actual_set))})"
            verdict_color = 'green'
        elif correct_diseases and not missed_diseases and not false_alarm_diseases:
            verdict = "CORRECT — ถูกต้องครบทุกโรค"
            verdict_color = 'green'
        else:
            parts = []
            if correct_diseases:
                parts.append(f"ทายถูก: {', '.join(sorted(correct_diseases))}")
            if missed_diseases:
                parts.append(f"พลาดโรค (FN): {', '.join(sorted(missed_diseases))}")
            if false_alarm_diseases:
                parts.append(f"ทายเกินจริง (FP): {', '.join(sorted(false_alarm_diseases))}")
            verdict = "WRONG — " + " | ".join(parts)
            verdict_color = 'red'

        return img_rgb, overlay_img, ensemble_preds, actual_text, pred_summary, filename, top_disease, verdict, verdict_color

    def update_plot(self):
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear()

        # ลบข้อความสรุปผลรอบก่อน (ถ้ามี) ก่อนวาดใหม่ กัน text ซ้อนกันตอนเปลี่ยนภาพ
        if hasattr(self, 'verdict_text_obj') and self.verdict_text_obj is not None:
            self.verdict_text_obj.remove()

        img_rgb, overlay, preds, actual, pred_sum, fname, top_d, verdict, verdict_color = self.analyze_current_image()

        self.ax1.imshow(img_rgb)
        self.ax1.set_title(
            f"[{MODEL_KEY} on {DATASET_KEY}] ({self.current_index+1}/{self.total_images})\n"
            f"Image: {fname}\n{actual}",
            color='blue', fontsize=10, fontweight='bold'
        )
        self.ax1.axis('off')

        self.ax2.imshow(overlay)
        t_color = 'red' if top_d != "Normal" else 'green'
        display_text = f"{pred_sum}\nTOP PREDICT: {top_d}"
        self.ax2.text(0.5, 1.08, display_text, transform=self.ax2.transAxes, ha='center',
                      color=t_color, fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        self.ax2.axis('off')

        y_pos = np.arange(len(CLASSES))
        colors = ['red' if preds[i] > THRESHOLDS.get(CLASSES[i], 0.5) else 'gray' for i in range(len(CLASSES))]
        self.ax3.barh(y_pos, preds, color=colors, edgecolor='black', alpha=0.8)
        self.ax3.set_yticks(y_pos)
        self.ax3.set_yticklabels(CLASSES, fontsize=9, fontweight='bold')
        self.ax3.set_xlim(0, 1.1)

        for i, v in enumerate(preds):
            t = THRESHOLDS.get(CLASSES[i], 0.5)
            self.ax3.text(v + 0.01, i, f"{v*100:.1f}% (T:{t})", va='center', fontsize=7.5)
        self.ax3.set_title("Confidence & Class Thresholds", fontsize=10, fontweight='bold')

        # ==== แถบสรุปผลด้านล่าง: ทายถูก/ผิด (วางกลางล่างของหน้าต่างทั้งหมด) ====
        self.verdict_text_obj = self.fig.text(
            0.5, 0.02, verdict,
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='white',
            bbox=dict(facecolor=verdict_color, alpha=0.85, boxstyle='round,pad=0.6')
        )

        plt.tight_layout(pad=3.0, rect=[0, 0.06, 1, 1])  # เผื่อที่ด้านล่างให้แถบสรุปผล
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key in ['enter', 'right']:
            self.current_index = (self.current_index + 1) % self.total_images
            self.update_plot()
        elif event.key in ['delete', 'left']:
            self.current_index = (self.current_index - 1) % self.total_images
            self.update_plot()
        elif event.key == 'escape':
            plt.close()


if __name__ == "__main__":
    viewer = ThesisViewer(TEST_DIR, CSV_PATH, MODELS_DIR)
    viewer.update_plot()
    plt.show(block=True)