import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# ================= ⚙️ 1. CONFIGURATION =================
CSV_PATH = '../../data/external_test/pneumonia_AP_report.csv'
IMG_DIR = '../../data/external_test/pneumonia_ap_images'
MODELS_DIR = '../../models_v7'

MODEL_PATH = os.path.join(MODELS_DIR, 'best_DenseNet121_v7.keras')
TARGET_INDEX = 6  # Pneumonia

# ================= 🔧 2. ฟังก์ชัน Grad-CAM =================
def get_img_array(img_path, size=(384, 384)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Matplotlib ใช้ RGB
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0

def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    img_tensor = np.expand_dims(img_array, axis=0)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    
    # แปลงเป็น float32 (แก้ปัญหา Mixed Precision)
    last_conv_layer_output = tf.cast(last_conv_layer_output, tf.float32)
    pooled_grads = tf.cast(pooled_grads, tf.float32)

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0.0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy()

def apply_clahe(img_normalized):
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    return clahe_rgb.astype(np.float32) / 255.0

def superimpose_heatmap_rgb(img, heatmap, alpha=0.4):
    # ปรับขนาด Heatmap ให้เท่ากับภาพจริง
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # ลงสี Heatmap (OpenCV คายออกมาเป็น BGR)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # แปลงกลับเป็น RGB ให้ Matplotlib
    
    img_uint8 = np.uint8(255 * img)
    superimposed_img = cv2.addWeighted(heatmap_rgb, alpha, img_uint8, 1 - alpha, 0)
    return superimposed_img

# ================= 🖼️ 3. คลาส Viewer สไตล์ Matplotlib =================
class HeatmapViewer:
    def __init__(self, df, model, grad_model):
        self.df = df
        self.model = model
        self.grad_model = grad_model
        self.total_images = len(df)
        self.current_idx = 0
        
        # สร้างหน้าต่าง Matplotlib 2x2
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("V7 Grad-CAM Compare Viewer")
        
        # ผูกปุ่มคีย์บอร์ด
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print("\n" + "="*40)
        print("🎮 GRAD-CAM VIEWER CONTROLS:")
        print("👉 กดปุ่ม 'right' เพื่อดูรูปถัดไป")
        print("👈 กดปุ่ม 'left' เพื่อดูรูปก่อนหน้า")
        print("💾 กดปุ่ม 's' เพื่อเซฟรูปหน้าจอปัจจุบันเก็บไว้")
        print("❌ กดปุ่ม 'q' หรือปิดหน้าต่างเพื่อออก")
        print("="*40)

        # โหลดรูปแรก
        self.update_plot()
        plt.show()

    def update_plot(self):
        row = self.df.iloc[self.current_idx]
        img_name = row['ImageID']
        img_path = os.path.join(IMG_DIR, img_name)
        
        if not os.path.exists(img_path):
            print(f"❌ ไม่พบรูปภาพ: {img_path}")
            return

        print(f"กำลังประมวลผลรูปที่ {self.current_idx+1}/{self.total_images} ...", end='\r')

        # --- คำนวณ Base ---
        img_base = get_img_array(img_path)
        pred_base = self.model.predict(np.expand_dims(img_base, axis=0), verbose=0)[0][TARGET_INDEX]
        cam_base = make_gradcam_heatmap(img_base, self.grad_model, TARGET_INDEX)
        res_base = superimpose_heatmap_rgb(img_base, cam_base)
        
        # --- คำนวณ CLAHE ---
        img_clahe = apply_clahe(img_base)
        pred_clahe = self.model.predict(np.expand_dims(img_clahe, axis=0), verbose=0)[0][TARGET_INDEX]
        cam_clahe = make_gradcam_heatmap(img_clahe, self.grad_model, TARGET_INDEX)
        res_clahe = superimpose_heatmap_rgb(img_clahe, cam_clahe)

        # --- พลอตกราฟเคลียร์ของเก่าก่อน ---
        for ax in self.axs.flat:
            ax.clear()
            ax.axis('off')

        # ซ้ายบน: ภาพดิบ
        self.axs[0, 0].imshow(img_base)
        self.axs[0, 0].set_title(f"RAW Image | Score: {pred_base*100:.1f}%", fontweight='bold')
        
        # ขวาบน: Heatmap ภาพดิบ
        self.axs[0, 1].imshow(res_base)
        self.axs[0, 1].set_title("RAW Grad-CAM", fontweight='bold', color='blue')

        # ซ้ายล่าง: ภาพ CLAHE
        self.axs[1, 0].imshow(img_clahe)
        self.axs[1, 0].set_title(f"CLAHE Image | Score: {pred_clahe*100:.1f}%", fontweight='bold')

        # ขวาล่าง: Heatmap ภาพ CLAHE
        self.axs[1, 1].imshow(res_clahe)
        self.axs[1, 1].set_title("CLAHE Grad-CAM", fontweight='bold', color='red')

        self.fig.suptitle(f"Image {self.current_idx+1}/{self.total_images} : {img_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.current_idx = (self.current_idx + 1) % self.total_images
            self.update_plot()
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % self.total_images
            self.update_plot()
        elif event.key == 's':
            save_path = f"heatmap_evidence_{self.current_idx+1}.png"
            self.fig.savefig(save_path, dpi=150)
            print(f"\n✅ บันทึกรูป {save_path} แล้ว! (เก็บไว้โชว์ตอนพรีเซนต์)")
        elif event.key == 'q':
            plt.close()

# ================= 🚀 4. เริ่มทำงาน =================
if __name__ == "__main__":
    print("⏳ Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    last_conv_layer_name = None
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                last_conv_layer_name = layer.name
                break
        except:
            pass

    print(f"✅ ใช้ Convolutional Layer ชั้นสุดท้าย: {last_conv_layer_name}")

    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    print("⏳ Loading Data...")
    df = pd.read_csv(CSV_PATH)
    df = df[df['Labels'].str.contains('pneumonia', case=False, na=False)].reset_index(drop=True)

    # เรียกใช้งาน Viewer
    viewer = HeatmapViewer(df, model, grad_model)