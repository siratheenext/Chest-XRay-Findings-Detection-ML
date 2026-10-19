import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tkinter import filedialog, Tk

# --- ✅ Config ---
# 📁 OLD PATHS (Commented Out)
# MODEL_PATH = '../models/best_DenseNet121.keras'  
# MODEL_PATH = '../models/best_ResNet50.keras'  
# IMG_SIZE = (360, 360) 

# 🚀 NEW PATH (3000PIC Version)
MODEL_PATH = '../models_v2_3000PIC/best_ResNet50_v2_3000PIC.keras'
IMG_SIZE = (384, 384) 

TEST_DIR = '../data/test_images'  
CLASSES = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Emphysema', 'Fibrosis']

# Mixed Precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
# ------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy().astype(np.float32)

class ChestXrayViewer:
    def __init__(self):
        # 1. โหลดโมเดล
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Error: ไม่เจอไฟล์โมเดลที่ {MODEL_PATH}")
            # Fallback check for old path structure just in case
            if os.path.exists('../models/best_ResNet50.keras'):
                print("💡 เจอโมเดลเก่าแทน... แนะนำให้เช็ค Path ใหม่นะครับ")
            exit()
        
        print(f"⏳ Loading Model: {os.path.basename(MODEL_PATH)}... (Please wait)")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        
        # หา Layer สุดท้าย
        self.last_conv_layer_name = 'conv5_block16_2_conv' # Default for DenseNet
        
        # Auto-detect layer name based on model type
        if 'resnet' in MODEL_PATH.lower():
            self.last_conv_layer_name = 'post_relu' # Default for ResNet50V2
        elif 'mobilenet' in MODEL_PATH.lower():
            self.last_conv_layer_name = 'Out_relu' # Default for MobileNetV2

        # Fallback Auto-search
        try:
            self.model.get_layer(self.last_conv_layer_name)
        except:
            print(f"⚠️ Layer '{self.last_conv_layer_name}' not found. Searching automatically...")
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name or 'relu' in layer.name:
                    if len(layer.output.shape) == 4:
                        self.last_conv_layer_name = layer.name
                        break
                        
        print(f"✅ Model Loaded! (Heatmap Layer: {self.last_conv_layer_name})")

        # 2. โหลดรายชื่อรูปภาพ
        if os.path.exists(TEST_DIR):
            self.image_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.mode = 'folder'
        else:
            self.image_files = []
            self.mode = 'manual'
            print("⚠️ ไม่เจอโฟลเดอร์ Test จะใช้โหมดเลือกไฟล์เอง")

        self.current_index = 0
        self.total_images = len(self.image_files)
        
        # GUI Setup
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(16, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        if self.mode == 'folder' and self.total_images > 0:
            self.update_view()
            plt.show()
        else:
            self.select_manual_file()

    def load_and_prep_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        return img

    def apply_smooth_heatmap(self, original_img, heatmap):
        # 1. Resize Heatmap
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # 2. Thresholding (ตัด Noise < 40% ทิ้ง)
        heatmap = np.where(heatmap < 0.40, 0, heatmap)
        
        # 3. Gaussian Blur
        if np.max(heatmap) > 0:
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # 4. Color Mapping
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # 5. Smart Blending
        alpha = heatmap 
        output = original_img.copy()
        
        for c in range(3):
            output[:, :, c] = (heatmap_colored[:, :, c] * alpha * 0.7) + (original_img[:, :, c] * (1 - (alpha * 0.7)))
            
        return output.astype(np.uint8)

    def process_and_display(self, img_path, img_name):
        original_img = self.load_and_prep_image(img_path)
        if original_img is None: return

        # Predict
        img_array = np.expand_dims(original_img / 255.0, axis=0)
        preds = self.model.predict(img_array, verbose=0)[0]
        top_idx = np.argmax(preds)
        top_class = CLASSES[top_idx]
        confidence = preds[top_idx]

        # Generate Raw Heatmap
        try:
            heatmap_raw = make_gradcam_heatmap(img_array, self.model, self.last_conv_layer_name, top_idx)
            # Apply Smooth Look
            final_img = self.apply_smooth_heatmap(original_img, heatmap_raw)
        except Exception as e:
            print(f"Heatmap Error: {e}")
            final_img = original_img

        # Clear Axes
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear()

        # 1. Original
        self.ax1.imshow(original_img)
        self.ax1.set_title(f"Original: {img_name}")
        self.ax1.axis('off')

        # 2. Smooth Heatmap
        self.ax2.imshow(final_img)
        self.ax2.set_title(f"AI Focus: {top_class}")
        self.ax2.axis('off')

        # 3. Bar Chart
        y_pos = np.arange(len(CLASSES))
        colors = ['red' if score > 0.5 else 'green' for score in preds]
        self.ax3.barh(y_pos, preds, color=colors)
        self.ax3.set_yticks(y_pos)
        self.ax3.set_yticklabels(CLASSES)
        self.ax3.set_xlim(0, 1.1)
        self.ax3.set_title("Confidence Scores")
        self.ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        for i, v in enumerate(preds):
            self.ax3.text(v + 0.02, i, f"{v*100:.1f}%", va='center', fontweight='bold')

        plt.figtext(0.5, 0.02, 
                    f"Controls: [Enter/Right] Next | [Backspace/Left] Previous | [Q] Quit | Image {self.current_index + 1}/{self.total_images}", 
                    ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

        self.fig.canvas.draw()

    def update_view(self):
        img_name = self.image_files[self.current_index]
        img_path = os.path.join(TEST_DIR, img_name)
        self.process_and_display(img_path, img_name)

    def select_manual_file(self):
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select X-ray Image", filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.process_and_display(file_path, os.path.basename(file_path))
            plt.show()

    def on_key(self, event):
        if self.mode == 'folder':
            if event.key in ['right', 'enter'] and self.current_index < self.total_images - 1:
                self.current_index += 1
                self.update_view()
            elif event.key in ['left', 'backspace'] and self.current_index > 0:
                self.current_index -= 1
                self.update_view()
            elif event.key == 'q':
                plt.close()

if __name__ == "__main__":
    ChestXrayViewer()