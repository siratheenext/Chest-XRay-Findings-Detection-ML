import pandas as pd
import os
import shutil

# ================= ⚙️ CONFIGURATION =================
# 1. ไฟล์ต้นทาง
MASTER_CSV = '../data/csv/Data_Entry_2017.csv' # ไฟล์ใหญ่แสนรูป
TRAIN_CSV = '../data/csv/train_labels_14_hybrid.csv' # ไฟล์ที่คุณใช้ Train ไปแล้ว
SOURCE_IMG_DIR = '../data/all_images_raw' # ถังรูปใหญ่

# 2. ไฟล์ปลายทาง (ชุด Test ใหม่จากภาพเก่า)
TEST_CSV_OUTPUT = '../data/csv/test_labels_internal.csv'
TEST_IMG_DIR = '../data/test_images_internal'
os.makedirs(TEST_IMG_DIR, exist_ok=True)

# จำนวนภาพที่อยากได้ในชุด Test (เช่น โรคละ 100 ภาพ)
SAMPLES_PER_CLASS = 100

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]
# ====================================================

print("⏳ กำลังโหลดข้อมูล...")
df_master = pd.read_csv(MASTER_CSV)
df_train = pd.read_csv(TRAIN_CSV)

# 1. คัดชื่อไฟล์ที่ถูกใช้ Train ออกไปก่อน (Unseen Images Only)
train_filenames = set(df_train['Filename'].tolist())
df_unseen = df_master[~df_master['Image Index'].isin(train_filenames)].copy()

print(f"📊 พบภาพที่โมเดลไม่เคยเห็นทั้งหมด {len(df_unseen)} ภาพ")

final_test_data = []
selected_filenames = set()

print(f"🎯 เริ่มสุ่มภาพโรคละ {SAMPLES_PER_CLASS} ภาพ...")

for disease in CLASSES:
    # กรองภาพที่มีโรคนั้นๆ (ในไฟล์ NIH ใช้เครื่องหมาย | แยกโรค)
    disease_df = df_unseen[df_unseen['Finding Labels'].str.contains(disease)]
    # ตัดภาพที่ซ้ำกับโรคอื่นที่สุ่มไปแล้ว
    disease_df = disease_df[~disease_df['Image Index'].isin(selected_filenames)]
    
    if len(disease_df) == 0:
        continue
        
    sample_size = min(SAMPLES_PER_CLASS, len(disease_df))
    sampled_df = disease_df.sample(n=sample_size, random_state=42)
    
    for _, row in sampled_df.iterrows():
        img_name = row['Image Index']
        src = os.path.join(SOURCE_IMG_DIR, img_name)
        dst = os.path.join(TEST_IMG_DIR, img_name)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            
            # เตรียมข้อมูลเข้า CSV ใหม่ (Format เดียวกับ Train)
            entry = {'Filename': img_name}
            for d in CLASSES:
                entry[d] = 1 if d in row['Finding Labels'] else 0
            
            final_test_data.append(entry)
            selected_filenames.add(img_name)
    
    print(f"  ✅ {disease:20}: สุ่มมา {sample_size} ภาพ")

# 2. บันทึกเป็น CSV สำหรับ Viewer
pd.DataFrame(final_test_data).to_csv(TEST_CSV_OUTPUT, index=False)
print(f"\n🎉 สร้างชุดทดสอบสำเร็จ!")
print(f"📂 ไฟล์ CSV: {TEST_CSV_OUTPUT}")
print(f"🖼️ โฟลเดอร์รูป: {TEST_IMG_DIR}")