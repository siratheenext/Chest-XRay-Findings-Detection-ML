import pandas as pd
import ast
import os

# ================= ⚙️ CONFIG =================
INPUT_CSV = '../data/csv/PadChest/PADCHEST_chest_x_ray_images_labels.csv'
OUTPUT_CSV = '../data/csv/padchest_test_v7_format.csv' # ไฟล์ใหม่ที่หน้าตาเหมือน Train CSV

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# พจนานุกรมแปลคำศัพท์ (Mapping)
LABEL_MAPPING = {
    'Atelectasis': ['atelectasis'], 'Cardiomegaly': ['cardiomegaly'], 'Effusion': ['pleural effusion'],
    'Infiltration': ['infiltrates'], 'Mass': ['mass'], 'Nodule': ['nodule'], 'Pneumonia': ['pneumonia'],
    'Pneumothorax': ['pneumothorax'], 'Consolidation': ['consolidation'], 'Edema': ['pulmonary edema'],
    'Emphysema': ['emphysema'], 'Fibrosis': ['pulmonary fibrosis'], 'Pleural_Thickening': ['pleural thickening'], 'Hernia': ['hernia']
}
# =============================================

print("⏳ กำลังแปลงไฟล์ PadChest CSV ให้เป็นรูปแบบ V7...")
df = pd.read_csv(INPUT_CSV, low_memory=False)

# 1. สร้างคอลัมน์ใหม่ Filename (รวม ImageDir/ImageID เพื่อให้ Viewer หาเจอ)
# ผลลัพธ์จะเป็น "0/image.png", "54/image.png"
df['Filename'] = df.apply(lambda r: f"{int(r['ImageDir'])}/{r['ImageID']}", axis=1)

# 2. ทำ One-hot Encoding 14 โรค
for cls in CLASSES:
    df[cls] = 0

print("🧠 กำลัง Mapping รายชื่อโรค...")
for index, row in df.iterrows():
    labels_str = str(row['Labels']).lower()
    for v7_class, terms in LABEL_MAPPING.items():
        if any(term in labels_str for term in terms):
            df.at[index, v7_class] = 1

# 3. เลือกเฉพาะคอลัมน์ที่ Viewer ต้องการ
final_df = df[['Filename'] + CLASSES]

# 4. (Optional) กรองเอาเฉพาะแถวที่มีอย่างน้อย 1 โรค (ไม่เอาภาพ Normal)
final_df = final_df[final_df[CLASSES].sum(axis=1) > 0]

# 5. บันทึกไฟล์
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"🎉 เสร็จแล้ว! ไฟล์อยู่ที่: {OUTPUT_CSV}")
print(f"📊 จำนวนแถวทั้งหมดที่แปลงได้: {len(final_df)} รายการ")