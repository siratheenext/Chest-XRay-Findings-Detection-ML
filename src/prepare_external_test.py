# import pandas as pd
# import os
# import shutil
# import ast
# from pathlib import Path

# # ================= ⚙️ CONFIGURATION =================
# # 1. ไฟล์และโฟลเดอร์ต้นทาง
# PADCHEST_CSV = '../data/csv/PadChest/PADCHEST_chest_x_ray_images_labels.csv' 
# PADCHEST_IMG_DIR = '../data/Padchest' # 🚩 สังเกตตัวพิมพ์เล็กใหญ่ให้ดีนะครับ

# # 2. โฟลเดอร์ปลายทาง
# OUTPUT_BASE_DIR = '../data/external_tests/Padchest'
# os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# SAMPLES_PER_CLASS = 100
# PROJECTIONS = {
#     'AP': '100_padchest_AP_chest',
#     'AP_horizontal': '100_padchest_AP_horizontal_chest',
#     'L': '100_padchest_L_chest',
#     'PA': '100_padchest_PA_chest'
# }

# CLASSES = [
#     'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
#     'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
#     'Pleural_Thickening', 'Hernia'
# ]

# LABEL_MAPPING = {
#     'Atelectasis': ['atelectasis', 'lobar atelectasis', 'segmental atelectasis', 'total atelectasis'],
#     'Cardiomegaly': ['cardiomegaly'],
#     'Effusion': ['pleural effusion', 'loculated pleural effusion'],
#     'Infiltration': ['infiltrates', 'interstitial pattern', 'alveolar pattern', 'reticular interstitial pattern', 'reticulonodular interstitial pattern', 'ground glass pattern'],
#     'Mass': ['mass'],
#     'Nodule': ['nodule', 'multiple nodules', 'calcified nodule', 'pseudonodule'],
#     'Pneumonia': ['pneumonia', 'atypical pneumonia'],
#     'Pneumothorax': ['pneumothorax', 'hydropneumothorax'],
#     'Consolidation': ['consolidation'],
#     'Edema': ['pulmonary edema'],
#     'Emphysema': ['emphysema'],
#     'Fibrosis': ['pulmonary fibrosis'],
#     'Pleural_Thickening': ['pleural thickening', 'apical pleural thickening'],
#     'Hernia': ['hiatal hernia', 'hernia']
# }
# # ====================================================

# # 🔍 ฟังก์ชันช่วยหาไฟล์แบบฉลาด (หาทุกซอกทุกมุม)
# def find_image_path(base_dir, img_dir_num, filename):
#     # 1. ลองหาตามพาทมาตรฐานก่อน (เร็วที่สุด)
#     std_path = os.path.join(base_dir, str(int(img_dir_num)), filename)
#     if os.path.exists(std_path):
#         return std_path
    
#     # 2. ลองหาในโฟลเดอร์ images (ถ้ามี)
#     alt_path = os.path.join(base_dir, 'images', str(int(img_dir_num)), filename)
#     if os.path.exists(alt_path):
#         return alt_path
        
#     # 3. ถ้าไม่เจอจริงๆ ให้ลองค้นหาแบบ Recursive (ช้าหน่อยแต่ชัวร์)
#     # มันจะควานหาไฟล์ชื่อนี้ในทุกโฟลเดอร์ย่อยของ Padchest
#     for path in Path(base_dir).rglob(filename):
#         return str(path)
        
#     return None

# print(f"⏳ กำลังตรวจสอบโฟลเดอร์ต้นทาง: {os.path.abspath(PADCHEST_IMG_DIR)}")
# if not os.path.exists(PADCHEST_IMG_DIR):
#     print(f"❌ ERROR: ไม่พบโฟลเดอร์ {PADCHEST_IMG_DIR} กรุณาเช็คตัวพิมพ์เล็กใหญ่!")
#     exit()

# print("⏳ กำลังโหลดไฟล์ CSV...")
# df = pd.read_csv(PADCHEST_CSV, low_memory=False)

# for cls in CLASSES: df[cls] = 0
# df['Filename'] = df['ImageID']

# print("🧠 ทำ Auto-Mapping โรค...")
# for index, row in df.iterrows():
#     labels_str = row['Labels']
#     if pd.isna(labels_str): continue
#     try:
#         actual_labels = ast.literal_eval(labels_str)
#         cleaned_labels = [str(l).strip().lower() for l in actual_labels]
#         for v7_class, padchest_terms in LABEL_MAPPING.items():
#             if any(term in cleaned_labels for term in padchest_terms):
#                 df.at[index, v7_class] = 1
#     except: pass

# print("\n🎯 เริ่มกระบวนการคัดแยก...")

# for proj_key, folder_name in PROJECTIONS.items():
#     print(f"\n" + "="*40)
#     print(f"📸 จัดการหมวดหมู่: {proj_key} -> {folder_name}")
    
#     out_img_dir = os.path.join(OUTPUT_BASE_DIR, folder_name)
#     out_csv_path = os.path.join(OUTPUT_BASE_DIR, f'labels_{folder_name}.csv')
#     os.makedirs(out_img_dir, exist_ok=True)
    
#     df_proj = df[df['Projection'] == proj_key].copy()
#     sampled_rows = []
#     selected_filenames = set()
    
#     for disease in CLASSES:
#         disease_df = df_proj[df_proj[disease] == 1]
#         disease_df = disease_df[~disease_df['Filename'].isin(selected_filenames)]
#         if len(disease_df) == 0: continue
            
#         sample_size = min(SAMPLES_PER_CLASS, len(disease_df))
#         sampled_df = disease_df.sample(n=sample_size, random_state=42)
#         selected_filenames.update(sampled_df['Filename'].tolist())
#         sampled_rows.append(sampled_df)
#         print(f"  ✅ {disease:20}: พบ {sample_size} รายการ")
        
#     if not sampled_rows: continue
        
#     final_proj_df = pd.concat(sampled_rows, ignore_index=True)
#     final_proj_df = final_proj_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
#     print(f"⏳ กำลังตรวจสอบและคัดลอกไฟล์จริง...")
#     success_count = 0
#     first_attempt = True
    
#     for _, r in final_proj_df.iterrows():
#         src_path = find_image_path(PADCHEST_IMG_DIR, r['ImageDir'], r['Filename'])
        
#         if src_path:
#             shutil.copy2(src_path, os.path.join(out_img_dir, r['Filename']))
#             success_count += 1
#         elif first_attempt:
#             # พ่นตัวอย่างพาทที่มันหาไม่เจอออกมาดูแค่ไฟล์แรก
#             print(f"  ❌ หาไม่เจอตัวอย่าง: {r['Filename']} (ImageDir: {r['ImageDir']})")
#             first_attempt = False

#     columns_to_keep = ['Filename', 'Projection'] + CLASSES
#     final_proj_df[columns_to_keep].to_csv(out_csv_path, index=False)
#     print(f"🎉 เสร็จสิ้น! คัดลอกสำเร็จ {success_count} ภาพ")

# print("\n🚀 ภารกิจเสร็จสมบูรณ์!")

import pandas as pd
import os
import shutil
import ast

# ================= ⚙️ CONFIGURATION =================
# 1. ไฟล์และโฟลเดอร์ต้นทาง (Dataset ใหม่ PadChest)
PADCHEST_CSV = '../data/csv/PadChest/PADCHEST_chest_x_ray_images_labels.csv' 
PADCHEST_IMG_DIR = '../data/Padchest' # โฟลเดอร์ที่มีเลข 0-54 อยู่ข้างใน

# 2. โฟลเดอร์ปลายทาง (ที่จะเก็บ 100 รูปต่อโรคไว้เทส)
OUTPUT_BASE_DIR = '../data/external_tests/Padchest'
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

SAMPLES_PER_CLASS = 100
PROJECTIONS = ['AP', 'AP_horizontal', 'L', 'PA']
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# 🧠 พจนานุกรมแปลภาษาโรค (PadChest -> Model V7)
LABEL_MAPPING = {
    'Atelectasis': ['atelectasis'], 'Cardiomegaly': ['cardiomegaly'], 'Effusion': ['pleural effusion'],
    'Infiltration': ['infiltrates'], 'Mass': ['mass'], 'Nodule': ['nodule'], 'Pneumonia': ['pneumonia'],
    'Pneumothorax': ['pneumothorax'], 'Consolidation': ['consolidation'], 'Edema': ['pulmonary edema'],
    'Emphysema': ['emphysema'], 'Fibrosis': ['pulmonary fibrosis'], 'Pleural_Thickening': ['pleural thickening'], 'Hernia': ['hernia']
}
# ====================================================

print("⏳ กำลังโหลดไฟล์ CSV และวิเคราะห์ข้อมูล...")
df = pd.read_csv(PADCHEST_CSV, low_memory=False)

# Mapping โรค
for cls in CLASSES: df[cls] = 0
for index, row in df.iterrows():
    labels_str = str(row['Labels']).lower()
    for v7_class, terms in LABEL_MAPPING.items():
        if any(term in labels_str for term in terms):
            df.at[index, v7_class] = 1

# 🌟 ฟังก์ชันหาไฟล์ภาพในลิ้นชัก 0-54 🌟
def get_path(r):
    # ลองหาแบบ ../data/Padchest/เลขโฟลเดอร์/ชื่อไฟล์.png
    p = os.path.join(PADCHEST_IMG_DIR, str(int(r['ImageDir'])), r['ImageID'])
    return p if os.path.exists(p) else None

print("🔍 กำลังสแกนหาไฟล์ภาพที่มีอยู่จริงในเครื่องคุณ (อาจใช้เวลาครู่หนึ่ง)...")
df['ActualPath'] = df.apply(get_path, axis=1)
df_valid = df[df['ActualPath'].notna()].copy()

print(f"📊 ตรวจพบไฟล์ภาพที่ใช้งานได้จริง {len(df_valid)} รูป จากทั้งหมดใน CSV")

for proj in PROJECTIONS:
    folder_name = f"100_padchest_{proj}_chest"
    print(f"\n📸 จัดการมุมกล้อง: {proj} -> {folder_name}")
    
    out_img_dir = os.path.join(OUTPUT_BASE_DIR, folder_name)
    os.makedirs(out_img_dir, exist_ok=True)
    
    df_proj = df_valid[df_valid['Projection'] == proj].copy()
    sampled_rows = []
    selected_files = set()
    
    for disease in CLASSES:
        disease_df = df_proj[(df_proj[disease] == 1) & (~df_proj['ImageID'].isin(selected_files))]
        if len(disease_df) == 0: continue
            
        sample_size = min(SAMPLES_PER_CLASS, len(disease_df))
        sampled = disease_df.sample(n=sample_size, random_state=42)
        selected_files.update(sampled['ImageID'].tolist())
        sampled_rows.append(sampled)
        print(f"  ✅ {disease:20}: สุ่มได้ {sample_size} รูป")
        
    if sampled_rows:
        final_df = pd.concat(sampled_rows).sample(frac=1, random_state=42)
        success = 0
        for _, r in final_df.iterrows():
            shutil.copy2(r['ActualPath'], os.path.join(out_img_dir, r['ImageID']))
            success += 1
            
        final_df[['ImageID', 'Projection'] + CLASSES].rename(columns={'ImageID':'Filename'}).to_csv(
            os.path.join(OUTPUT_BASE_DIR, f'labels_{folder_name}.csv'), index=False
        )
        print(f"🎉 สำเร็จ! คัดลอกรูปภาพไปไว้เทสแล้ว {success} รูป")

print("\n🚀 เสร็จสมบูรณ์! ตอนนี้คุณมีรูปใหม่พร้อมเทสโมเดลแล้วครับ")