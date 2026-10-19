# import pandas as pd
# import ast
# import os


# #===============(Drop)================
# # 1. ภาพคนไข้ปกติ (Normal / No Finding): ภาพที่ปอดใสสะอาด ไม่มีโรคอะไรเลย
# # 2. ภาพที่มีโรคอื่นนอกเหนือจาก 14 โรคของเรา: เช่น ภาพที่มีกระดูกซี่โครงหัก (Rib fracture), ภาพที่มีเครื่องกระตุ้นหัวใจ, หรือภาพที่มีสิ่งแปลกปลอมแบบในตัวอย่างก่อนหน้านี้ของคุณ 

# # ================= ⚙️ CONFIG =================
# INPUT_CSV = '../data/csv/PadChest/PADCHEST_chest_x_ray_images_labels.csv'
# OUTPUT_CSV = '../data/csv/padchest_test_v7_format.csv'

# CLASSES = [
#     'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
#     'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
#     'Pleural_Thickening', 'Hernia'
# ]

# # 🧠 Keyword สำหรับจับคู่โรค
# LABEL_MAPPING = {
#     'Atelectasis': ['atelectasis', 'lobar atelectasis', 'segmental atelectasis'],
#     'Cardiomegaly': ['cardiomegaly', 'heart enlargement'],
#     'Effusion': ['pleural effusion', 'effusion', 'loculated pleural effusion'],
#     'Infiltration': ['infiltrates', 'interstitial pattern', 'ground glass pattern'],
#     'Mass': ['mass', 'lung mass'],
#     'Nodule': ['nodule', 'pulmonary nodule', 'multiple nodules'],
#     'Pneumonia': ['pneumonia', 'atypical pneumonia'],
#     'Pneumothorax': ['pneumothorax'],
#     'Consolidation': ['consolidation'],
#     'Edema': ['pulmonary edema', 'edema'],
#     'Emphysema': ['emphysema'],
#     'Fibrosis': ['pulmonary fibrosis', 'fibrosis', 'chronic changes'], 
#     'Pleural_Thickening': ['pleural thickening'],
#     'Hernia': ['hiatal hernia', 'hernia']
# }
# # =============================================

# print("⏳ กำลังวิเคราะห์และแมพข้อมูล PadChest...")
# df = pd.read_csv(INPUT_CSV, low_memory=False)

# # สร้างโครงสร้างไฟล์ Filename (Directory/ID)
# df['Filename'] = df.apply(lambda r: f"{int(r['ImageDir'])}/{r['ImageID']}", axis=1)

# def map_labels(row_labels):
#     if pd.isna(row_labels): return [0] * 14
    
#     try:
#         # แปลง Text ให้กลายเป็น List
#         actual_list = ast.literal_eval(row_labels)
        
#         # 🐞 แก้บั๊กตรงนี้ครับ: ใช้ actual_list ที่รับค่ามาถูกต้อง
#         cleaned_list = [str(item).strip().lower() for item in actual_list]
        
#         row_results = []
#         for v7_class in CLASSES:
#             match = 0
#             # อัปเกรดให้ฉลาดขึ้น: หาว่ามี Keyword แฝงอยู่ในคำวินิจฉัยหรือไม่
#             for keyword in LABEL_MAPPING[v7_class]:
#                 if any(keyword in doctor_label for doctor_label in cleaned_list):
#                     match = 1
#                     break # เจอโรคนี้แล้ว ข้ามไปหาโรคต่อไปเลย
#             row_results.append(match)
#         return row_results
#     except Exception as e:
#         return [0] * 14

# # รันการ Mapping
# print("🧠 กำลังจับคู่คำวินิจฉัย (อาจใช้เวลาสักครู่)...")
# results = df['Labels'].apply(map_labels)
# df[CLASSES] = pd.DataFrame(results.tolist(), index=df.index)

# # เก็บเฉพาะไฟล์ที่มีอย่างน้อย 1 โรค
# final_df = df[['Filename'] + CLASSES]
# final_df = final_df[final_df[CLASSES].sum(axis=1) > 0]

# final_df.to_csv(OUTPUT_CSV, index=False)
# print(f"🎉 สำเร็จ! บันทึกไฟล์เฉลยใหม่ที่: {OUTPUT_CSV}")
# print(f"📊 ตรวจพบรอยโรคที่ตรงกับ 14 โรคของเราทั้งหมด: {len(final_df)} รายการ")


import pandas as pd
import ast
import os

# ================= ⚙️ CONFIG =================
INPUT_CSV = '../data/csv/PadChest/PADCHEST_chest_x_ray_images_labels.csv'
OUTPUT_CSV = '../data/csv/padchest_test_v7_format.csv'

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# 🧠 Keyword สำหรับจับคู่โรค (14 โรคหลัก)
LABEL_MAPPING = {
    'Atelectasis': ['atelectasis', 'lobar atelectasis', 'segmental atelectasis'],
    'Cardiomegaly': ['cardiomegaly', 'heart enlargement'],
    'Effusion': ['pleural effusion', 'effusion', 'loculated pleural effusion'],
    'Infiltration': ['infiltrates', 'interstitial pattern', 'ground glass pattern'],
    'Mass': ['mass', 'lung mass'],
    'Nodule': ['nodule', 'pulmonary nodule', 'multiple nodules'],
    'Pneumonia': ['pneumonia', 'atypical pneumonia'],
    'Pneumothorax': ['pneumothorax'],
    'Consolidation': ['consolidation'],
    'Edema': ['pulmonary edema', 'edema'],
    'Emphysema': ['emphysema'],
    'Fibrosis': ['pulmonary fibrosis', 'fibrosis', 'chronic changes'], 
    'Pleural_Thickening': ['pleural thickening'],
    'Hernia': ['hiatal hernia', 'hernia']
}
# =============================================

print("⏳ กำลังวิเคราะห์และแมพข้อมูล PadChest (รวมภาพ Normal)...")
df = pd.read_csv(INPUT_CSV, low_memory=False)

# สร้างโครงสร้างไฟล์ Filename (Directory/ID)
df['Filename'] = df.apply(lambda r: f"{int(r['ImageDir'])}/{r['ImageID']}", axis=1)

def map_labels(row_labels):
    if pd.isna(row_labels): return [0] * 14
    
    try:
        # แปลง Text ให้กลายเป็น List
        actual_list = ast.literal_eval(row_labels)
        cleaned_list = [str(item).strip().lower() for item in actual_list]
        
        row_results = []
        for v7_class in CLASSES:
            match = 0
            for keyword in LABEL_MAPPING[v7_class]:
                if any(keyword in doctor_label for doctor_label in cleaned_list):
                    match = 1
                    break 
            row_results.append(match)
        return row_results
    except Exception as e:
        return [0] * 14

# รันการ Mapping
print("🧠 กำลังจับคู่คำวินิจฉัย (รอบนี้เก็บทุกไฟล์)...")
results = df['Labels'].apply(map_labels)
df[CLASSES] = pd.DataFrame(results.tolist(), index=df.index)

# เลือกเฉพาะคอลัมน์ที่จำเป็น
final_df = df[['Filename'] + CLASSES]

# 🌟 หมายเหตุ: เราเอาบรรทัดที่กรอง sum(axis=1) > 0 ออกแล้ว 
# เพื่อให้ภาพที่เป็น Normal ยังคงอยู่ใน CSV

final_df.to_csv(OUTPUT_CSV, index=False)
print(f"🎉 สำเร็จ! บันทึกไฟล์เฉลยใหม่ที่: {OUTPUT_CSV}")
print(f"📊 จำนวนแถวทั้งหมดใน CSV ใหม่: {len(final_df)} รายการ (รวม Normal แล้ว)")