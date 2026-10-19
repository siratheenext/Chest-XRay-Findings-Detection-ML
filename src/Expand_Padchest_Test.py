import pandas as pd
import shutil
import os

# =========================================================================
# CONFIG: รันจากโฟลเดอร์ src/ (เหมือนสคริปต์อื่นๆ ที่ผ่านมา)
# =========================================================================
PADCHEST_LABELS_CSV = '../data/csv/padchest_test_v7_format.csv'
PADCHEST_FULL_META_CSV = '../data/csv/PadChest/PADCHEST_chest_x_ray_images_labels.csv'

# โฟลเดอร์ที่เพิ่งแตก zip ออกมา (เปลี่ยน path นี้ทุกครั้งที่แตกไฟล์ใหม่)
EXTRACTED_DIR = '../data/Padchest/extracted_54'

# โฟลเดอร์ปลายทางที่มีอยู่แล้ว (จะ copy ภาพที่ตรงเงื่อนไขไปเพิ่มตรงนี้)
TARGET_DIRS = {
    'PA':            '../data/Padchest/100_padchest_PA_chest',
    'AP':            '../data/Padchest/100_padchest_AP_chest',
    'AP_horizontal': '../data/Padchest/100_padchest_AP_horizontal_chest',
}

TARGET_COUNT_PER_GROUP = 500  # เป้าหมายจำนวนภาพต่อกลุ่ม
# =========================================================================


def main():
    if not os.path.exists(EXTRACTED_DIR):
        print(f"Error: ไม่เจอโฟลเดอร์ {EXTRACTED_DIR}")
        return
    if not os.path.exists(PADCHEST_LABELS_CSV) or not os.path.exists(PADCHEST_FULL_META_CSV):
        print("Error: ไม่เจอไฟล์ CSV ที่ต้องใช้ (padchest_test_v7_format.csv หรือ PADCHEST_chest_x_ray_images_labels.csv)")
        return

    print("กำลังอ่าน label และ metadata...")
    labels_df = pd.read_csv(PADCHEST_LABELS_CSV)
    labels_df['basename'] = labels_df['Filename'].apply(lambda x: os.path.basename(x))

    meta_df = pd.read_csv(PADCHEST_FULL_META_CSV, usecols=['ImageID', 'Projection'])
    merged = labels_df.merge(meta_df, left_on='basename', right_on='ImageID', how='left')

    # หาไฟล์ที่มีอยู่จริงในโฟลเดอร์ที่เพิ่งแตกมา (อาจมี subfolder ซ้อนอยู่ เผื่อไว้)
    print(f"กำลังสแกนไฟล์ใน {EXTRACTED_DIR} ...")
    extracted_files = {}  # basename -> full path
    for root, _, files in os.walk(EXTRACTED_DIR):
        for f in files:
            if f.lower().endswith('.png'):
                extracted_files[f] = os.path.join(root, f)
    print(f"เจอไฟล์ .png ทั้งหมด {len(extracted_files)} ไฟล์ในโฟลเดอร์ที่แตกมา")

    if len(extracted_files) == 0:
        print("ไม่เจอไฟล์ .png เลย เช็คว่า EXTRACTED_DIR ถูกต้องมั้ย หรือไฟล์เป็นนามสกุลอื่น (เช่น .dcm)")
        return

    # ทำทีละกลุ่ม (PA / AP / AP_horizontal)
    for proj_key, target_dir in TARGET_DIRS.items():
        os.makedirs(target_dir, exist_ok=True)

        current_count = len([f for f in os.listdir(target_dir) if f.lower().endswith('.png')])
        need_more = TARGET_COUNT_PER_GROUP - current_count

        print(f"\n=== {proj_key} ===")
        print(f"  มีอยู่แล้ว: {current_count} ภาพ | เป้าหมาย: {TARGET_COUNT_PER_GROUP} ภาพ | ต้องการเพิ่ม: {max(0, need_more)} ภาพ")

        if need_more <= 0:
            print(f"  ครบแล้ว ไม่ต้องเพิ่ม")
            continue

        # หาภาพที่ตรงกับ Projection นี้ และมีไฟล์อยู่จริงในโฟลเดอร์ที่แตกมา
        candidates = merged[merged['Projection'] == proj_key].copy()
        candidates = candidates[candidates['basename'].isin(extracted_files.keys())]

        # ตัดภาพที่มีอยู่ในโฟลเดอร์ปลายทางอยู่แล้ว (กันก็อปซ้ำ)
        existing_in_target = set(os.listdir(target_dir))
        candidates = candidates[~candidates['basename'].isin(existing_in_target)]

        print(f"  เจอภาพที่ตรงเงื่อนไข (Projection={proj_key}, มี label, ยังไม่เคยก็อป): {len(candidates)} ภาพ")

        # สุ่มเลือกเท่าที่ต้องการ (หรือเอาหมดถ้าน้อยกว่าที่ต้องการ)
        to_copy = candidates.sample(n=min(need_more, len(candidates)), random_state=42) if len(candidates) > 0 else candidates

        copied = 0
        for _, row in to_copy.iterrows():
            basename = row['basename']
            src = extracted_files[basename]
            dst = os.path.join(target_dir, basename)
            try:
                shutil.copy(src, dst)
                copied += 1
            except Exception as e:
                print(f"  ERROR copy {basename}: {e}")

        new_total = current_count + copied
        print(f"  ก็อปเพิ่มสำเร็จ: {copied} ภาพ | รวมตอนนี้: {new_total} ภาพ")
        if new_total < TARGET_COUNT_PER_GROUP:
            print(f"  ยังขาดอีก {TARGET_COUNT_PER_GROUP - new_total} ภาพ -> ต้องโหลด zip ไฟล์อื่นเพิ่ม")

    print("\n" + "="*50)
    print("สรุปหลังรันรอบนี้:")
    for proj_key, target_dir in TARGET_DIRS.items():
        count = len([f for f in os.listdir(target_dir) if f.lower().endswith('.png')])
        status = "ครบแล้ว" if count >= TARGET_COUNT_PER_GROUP else f"ขาดอีก {TARGET_COUNT_PER_GROUP - count}"
        print(f"  {proj_key}: {count}/{TARGET_COUNT_PER_GROUP} ({status})")
    print("="*50)
    print("\nเสร็จแล้ว! ถ้ายังไม่ครบ ให้โหลด .zip ไฟล์ถัดไป แตกไฟล์ แล้วแก้ EXTRACTED_DIR ในสคริปต์นี้ รันซ้ำได้เลย")
    print(f"หลังจากนี้ลบโฟลเดอร์ {EXTRACTED_DIR} และไฟล์ zip ทิ้งได้ เพื่อประหยัดพื้นที่ดิสก์")


if __name__ == "__main__":
    main()