import pandas as pd
import shutil
import os

# ================= ⚙️ CONFIG PATHS =================
# ย้อนกลับ 1 ชั้น (..) เพื่อออกจาก src ไปหา data
SOURCE_FOLDER = '../data/all_images_raw' 
DEST_FOLDER = '../data/train_images_3000PIC'
CSV_PATH = '../data/csv/Data_Entry_2017.csv'
OUTPUT_CSV = '../data/csv/train_labels_3000PIC.csv'

TARGET_DISEASES = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Emphysema', 'Fibrosis', 'No Finding']
# IMAGES_PER_CLASS = 1500 # ปรับจำนวนรูปตรงนี้
IMAGES_PER_CLASS = 3000 # ปรับจำนวนรูปตรงนี้
# ====================================================

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: ไม่เจอไฟล์ CSV ที่ {CSV_PATH}")
        print("💡 อย่าลืมย้ายไฟล์ Data_Entry_2017.csv ไปไว้ในโฟลเดอร์ data/csv/ นะครับ")
        return

    print("⏳ Reading CSV...")
    df = pd.read_csv(CSV_PATH)

    if not os.path.exists(DEST_FOLDER):
        os.makedirs(DEST_FOLDER)
        print(f"✅ Created folder: {DEST_FOLDER}")

    pilot_data = []
    print(f"🚀 Starting selection ({IMAGES_PER_CLASS} images/class)...")

    for disease in TARGET_DISEASES:
        if disease == 'No Finding':
            subset = df[df['Finding Labels'] == 'No Finding']
        else:
            subset = df[df['Finding Labels'].str.contains(disease, regex=False)]
        
        # Sampling
        if len(subset) >= IMAGES_PER_CLASS:
            selected_files = subset.sample(n=IMAGES_PER_CLASS, random_state=42)
        else:
            selected_files = subset
            print(f"⚠️ {disease} has only {len(subset)} images.")

        # Copy Files
        count = 0
        for _, row in selected_files.iterrows():
            image_name = row['Image Index']
            src = os.path.join(SOURCE_FOLDER, image_name)
            dst = os.path.join(DEST_FOLDER, image_name)
            
            try:
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    # Create Label Entry
                    entry = {'Filename': image_name}
                    for d in TARGET_DISEASES:
                        if d == 'No Finding': continue
                        entry[d] = 1 if d in row['Finding Labels'] else 0
                    pilot_data.append(entry)
                    count += 1
            except Exception as e:
                print(f"Error copying {image_name}: {e}")
        
        print(f"✅ {disease}: Collected {count} images")

    # Save New CSV
    if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
        os.makedirs(os.path.dirname(OUTPUT_CSV))
        
    pd.DataFrame(pilot_data).to_csv(OUTPUT_CSV, index=False)
    print("\n🎉 Train Dataset Created Successfully!")
    print(f"📂 Images: {DEST_FOLDER}")
    print(f"📄 Labels: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()