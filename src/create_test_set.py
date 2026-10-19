import pandas as pd
import os
import shutil

# ================= ⚙️ CONFIG PATHS =================
SOURCE_FOLDER = '../data/all_images_raw'
TRAIN_CSV_PATH = '../data/csv/train_labels.csv' # ต้องรัน SelectTrainData ก่อน
TEST_FOLDER = '../data/test_images'
ORIGINAL_CSV = '../data/csv/Data_Entry_2017.csv'
OUTPUT_TEST_CSV = '../data/csv/test_labels.csv'

TARGET_DISEASES = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Emphysema', 'Fibrosis', 'No Finding']
TEST_IMAGES_PER_CLASS = 150
# ====================================================

def main():
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"❌ Error: ไม่เจอไฟล์ {TRAIN_CSV_PATH}")
        print("💡 กรุณารัน 'SelectTrainData.py' ก่อนนะครับ")
        return

    print("⏳ Loading Data...")
    df_all = pd.read_csv(ORIGINAL_CSV)
    df_train = pd.read_csv(TRAIN_CSV_PATH)

    # Exclude training images
    train_filenames = set(df_train['Filename'].values)
    df_remaining = df_all[~df_all['Image Index'].isin(train_filenames)]
    
    print(f"🔹 Total available for testing: {len(df_remaining)} images")

    if not os.path.exists(TEST_FOLDER):
        os.makedirs(TEST_FOLDER)

    test_data = []
    print("🚀 Creating Test Set...")

    for disease in TARGET_DISEASES:
        if disease == 'No Finding':
            subset = df_remaining[df_remaining['Finding Labels'] == 'No Finding']
        else:
            subset = df_remaining[df_remaining['Finding Labels'].str.contains(disease, regex=False)]
        
        if len(subset) >= TEST_IMAGES_PER_CLASS:
            selected = subset.sample(n=TEST_IMAGES_PER_CLASS, random_state=999)
        else:
            selected = subset
        
        for _, row in selected.iterrows():
            img_name = row['Image Index']
            src_path = os.path.join(SOURCE_FOLDER, img_name)
            dst_path = os.path.join(TEST_FOLDER, img_name)
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                entry = {'Filename': img_name}
                for d in TARGET_DISEASES:
                    if d == 'No Finding': continue
                    entry[d] = 1 if d in row['Finding Labels'] else 0
                test_data.append(entry)

    # Save CSV
    pd.DataFrame(test_data).to_csv(OUTPUT_TEST_CSV, index=False)
    print(f"\n✅ Test Set Created: {len(test_data)} images")
    print(f"📂 Images: {TEST_FOLDER}")
    print(f"📄 Labels: {OUTPUT_TEST_CSV}")

if __name__ == "__main__":
    main()