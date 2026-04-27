"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║        🧠 AI-Assisted Brain Tumor Segmentation (BraTS 2021)                    ║
║        การพัฒนาแบบจำลอง AI เพื่อระบุตำแหน่งเนื้องอกในสมองด้วย Attention U-Net ║
╚══════════════════════════════════════════════════════════════════════════════════╝

📖 ภาพรวมของโครงงาน:
    โครงงานนี้ใช้ Deep Learning ในการทำ Brain Tumor Segmentation จากภาพ MRI
    โดยใช้ชุดข้อมูล RSNA-ASNR-MICCAI BraTS 2021 ที่มีผู้ป่วย 1,251 เคส
    แต่ละเคสประกอบด้วยภาพ MRI 4 รูปแบบ (Modalities):
      - T1:    โครงสร้างเนื้อเยื่อสมองทั่วไป
      - T1c:   ภาพที่ฉีดสารทึบรังสี ช่วยเห็นขอบเนื้องอกชัดขึ้น
      - T2:    แสดงพื้นที่บวมน้ำได้ดี
      - FLAIR: ตัดสัญญาณน้ำในโพรงสมอง เห็นรอยผิดปกติชัดเจน

🎯 เป้าหมาย: แบ่งเนื้องอกออกเป็น 3 บริเวณ:
      Class 1: NCR  - Necrotic Core       (แกนกลางที่ตายแล้ว)
      Class 2: ED   - Edema               (บริเวณบวมน้ำ)
      Class 3: ET   - Enhancing Tumor     (ส่วนที่กำลังเติบโต)
      Class 0: Background                 (ส่วนที่เป็นปกติ)

🗺️ แผนการดำเนินงาน (10 ส่วน):
      ส่วนที่ 1:  แตกไฟล์และโหลดข้อมูล
      ส่วนที่ 2:  สำรวจและแบ่งข้อมูลระดับผู้ป่วย
      ส่วนที่ 3:  Preprocessing (Crop, Normalize, Resize)
      ส่วนที่ 4:  PyTorch Dataset และ DataLoader
      ส่วนที่ 5:  สถาปัตยกรรม Attention U-Net
      ส่วนที่ 6:  Loss Function และ Metrics
      ส่วนที่ 7:  Training Loop พื้นฐาน (10 Epochs)
      ส่วนที่ 8:  Inference และ Visualization
      ส่วนที่ 9:  Advanced Training (30 Epochs + Dice-CE Loss)
      ส่วนที่ 10: Ultimate Training (100 Epochs + AMP + Augmentation)
      ส่วนที่ 11: Learning Curves และ Confusion Matrix
"""

# ==============================================================================
# ส่วนที่ 1: แตกไฟล์ข้อมูล (Extract Dataset)
# ==============================================================================
"""
📦 คำอธิบาย:
    ชุดข้อมูล BraTS 2021 บน Kaggle จะอยู่ในรูปแบบไฟล์ .tar (เหมือนกล่องบรรจุ)
    เราต้องแตกไฟล์ออกมาก่อนจึงจะใช้งานได้

🔑 แนวคิดสำคัญ:
    - ตรวจสอบก่อนว่าเคยแตกไฟล์ไปแล้วหรือยัง (ด้วย os.path.exists)
    - ถ้ายังไม่มีโฟลเดอร์ ให้สร้างและแตกไฟล์
    - ถ้ามีแล้ว ก็ข้ามขั้นตอนนี้ไปได้เลย ไม่ต้องทำซ้ำ (ประหยัดเวลา 2-5 นาที)
"""
import tarfile
import os

# ระบุตำแหน่งไฟล์ .tar ที่เก็บข้อมูลทั้งหมดไว้
tar_path = "/kaggle/input/datasets/dschettler8845/brats-2021-task1/BraTS2021_Training_Data.tar"

# โฟลเดอร์ปลายทางที่จะนำข้อมูลออกมาวาง
extract_path = "/kaggle/working/BraTS2021_Training_Data"

# ตรวจสอบว่าเคยแตกไฟล์ไปแล้วหรือยัง
if not os.path.exists(extract_path):
    os.makedirs(extract_path)
    print(f"📦 กำลังแตกไฟล์ {tar_path} \\n⏳ อาจใช้เวลาประมาณ 2-5 นาที...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_path)
    print("✅ แตกไฟล์สำเร็จเรียบร้อย!")
else:
    print("✅ โฟลเดอร์มีอยู่แล้ว พร้อมไปต่อได้เลย!")


# ==============================================================================
# ส่วนที่ 2: สำรวจข้อมูลและแบ่งระดับผู้ป่วย (Data Exploration & Patient-Level Split)
# ==============================================================================
"""
📦 คำอธิบาย:
    ขั้นตอนนี้สำคัญมาก! เราต้องแบ่งข้อมูลในระดับ "ผู้ป่วย" ไม่ใช่ระดับ "ภาพ"
    เพื่อป้องกัน Data Leakage

⚠️ ทำไม Patient-Level Split ถึงสำคัญ?
    ถ้าเราสุ่มตัดภาพ 2D รวมๆ กัน อาจเกิดกรณีที่:
    - สไลซ์ที่ 70 ของผู้ป่วย A อยู่ใน Train Set
    - สไลซ์ที่ 71 ของผู้ป่วย A อยู่ใน Test Set
    ซึ่งโมเดลจะ "จำ" ผู้ป่วยคนนั้นได้ และ Test Score จะสูงเกินจริง (โกง!)
    
    วิธีที่ถูกต้องคือ แยกรายชื่อผู้ป่วยก่อน:
      Train: ผู้ป่วยกลุ่ม A (70%)
      Val:   ผู้ป่วยกลุ่ม B (15%)
      Test:  ผู้ป่วยกลุ่ม C (15%) <- ชุดข้อมูลตาบอด โมเดลไม่เคยเห็นเลย

🔑 ฟังก์ชันที่ใช้:
    - glob.glob(): ค้นหาไฟล์/โฟลเดอร์ตาม pattern ที่กำหนด
    - train_test_split(): แบ่งข้อมูล จาก sklearn
    - nib.load().get_fdata(): โหลดไฟล์ NIfTI (.nii.gz) และแปลงเป็น NumPy array
"""
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATASET_PATH = "/kaggle/working/BraTS2021_Training_Data"

# ค้นหาโฟลเดอร์ผู้ป่วยทั้งหมด (ชื่อขึ้นต้นด้วย 'BraTS2021_')
all_patient_paths = glob.glob(os.path.join(DATASET_PATH, "**", "BraTS2021_*"), recursive=True)

# กรองเอาเฉพาะโฟลเดอร์ (ตัดไฟล์ .tar ออก)
patient_folders = [p for p in all_patient_paths if os.path.isdir(p)]
patient_folders = sorted(list(set(patient_folders)))

print(f"✅ ค้นพบข้อมูลผู้ป่วยทั้งหมด: {len(patient_folders)} เคส")

# แบ่งข้อมูลระดับผู้ป่วย (Patient-Level Split)
# ขั้นที่ 1: แบ่ง Train 70% และ ส่วนที่เหลือ 30%
train_patients, temp_patients = train_test_split(patient_folders, test_size=0.30, random_state=42)
# ขั้นที่ 2: แบ่งส่วนที่เหลือออกเป็น Val 15% และ Test 15%
val_patients, test_patients = train_test_split(temp_patients, test_size=0.50, random_state=42)

print(f"📊 แบ่งข้อมูล: Train={len(train_patients)}, Val={len(val_patients)}, Test={len(test_patients)}")

# --- ทดลองโหลดและแสดงภาพผู้ป่วย 1 คน ---
def load_nifti(file_path):
    """
    โหลดไฟล์ NIfTI (.nii.gz) และแปลงเป็น NumPy array
    NIfTI คือรูปแบบไฟล์มาตรฐานสำหรับภาพทางการแพทย์ 3D
    """
    if os.path.exists(file_path):
        return nib.load(file_path).get_fdata()
    return None

# ทดลองกับผู้ป่วยคนแรกใน Train Set
sample_patient = train_patients[0]
patient_id = os.path.basename(sample_patient)

img_t1    = load_nifti(os.path.join(sample_patient, f"{patient_id}_t1.nii.gz"))
img_t1ce  = load_nifti(os.path.join(sample_patient, f"{patient_id}_t1ce.nii.gz"))
img_t2    = load_nifti(os.path.join(sample_patient, f"{patient_id}_t2.nii.gz"))
img_flair = load_nifti(os.path.join(sample_patient, f"{patient_id}_flair.nii.gz"))
img_seg   = load_nifti(os.path.join(sample_patient, f"{patient_id}_seg.nii.gz"))

if img_t1 is not None:
    print(f"📏 ขนาดภาพ 3D: {img_t1.shape}")  # ปกติ (240, 240, 155)

    # เลือกสไลซ์ตรงกลางตามแกน Z มาแสดงผล
    slice_idx = img_t1.shape[2] // 2

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    modalities = [img_t1, img_t1ce, img_t2, img_flair, img_seg]
    titles = ['T1', 'T1c (T1Gd)', 'T2', 'FLAIR', 'Segmentation Mask']
    cmaps  = ['gray', 'gray', 'gray', 'gray', 'nipy_spectral']

    for ax, img, title, cmap in zip(axes, modalities, titles, cmaps):
        ax.imshow(img[:, :, slice_idx], cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ==============================================================================
# ส่วนที่ 3: การ Preprocessing (Cropping, Normalization, Resizing)
# ==============================================================================
"""
📦 คำอธิบาย:
    ก่อนป้อนภาพเข้าโมเดล เราต้องเตรียมภาพให้อยู่ในรูปแบบที่เหมาะสม
    กระบวนการ Preprocessing ประกอบด้วย 4 ขั้นตอนหลัก:

    1. Cropping (ตัดขอบดำ):
       ภาพ MRI ดิบมักมีพื้นที่สีดำรอบๆ สมอง (Background) ซึ่งไม่มีข้อมูลที่มีประโยชน์
       เราหา Bounding Box ของสมอง แล้วตัดส่วนที่ไม่จำเป็นออก
       → โมเดลได้โฟกัสกับเนื้อสมองทุกพิกเซล

    2. Z-Score Normalization:
       ภาพ MRI แต่ละเครื่อง แต่ละวัน มีความสว่างต่างกัน (ค่า pixel ไม่เป็นมาตรฐาน)
       Z-Score ทำให้ภาพทุกรูปมี Mean=0, Std=1
       → โมเดลเรียนรู้ได้เร็วและเสถียรกว่า

    3. Resizing เป็น 256×256:
       สถาปัตยกรรม U-Net ต้องการขนาด Input ที่คงที่
       → ปรับทุกภาพให้เป็น 256×256 พิกเซล

    4. Label Remapping (ปรับ Label 4 → 3):
       Mask เดิมมีค่า: 0, 1, 2, 4 (ข้ามเลข 3!)
       เราเปลี่ยน 4 → 3 เพื่อให้คลาสเรียงต่อเนื่อง: 0, 1, 2, 3
       → PyTorch Loss Function ต้องการ Label ที่เรียงต่อกัน

⚠️ ข้อควรระวังสำคัญมาก:
    การ Resize ภาพ (Input): ใช้ cv2.INTER_LINEAR  (แก้ pixel เรียบขึ้น)
    การ Resize Mask (Label): ต้องใช้ cv2.INTER_NEAREST เท่านั้น!
    เหตุผล: ถ้าใช้ INTER_LINEAR กับ Mask จะเกิด pixel ค่า 1.5, 2.3 ฯลฯ
            ซึ่งไม่ใช่ Label ที่ถูกต้อง → โมเดลจะสับสนและเทรนไม่ได้ผล
"""
import cv2

def get_brain_bounding_box(volume):
    """
    หา Bounding Box ของเนื้อสมอง เพื่อตัดขอบดำออก
    
    วิธีการ:
    - หาพิกัด (X, Y, Z) ทั้งหมดที่มีค่า > 0 (ไม่ใช่พื้นหลังสีดำ)
    - หาค่า min และ max ของแต่ละแกน
    - เพิ่ม Padding 5 pixel รอบๆ เพื่อไม่ให้ภาพชิดขอบเกินไป
    
    Args:
        volume: NumPy array 3D ของภาพ MRI
    Returns:
        (x_min, x_max, y_min, y_max, z_min, z_max): พิกัดขอบเขตสมอง
    """
    mask = volume > 0
    coords = np.array(np.nonzero(mask))

    if coords.shape[1] == 0:
        return 0, volume.shape[0], 0, volume.shape[1], 0, volume.shape[2]

    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()

    # เพิ่ม Padding 5 pixel และ clamp ไม่ให้เกินขอบภาพ
    return (max(0, x_min - 5), min(volume.shape[0], x_max + 5),
            max(0, y_min - 5), min(volume.shape[1], y_max + 5),
            max(0, z_min - 5), min(volume.shape[2], z_max + 5))


def normalize_z_score(image):
    """
    Z-Score Normalization: ปรับภาพให้มีค่าเฉลี่ย=0, ส่วนเบี่ยงเบนมาตรฐาน=1
    
    สูตร: z = (x - mean) / std
    
    ข้อสำคัญ: คำนวณ mean และ std เฉพาะพิกเซลที่ > 0 (เนื้อสมอง)
    และบังคับให้พื้นหลัง (ค่า = 0) กลับมาเป็น 0 เหมือนเดิมหลัง Normalize
    
    Args:
        image: NumPy array 2D หรือ 3D ของภาพ
    Returns:
        image ที่ผ่านการ Normalize แล้ว
    """
    mask = image > 0
    if mask.sum() > 0:
        mean = image[mask].mean()
        std  = image[mask].std()
        image = (image - mean) / (std + 1e-8)
        image[~mask] = 0  # บังคับให้พื้นหลังกลับมาเป็น 0
    return image


# ทดลองทำ Preprocessing กับผู้ป่วยตัวอย่าง
x_min, x_max, y_min, y_max, z_min, z_max = get_brain_bounding_box(img_t1)

# ครอปทุก Modality ด้วย Bounding Box เดียวกัน (อ้างอิงจาก T1)
crop_t1    = img_t1[x_min:x_max, y_min:y_max, z_min:z_max]
crop_t1ce  = img_t1ce[x_min:x_max, y_min:y_max, z_min:z_max]
crop_t2    = img_t2[x_min:x_max, y_min:y_max, z_min:z_max]
crop_flair = img_flair[x_min:x_max, y_min:y_max, z_min:z_max]
crop_seg   = img_seg[x_min:x_max, y_min:y_max, z_min:z_max]

# ดึงสไลซ์ 2D ตรงกลาง
new_slice_idx = crop_t1.shape[2] // 2
img_size = (256, 256)

# Normalize และ Resize ภาพ
norm_t1    = normalize_z_score(crop_t1[:, :, new_slice_idx])
norm_flair = normalize_z_score(crop_flair[:, :, new_slice_idx])
res_t1     = cv2.resize(norm_t1,    img_size, interpolation=cv2.INTER_LINEAR)
res_flair  = cv2.resize(norm_flair, img_size, interpolation=cv2.INTER_LINEAR)

# Resize Mask ต้องใช้ INTER_NEAREST เท่านั้น! (ห้ามเบลอ Label)
res_seg = cv2.resize(crop_seg[:, :, new_slice_idx], img_size, interpolation=cv2.INTER_NEAREST)
res_seg[res_seg == 4] = 3  # Remap: 4 → 3

print(f"📏 ขนาดภาพหลัง Preprocessing: {res_t1.shape}")


# ==============================================================================
# ส่วนที่ 4: PyTorch Dataset และ DataLoader
# ==============================================================================
"""
📦 คำอธิบาย:
    ใน PyTorch เราต้องสร้าง Dataset class ที่บอกให้โมเดลรู้ว่า:
    1. มีข้อมูลกี่ชิ้น (__len__)
    2. ดึงข้อมูลชิ้นที่ N ออกมาอย่างไร (__getitem__)

    จากนั้น DataLoader จะดูแลการดึงข้อมูลแบบ Batch และ Shuffle อัตโนมัติ

🔑 เทคนิคสำคัญ - Dynamic Slice Selection:
    แทนที่จะเก็บสไลซ์ทุกแผ่นไว้ล่วงหน้า (ต้องใช้ RAM มาก)
    เราโหลดภาพ 3D แล้ว "สุ่มเลือก" สไลซ์ที่มีเนื้องอกมาแบบ Real-time
    
    ทำไมต้องสุ่มสไลซ์ที่มีเนื้องอก?
    → แก้ปัญหา Class Imbalance: ใน 155 สไลซ์ มีแค่ 30-50 สไลซ์ที่มีเนื้องอก
    → ถ้าสุ่มปกติ โมเดลจะเจอแต่สไลซ์ Background และเรียนรู้แต่ "ทายว่าปกติ"
    → กำหนด threshold ว่าต้องมี > 50 pixel ที่เป็นเนื้องอก จึงเลือกสไลซ์นั้น

📐 รูปแบบ Tensor ที่ออกมา:
    Image: (4, 256, 256) → 4 Channels (T1, T1c, T2, FLAIR), ขนาด 256×256
    Mask:  (256, 256)    → ค่าแต่ละ pixel คือ Label (0, 1, 2, 3)
"""
import torch
from torch.utils.data import Dataset, DataLoader
import random

class BraTS2DDataset(Dataset):
    def __init__(self, patient_paths, is_train=True):
        """
        Args:
            patient_paths: ลิสต์ path โฟลเดอร์ผู้ป่วย
            is_train: True = สุ่มสไลซ์ที่มีเนื้องอก (Train mode)
                      False = ใช้สไลซ์ตรงกลาง (Eval mode)
        """
        self.patient_paths = patient_paths
        self.is_train = is_train

    def __len__(self):
        # จำนวน "ตัวอย่าง" = จำนวนผู้ป่วย (1 คน = 1 สไลซ์ต่อ Epoch)
        return len(self.patient_paths)

    def __getitem__(self, idx):
        patient_dir = self.patient_paths[idx]
        patient_id  = os.path.basename(patient_dir)

        # 1. โหลดไฟล์ NIfTI 3D ทั้ง 5 ไฟล์
        t1    = load_nifti(os.path.join(patient_dir, f"{patient_id}_t1.nii.gz"))
        t1ce  = load_nifti(os.path.join(patient_dir, f"{patient_id}_t1ce.nii.gz"))
        t2    = load_nifti(os.path.join(patient_dir, f"{patient_id}_t2.nii.gz"))
        flair = load_nifti(os.path.join(patient_dir, f"{patient_id}_flair.nii.gz"))
        seg   = load_nifti(os.path.join(patient_dir, f"{patient_id}_seg.nii.gz"))

        # 2. Cropping: ตัดขอบดำออก (ใช้ Bounding Box จาก T1 เป็นอ้างอิง)
        x_min, x_max, y_min, y_max, z_min, z_max = get_brain_bounding_box(t1)
        t1    = t1[x_min:x_max, y_min:y_max, z_min:z_max]
        t1ce  = t1ce[x_min:x_max, y_min:y_max, z_min:z_max]
        t2    = t2[x_min:x_max, y_min:y_max, z_min:z_max]
        flair = flair[x_min:x_max, y_min:y_max, z_min:z_max]
        seg   = seg[x_min:x_max, y_min:y_max, z_min:z_max]

        # 3. Dynamic Slice Selection: เลือกสไลซ์ที่มีเนื้องอก
        # นับจำนวน pixel เนื้องอก (> 0) ในแต่ละสไลซ์ตามแกน Z
        tumor_pixels_per_slice = np.sum(seg > 0, axis=(0, 1))
        valid_slices = np.where(tumor_pixels_per_slice > 50)[0]

        if len(valid_slices) > 0 and self.is_train:
            chosen_slice = random.choice(valid_slices)  # สุ่มสไลซ์ที่มีเนื้องอก
        else:
            chosen_slice = t1.shape[2] // 2             # ใช้สไลซ์กลาง

        # 4. ฟังก์ชัน Normalize + Resize แต่ละ Modality
        img_size = (256, 256)

        def process_slice(vol, is_mask=False):
            slc = vol[:, :, chosen_slice]
            if not is_mask:
                slc = normalize_z_score(slc)
                slc = cv2.resize(slc, img_size, interpolation=cv2.INTER_LINEAR)
            else:
                # Mask ต้องใช้ INTER_NEAREST เพื่อไม่ให้ Label เบลอ
                slc = cv2.resize(slc, img_size, interpolation=cv2.INTER_NEAREST)
                slc[slc == 4] = 3  # Remap Label 4 → 3
            return slc

        slice_t1    = process_slice(t1)
        slice_t1ce  = process_slice(t1ce)
        slice_t2    = process_slice(t2)
        slice_flair = process_slice(flair)
        slice_seg   = process_slice(seg, is_mask=True)

        # 5. Stack 4 Modalities เป็น 4-Channel Image
        # Shape: (4, 256, 256) ← รูปแบบที่ PyTorch ต้องการ (C, H, W)
        image_stack = np.stack([slice_t1, slice_t1ce, slice_t2, slice_flair], axis=0)

        # 6. แปลงเป็น PyTorch Tensor
        image_tensor = torch.from_numpy(image_stack).float()  # float32 สำหรับภาพ
        mask_tensor  = torch.from_numpy(slice_seg).long()      # int64 สำหรับ Label

        return image_tensor, mask_tensor


# สร้าง DataLoader
BATCH_SIZE   = 8  # จำนวนสไลซ์ต่อ Batch (T4 GPU รับได้ถึง 16)
train_dataset = BraTS2DDataset(train_patients, is_train=True)
val_dataset   = BraTS2DDataset(val_patients,   is_train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ตรวจสอบรูปร่างของ Tensor
images, masks = next(iter(train_loader))
print(f"📦 Images shape: {images.shape}  → (Batch, Channels, H, W)")
print(f"📦 Masks  shape: {masks.shape}   → (Batch, H, W)")


# ==============================================================================
# ส่วนที่ 5: สถาปัตยกรรม Attention U-Net
# ==============================================================================
"""
📦 คำอธิบาย:
    Attention U-Net คือการพัฒนาต่อยอดจาก U-Net ปกติ โดยเพิ่ม Attention Gate
    ซึ่งทำหน้าที่เป็น "แว่นขยาย" ให้โมเดลโฟกัสเฉพาะบริเวณที่เป็นเนื้องอก

🏗️ โครงสร้าง 3 ส่วนหลัก:

    1. Encoder (ขาซ้าย/ขาลง):
       Input (4, 256, 256)
         ↓ ConvBlock → e1 (64, 256, 256)
         ↓ MaxPool + ConvBlock → e2 (128, 128, 128)
         ↓ MaxPool + ConvBlock → e3 (256, 64, 64)
         ↓ MaxPool + ConvBlock → e4 (512, 32, 32)
         ↓ MaxPool + ConvBlock → Bottleneck (1024, 16, 16)

    2. Attention Gate (แว่นขยาย):
       รับ 2 Input:
       - g (Gating signal): สัญญาณจากชั้นที่ลึกกว่า (บอกว่า "ควรมองที่ไหน")
       - x (Skip connection): ภาพจาก Encoder ที่มีรายละเอียดสูง
       Output: x ที่ถูกกรองเหลือแต่ส่วนที่สำคัญ

    3. Decoder (ขาขวา/ขาขึ้น):
       Bottleneck
         ↓ ConvTranspose (Upsample) + Attention Gate + Concat + ConvBlock
         ↓ ... ×4 ระดับ
         ↓ Output Conv → (num_classes=4, 256, 256)
       แต่ละระดับจะนำ Feature Map จาก Encoder มารวม (Skip Connection)
       ทำให้โมเดลรักษาทั้งข้อมูลระดับ Global และ Local ไว้ได้

🔑 ConvBlock: Conv→BN→ReLU→Conv→BN→ReLU
    - BatchNorm: ทำให้การเทรนเสถียร ลด Internal Covariate Shift
    - ReLU:      ฟังก์ชัน Activation ที่เพิ่มความ Non-linear
    - padding=1: รักษาขนาดภาพไม่ให้เล็กลงหลัง Conv

🔑 Attention Gate:
    สูตร: α = Sigmoid(ψ(ReLU(Wg·g + Wx·x)))
    Output: α × x  (คูณ Attention Map กับ Feature Map)
    ค่า α ใกล้ 1 = สำคัญ, ใกล้ 0 = ไม่สำคัญ
"""
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    บล็อกพื้นฐานของ U-Net
    ประกอบด้วย Conv3x3 → BatchNorm → ReLU → Conv3x3 → BatchNorm → ReLU
    padding=1 ทำให้ขนาด Output เท่ากับ Input (same padding)
    """
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # inplace=True ประหยัด Memory
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    กลไก Attention Gate: กรองข้อมูลจาก Skip Connection
    
    ให้ Decoder ที่ "รู้บริบท" (g) บอก Encoder ว่า "ควรโฟกัสที่ไหน" (x)
    
    Args:
        F_g:   จำนวน Channel ของ Gating Signal (จาก Decoder)
        F_l:   จำนวน Channel ของ Skip Connection (จาก Encoder)
        F_int: จำนวน Channel ตรงกลาง (ขนาดตัวกลาง)
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # แปลง Gating signal ให้มี F_int channels
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # แปลง Skip connection ให้มี F_int channels
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # คำนวณ Attention Map (ค่า 0-1 ต่อ pixel)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # บีบให้อยู่ระหว่าง 0-1
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g = Gating signal (จาก Decoder), x = Skip connection (จาก Encoder)
        g1 = self.W_g(g)          # แปลง g
        x1 = self.W_x(x)          # แปลง x
        psi = self.relu(g1 + x1)  # รวมสัญญาณ + ReLU
        psi = self.psi(psi)        # คำนวณน้ำหนัก Attention (0 ถึง 1)
        return x * psi             # ใช้ Attention กรองข้อมูล


class AttentionUNet(nn.Module):
    """
    Attention U-Net สำหรับ Brain Tumor Segmentation
    
    Input:  (Batch, 4, 256, 256)   ← 4 MRI Modalities
    Output: (Batch, 4, 256, 256)   ← Logits สำหรับ 4 Classes (0=BG, 1=NCR, 2=ED, 3=ET)
    
    Args:
        img_ch:      จำนวน Input Channel (4 สำหรับ T1, T1c, T2, FLAIR)
        num_classes: จำนวนคลาส (4: Background, NCR, ED, ET)
    """
    def __init__(self, img_ch=4, num_classes=4):
        super(AttentionUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # ลดขนาด 2 เท่า

        # ===== Encoder (ขาลง) =====
        # แต่ละขั้นเพิ่มจำนวน Channel เป็น 2 เท่า
        self.e1 = ConvBlock(img_ch, 64)    # 4   → 64  channels
        self.e2 = ConvBlock(64, 128)        # 64  → 128 channels
        self.e3 = ConvBlock(128, 256)       # 128 → 256 channels
        self.e4 = ConvBlock(256, 512)       # 256 → 512 channels

        # ===== Bottleneck (จุดลึกที่สุด) =====
        self.bottle = ConvBlock(512, 1024)  # 512 → 1024 channels, ขนาด 16×16

        # ===== Decoder (ขาขึ้น) พร้อม Attention Gates =====
        # ConvTranspose2d: เพิ่มขนาดภาพ 2 เท่า (Upsampling)
        self.up4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.d4   = ConvBlock(1024, 512)  # Concat (512+512=1024) → 512

        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.d3   = ConvBlock(512, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.d2   = ConvBlock(256, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.d1   = ConvBlock(128, 64)

        # Output Layer: แปลงเป็น Logits สำหรับแต่ละคลาส
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # ===== Encoder Pass =====
        e1 = self.e1(x)                       # (B, 64,  256, 256)
        e2 = self.e2(self.Maxpool(e1))         # (B, 128, 128, 128)
        e3 = self.e3(self.Maxpool(e2))         # (B, 256, 64,  64)
        e4 = self.e4(self.Maxpool(e3))         # (B, 512, 32,  32)
        bottle = self.bottle(self.Maxpool(e4)) # (B, 1024, 16, 16)

        # ===== Decoder Pass (พร้อม Skip Connection + Attention) =====
        d4 = self.up4(bottle)           # Upsample: (B, 512, 32, 32)
        x4 = self.att4(g=d4, x=e4)     # กรองด้วย Attention
        d4 = torch.cat((x4, d4), dim=1) # Skip Connection: (B, 1024, 32, 32)
        d4 = self.d4(d4)               # ConvBlock: (B, 512, 32, 32)

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.d3(d3)

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.d2(d2)

        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.d1(d1)

        return self.out(d1)  # (B, 4, 256, 256) ← Raw Logits


# ทดสอบ Forward Pass
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 ใช้ Hardware: {device}")

model = AttentionUNet(img_ch=4, num_classes=4).to(device)

with torch.no_grad():
    dummy = images.to(device)
    out   = model(dummy)
    print(f"📤 Output shape: {out.shape} → (Batch, 4 Classes, H, W)")


# ==============================================================================
# ส่วนที่ 6: Loss Function และ Metrics
# ==============================================================================
"""
📦 คำอธิบาย:
    Loss Function บอกโมเดลว่า "ผิดพลาดแค่ไหน" ในแต่ละรอบการเทรน
    เราใช้ Dice Loss ที่ออกแบบมาเฉพาะสำหรับ Medical Segmentation

📐 Dice Score (F1-Score สำหรับ Segmentation):
    Dice = (2 × |A ∩ B|) / (|A| + |B|)
    A = พื้นที่ที่ AI ทำนาย, B = พื้นที่จริงจากแพทย์
    ค่า = 0: ไม่ทับซ้อนกันเลย, ค่า = 1: ทับซ้อนกันสมบูรณ์

    Dice Loss = 1 - Dice Score (เพื่อให้ Minimize ได้)

🎯 ทำไมถึงข้ามคลาส 0 (Background)?
    - พื้นหลังมีพิกเซลมากกว่าเนื้องอกมาก (Class Imbalance รุนแรง)
    - ถ้าคิด Loss รวมฉากหลัง โมเดลจะ "โกง" โดยทายว่าทุกอย่างเป็นปกติ
    - Dice สูงปลอม แต่จริงๆ หาเนื้องอกไม่เจอเลย

📊 Metrics - Confusion Matrix:
    TP (True Positive):  ทาย "เนื้องอก" และจริงๆ เป็น "เนื้องอก" ✓ ดี
    TN (True Negative):  ทาย "ปกติ" และจริงๆ เป็น "ปกติ" ✓ ดี
    FP (False Positive): ทาย "เนื้องอก" แต่จริงๆ เป็น "ปกติ" ✗ แย่ (ตื่นตูม)
    FN (False Negative): ทาย "ปกติ" แต่จริงๆ เป็น "เนื้องอก" ✗ แย่มาก (มองข้ามรอยโรค)

    Precision = TP / (TP + FP)  ← ถ้าบอกว่าเนื้องอก มีความน่าเชื่อถือแค่ไหน
    Recall    = TP / (TP + FN)  ← หาเนื้องอกได้ครบแค่ไหน (ทางการแพทย์สำคัญมาก!)
    Dice/F1   = 2TP / (2TP + FP + FN)
"""

class DiceLoss(nn.Module):
    """
    Dice Loss: คำนวณจากความทับซ้อนระหว่าง Prediction กับ Ground Truth
    
    ข้ามคลาส 0 (Background) เพื่อโฟกัสเฉพาะเนื้องอก
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits:  (Batch, 4, 256, 256) ← Raw output จากโมเดล
        # targets: (Batch, 256, 256)    ← Ground Truth Labels

        num_classes = logits.shape[1]

        # 1. แปลง Logits → Probability (0 ถึง 1) ด้วย Softmax
        probs = F.softmax(logits, dim=1)

        # 2. แปลง targets → One-Hot Encoding
        # (Batch, H, W) → (Batch, 4, H, W) โดยแต่ละ channel = 1 ถ้าเป็นคลาสนั้น
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 3. คำนวณ Dice แยกตามคลาส (เริ่มที่ 1 ข้าม Background)
        dice_scores = []
        for class_idx in range(1, num_classes):
            prob_c   = probs[:, class_idx, :, :]
            target_c = targets_one_hot[:, class_idx, :, :]

            intersection = torch.sum(prob_c * target_c)
            cardinality  = torch.sum(prob_c + target_c)

            dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
            dice_scores.append(dice)

        # หาค่าเฉลี่ย Dice ของ 3 คลาสเนื้องอก
        mean_dice = torch.mean(torch.stack(dice_scores))
        return 1.0 - mean_dice  # Dice Loss (ยิ่งน้อยยิ่งดี)


def calculate_metrics(preds, targets, num_classes=4):
    """
    คำนวณ TP, FP, FN และ Dice Score จาก Prediction
    
    Args:
        preds:   (Batch, H, W) ← argmax จาก logits
        targets: (Batch, H, W) ← Ground Truth
    Returns:
        dict ที่มี TP, FP, FN, Dice
    """
    metrics = {'TP': 0, 'FP': 0, 'FN': 0, 'Dice': 0}
    smooth  = 1e-5

    for c in range(1, num_classes):
        pred_c   = (preds == c).float()
        target_c = (targets == c).float()

        TP = torch.sum(pred_c * target_c).item()
        FP = torch.sum(pred_c * (1 - target_c)).item()
        FN = torch.sum((1 - pred_c) * target_c).item()

        metrics['TP'] += TP
        metrics['FP'] += FP
        metrics['FN'] += FN

    # สูตร Dice/F1: 2TP / (2TP + FP + FN)
    metrics['Dice'] = (
        (2 * metrics['TP'] + smooth) /
        (2 * metrics['TP'] + metrics['FP'] + metrics['FN'] + smooth)
    )
    return metrics


print("✅ Dice Loss และ Metrics พร้อมใช้งาน!")


# ==============================================================================
# ส่วนที่ 7: Training Loop พื้นฐาน (10 Epochs)
# ==============================================================================
"""
📦 คำอธิบาย:
    กระบวนการฝึกสอนโมเดลในแต่ละ Epoch ประกอบด้วย 3 ขั้นตอน:
    
    1. Train Phase:
       - ป้อนภาพ Input เข้าโมเดล (Forward Pass)
       - คำนวณ Dice Loss
       - Backpropagation: คำนวณ Gradient
       - Optimizer อัปเดตน้ำหนักโมเดล
    
    2. Validation Phase:
       - ปิดการอัปเดตน้ำหนัก (torch.no_grad())
       - คำนวณ Dice Score บน Validation Set
       - ใช้ประเมินว่าโมเดลเรียนรู้ได้จริงหรือ Overfit
    
    3. Save Best Model:
       - บันทึกน้ำหนักโมเดลที่ได้ Val Dice สูงที่สุด
       - ป้องกันการใช้โมเดลที่ Overfit หลังจบการเทรน

🔑 Multi-GPU Training (nn.DataParallel):
    Kaggle ให้ GPU T4 สองตัว เราใช้ DataParallel เพื่อ:
    - แบ่ง Batch ออกเป็น 2 ส่วน ส่งไปคำนวณบน GPU คนละตัว
    - รวมผลและอัปเดตน้ำหนักพร้อมกัน
    - ความเร็วเพิ่มขึ้นเกือบ 2 เท่า

🔑 AdamW Optimizer:
    พัฒนาต่อจาก Adam โดยแยก Weight Decay ออกมาจาก Gradient
    → ลด Overfitting ได้ดีกว่า Adam ปกติ
    lr=1e-4, weight_decay=1e-5
"""
import time

# ตั้งค่า Multi-GPU
if torch.cuda.device_count() > 1:
    print(f"🔥 ระบบตรวจพบ {torch.cuda.device_count()} GPU!")
    model = nn.DataParallel(model)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = DiceLoss()

num_epochs    = 10
best_val_dice = 0.0

print("🚀 เริ่มการฝึกสอนโมเดล...")

for epoch in range(num_epochs):
    start_time = time.time()

    # --- Train Phase ---
    model.train()  # เปิดโหมด Train (BatchNorm และ Dropout ทำงาน)
    train_loss = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()           # ล้าง Gradient เก่าออก
        outputs = model(images)          # Forward Pass
        loss    = criterion(outputs, masks)  # คำนวณ Dice Loss
        loss.backward()                  # Backpropagation
        optimizer.step()                 # ปรับน้ำหนัก

        train_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print(f"[Train] Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

    # --- Validation Phase ---
    model.eval()  # เปิดโหมด Eval (ปิด Dropout/BatchNorm update)
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():  # ปิดการคำนวณ Gradient (ประหยัด Memory)
        for images, masks in val_loader:
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            loss    = criterion(outputs, masks)
            val_loss += loss.item()

            # หาคลาสที่มีความน่าจะเป็นสูงสุดในแต่ละ pixel
            preds   = torch.argmax(outputs, dim=1)
            metrics = calculate_metrics(preds, masks)
            val_dice += metrics['Dice']

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    time_taken   = time.time() - start_time

    print(f"\n📊 Epoch[{epoch+1}/{num_epochs}] ({time_taken:.0f}s)")
    print(f"   Train Loss: {avg_train_loss:.4f}")
    print(f"   Val Loss:   {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

    # --- Save Best Model ---
    if avg_val_dice > best_val_dice:
        print(f"   🏆 Val Dice: {best_val_dice:.4f} → {avg_val_dice:.4f} (บันทึกโมเดล!)")
        best_val_dice = avg_val_dice
        torch.save(model.state_dict(), '/kaggle/working/best_attention_unet.pth')

    print("-" * 50)

print(f"\n🎉 Training สำเร็จ! Best Val Dice = {best_val_dice:.4f}")


# ==============================================================================
# ส่วนที่ 8: Inference และ Visualization
# ==============================================================================
"""
📦 คำอธิบาย:
    นำโมเดลที่ฝึกแล้วมาทดสอบกับ Test Set (ชุดข้อมูลตาบอด)
    แสดงผลเปรียบเทียบ 3 ภาพ:
    1. ภาพ MRI ต้นฉบับ (FLAIR Modality)
    2. Ground Truth จากแพทย์
    3. AI Prediction จาก Attention U-Net

🎨 Legend สี (nipy_spectral colormap):
    สีม่วง/ดำ = Class 0: Background
    สีเหลือง  = Class 1: Necrotic Core (NCR)
    สีน้ำเงิน = Class 2: Edema (ED)
    สีแดง     = Class 3: Enhancing Tumor (ET)
"""
import matplotlib.patches as mpatches

# โหลดโมเดลที่ดีที่สุด
inference_model = AttentionUNet(img_ch=4, num_classes=4)
if torch.cuda.device_count() > 1:
    inference_model = nn.DataParallel(inference_model)

inference_model.load_state_dict(torch.load('/kaggle/working/best_attention_unet.pth'))
inference_model = inference_model.to(device)
inference_model.eval()

# สร้าง DataLoader สำหรับ Test Set
test_dataset = BraTS2DDataset(test_patients, is_train=False)
test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=True)

# ดึงข้อมูล 1 Batch จาก Test Set
test_images, test_masks = next(iter(test_loader))
test_images = test_images.to(device)

with torch.no_grad():
    predictions    = inference_model(test_images)
    predicted_masks = torch.argmax(predictions, dim=1)  # เลือกคลาสที่มีค่าสูงสุด

# ดึงกลับมาที่ CPU สำหรับการแสดงผล
test_images_cpu    = test_images.cpu().numpy()
test_masks_cpu     = test_masks.cpu().numpy()
predicted_masks_cpu = predicted_masks.cpu().numpy()


def plot_prediction(image, truth, pred):
    """
    พล็อตภาพเปรียบเทียบ MRI ต้นฉบับ / Ground Truth / AI Prediction
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    flair_img = image[3, :, :]  # Channel 3 = FLAIR (เห็นเนื้องอกชัดที่สุด)

    # ภาพ MRI ต้นฉบับ
    axes[0].imshow(flair_img, cmap='gray')
    axes[0].set_title('Original MRI (FLAIR)', fontsize=14)
    axes[0].axis('off')

    # Ground Truth ทับบน FLAIR (alpha=0.6 คือโปร่งใส 40%)
    axes[1].imshow(flair_img, cmap='gray')
    im1 = axes[1].imshow(truth, cmap='nipy_spectral', alpha=0.6, interpolation='nearest')
    axes[1].set_title('Ground Truth (Doctor)', fontsize=14)
    axes[1].axis('off')

    # AI Prediction ทับบน FLAIR
    axes[2].imshow(flair_img, cmap='gray')
    axes[2].imshow(pred, cmap='nipy_spectral', alpha=0.6, interpolation='nearest')
    axes[2].set_title('AI Prediction (Attention U-Net)', fontsize=14)
    axes[2].axis('off')

    # Legend อธิบายสี
    colors  = [im1.cmap(im1.norm(v)) for v in [0, 1, 2, 3]]
    labels  = ['Background', 'Necrotic Core (NCR)', 'Edema (ED)', 'Enhancing Tumor (ET)']
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(1, 4)]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()


plot_prediction(test_images_cpu[0], test_masks_cpu[0], predicted_masks_cpu[0])


# ==============================================================================
# ส่วนที่ 9: Clinical Evaluation (TP, TN, FP, FN ทั้ง Test Set)
# ==============================================================================
"""
📦 คำอธิบาย:
    ประเมินผลเชิงสถิติทั้งหมดบน Test Set
    
    เราแปลงปัญหาเป็น Binary Classification (มีเนื้องอก / ไม่มี):
    - เนื้องอก: ทุก pixel ที่ label > 0 (Class 1, 2, 3)
    - ปกติ:     pixel ที่ label = 0

📐 สูตรตัวชี้วัด:
    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    Dice/F1   = 2TP / (2TP + FP + FN)
"""
total_TP = total_TN = total_FP = total_FN = 0.0

inference_model.eval()
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks  = masks.to(device)

        outputs = inference_model(images)
        preds   = torch.argmax(outputs, dim=1)

        # แปลงเป็น Binary (0 = ปกติ, 1 = เนื้องอก)
        preds_bin = (preds > 0).float()
        masks_bin = (masks > 0).float()

        total_TP += torch.sum(preds_bin * masks_bin).item()
        total_TN += torch.sum((1 - preds_bin) * (1 - masks_bin)).item()
        total_FP += torch.sum(preds_bin * (1 - masks_bin)).item()
        total_FN += torch.sum((1 - preds_bin) * masks_bin).item()

smooth    = 1e-5
Accuracy  = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN + smooth)
Precision = total_TP / (total_TP + total_FP + smooth)
Recall    = total_TP / (total_TP + total_FN + smooth)
Dice_F1   = (2 * total_TP) / (2 * total_TP + total_FP + total_FN + smooth)

print(f"✅ Accuracy  : {Accuracy  * 100:.2f}%")
print(f"🎯 Precision : {Precision * 100:.2f}%")
print(f"🔎 Recall    : {Recall    * 100:.2f}%")
print(f"⭐ Dice/F1   : {Dice_F1   * 100:.2f}%")


# ==============================================================================
# ส่วนที่ 10: Advanced Training (30 Epochs + Dice-CE Loss + LR Scheduler)
# ==============================================================================
"""
📦 คำอธิบาย:
    อัปเกรดการเทรนด้วย 3 เทคนิคใหม่:

    1. Dice-CE Loss (Combined Loss):
       = (0.5 × CrossEntropy) + (0.5 × DiceLoss)
       
       CrossEntropy: คำนวณทุกคลาสรวมถึง Background
       → ป้องกัน False Positive (ไม่ให้ตีความสีฟ้าล้นออกนอกเนื้องอก)
       
       DiceLoss: โฟกัสเฉพาะเนื้องอก
       → เพิ่ม Overlap ให้ขอบเขตแม่นยำ

    2. CosineAnnealingLR Scheduler:
       ลด Learning Rate แบบโค้งไซน์จาก lr_max ถึง lr_min
       → ให้โมเดลเรียนรู้หยาบๆ ก่อน แล้วค่อยๆ ละเอียดขึ้น
       → ป้องกันการกระโดดข้าม Optima ที่ดี

    3. tqdm Progress Bar:
       แสดงความคืบหน้าแบบ Real-time พร้อมค่า Loss และ LR ปัจจุบัน
"""
from tqdm.auto import tqdm

class DiceCELoss(nn.Module):
    """
    Combined Loss = CrossEntropy + DiceLoss
    ผสมจุดแข็งของทั้งสองเข้าด้วยกัน
    """
    def __init__(self, weight_ce=0.5, weight_dice=0.5):
        super(DiceCELoss, self).__init__()
        self.weight_ce   = weight_ce
        self.weight_dice = weight_dice
        self.ce          = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # Cross Entropy คิดทุกคลาสรวม Background
        ce_loss = self.ce(logits, targets)

        # Dice Loss เฉพาะคลาสเนื้องอก (1, 2, 3)
        probs          = F.softmax(logits, dim=1)
        num_classes    = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dice_loss = 0.0
        smooth    = 1e-5
        for c in range(1, num_classes):
            prob_c   = probs[:, c]
            target_c = targets_one_hot[:, c]
            inter    = torch.sum(prob_c * target_c)
            card     = torch.sum(prob_c + target_c)
            dice_loss += 1.0 - (2. * inter + smooth) / (card + smooth)

        dice_loss = dice_loss / (num_classes - 1)

        return (self.weight_ce * ce_loss) + (self.weight_dice * dice_loss)


# Setup สำหรับ Advanced Training
criterion_advanced = DiceCELoss(weight_ce=0.5, weight_dice=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)

advanced_epochs = 30
scheduler       = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=advanced_epochs, eta_min=1e-6
)

best_advanced_dice = 0.0
history = {'train_loss': [], 'val_loss': [], 'val_dice': []}

print("🔥 เริ่ม Advanced Training (Dice-CE + Scheduler)...")

for epoch in range(advanced_epochs):
    # --- Train ---
    model.train()
    train_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{advanced_epochs} [Train]", leave=False)

    for images, masks in train_pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion_advanced(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_pbar.set_postfix({'Loss': f"{loss.item():.4f}",
                                'LR':   f"{optimizer.param_groups[0]['lr']:.6f}"})

    scheduler.step()  # ลด LR ตาม Cosine Schedule
    avg_train_loss = train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = val_dice = 0.0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{advanced_epochs} [Val]", leave=False)

    with torch.no_grad():
        for images, masks in val_pbar:
            images, masks = images.to(device), masks.to(device)
            outputs  = model(images)
            loss     = criterion_advanced(outputs, masks)
            val_loss += loss.item()
            preds    = torch.argmax(outputs, dim=1)
            val_dice += calculate_metrics(preds, masks)['Dice']

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_dice'].append(avg_val_dice)

    print(f"📊 Epoch {epoch+1}/{advanced_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f}")

    if avg_val_dice > best_advanced_dice:
        best_advanced_dice = avg_val_dice
        torch.save(model.state_dict(), '/kaggle/working/best_attention_unet_advanced.pth')
        print(f"   🏆 New Best! ({best_advanced_dice:.4f})")

print(f"\n🎉 Advanced Training จบ! Best Val Dice = {best_advanced_dice:.4f}")


# ==============================================================================
# ส่วนที่ 11: Ultimate Training (100 Epochs + Augmentation + AMP + Focal Loss)
# ==============================================================================
"""
📦 คำอธิบาย:
    การเทรนระดับสูงสุด ใช้ทุกเทคนิคที่มีเพื่อผลักดัน Dice Score ให้สูงสุด

🚀 เทคนิคที่เพิ่มใหม่:

    1. Data Augmentation (Real-time):
       สุ่มพลิก/หมุนภาพในขณะเทรน ไม่ต้องสร้างภาพใหม่ล่วงหน้า
       - Horizontal Flip (50% โอกาส)
       - Vertical Flip   (50% โอกาส)
       - Rotation 90/180/270 องศา (50% โอกาส)
       → โมเดลเห็นเนื้องอกในมุมหลากหลาย ป้องกัน Overfitting

    2. Dice-Focal Loss:
       Focal Loss = -α × (1-p)^γ × log(p)
       (1-p)^γ คือ "modulating factor" ที่ลดน้ำหนัก pixel ที่ทายถูกง่ายๆ
       → บังคับโมเดลให้โฟกัสเฉพาะ pixel ที่ยากและทายผิดบ่อย
       → ลด FP ในพื้นที่ว่างๆ ที่โมเดล "ชินกับการทายถูก"

    3. Mixed Precision Training (AMP):
       ใช้ Float16 แทน Float32 ในบางส่วนของการคำนวณ
       - ความเร็วเพิ่ม 1.5-2 เท่า
       - Memory ลดลง 50%
       GradScaler ป้องกัน Gradient Underflow จาก Float16

    4. Optimized DataLoader:
       num_workers=4: ใช้ CPU 4 core โหลดข้อมูลขนานกัน
       pin_memory=True: จอง RAM พิเศษเพื่อส่งข้อมูลไป GPU เร็วขึ้น
"""
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp


def apply_augmentation(images, masks):
    """
    Data Augmentation แบบ On-the-fly
    ทำทั้ง Image และ Mask พร้อมกัน เพื่อให้ยังคง Alignment
    
    Args:
        images: (Batch, 4, H, W)
        masks:  (Batch, H, W)
    Returns:
        images, masks ที่ผ่านการ Augment
    """
    for i in range(images.size(0)):
        if random.random() > 0.5:  # 50% โอกาสพลิกซ้าย-ขวา
            images[i] = TF.hflip(images[i])
            masks[i]  = TF.hflip(masks[i])

        if random.random() > 0.5:  # 50% โอกาสพลิกบน-ล่าง
            images[i] = TF.vflip(images[i])
            masks[i]  = TF.vflip(masks[i])

        if random.random() > 0.5:  # 50% โอกาสหมุน
            angle    = random.choice([90, 180, 270])
            images[i] = TF.rotate(images[i], angle)
            # Mask ต้องใช้ INTER_NEAREST เพื่อไม่ให้ Label เบลอ
            masks[i] = TF.rotate(
                masks[i].unsqueeze(0).float(), angle,
                interpolation=TF.InterpolationMode.NEAREST
            ).squeeze(0).long()
    return images, masks


class DiceFocalLoss(nn.Module):
    """
    Combined Loss = FocalLoss + DiceLoss
    
    Focal Loss โฟกัส Hard Examples, Dice Loss โฟกัสขอบเขตเนื้องอก
    
    Args:
        alpha: น้ำหนักสำหรับ Focal Loss (ปกติ 0.25)
        gamma: Focusing parameter (ปกติ 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0, weight_dice=0.5, weight_focal=0.5):
        super(DiceFocalLoss, self).__init__()
        self.alpha   = alpha
        self.gamma   = gamma
        self.w_d     = weight_dice
        self.w_f     = weight_focal

    def forward(self, logits, targets):
        # Focal Loss = α × (1-p)^γ × CrossEntropy
        ce_loss   = F.cross_entropy(logits, targets, reduction='none')
        pt         = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        # Dice Loss (เฉพาะคลาสเนื้องอก)
        probs           = F.softmax(logits, dim=1)
        targets_one_hot  = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()
        dice_loss, smooth = 0.0, 1e-5

        for c in range(1, logits.shape[1]):
            p_c   = probs[:, c]
            t_c   = targets_one_hot[:, c]
            inter = torch.sum(p_c * t_c)
            card  = torch.sum(p_c + t_c)
            dice_loss += 1.0 - (2. * inter + smooth) / (card + smooth)

        dice_loss = dice_loss / 3.0
        return (self.w_f * focal_loss) + (self.w_d * dice_loss)


# สร้างโมเดลใหม่สำหรับเทรน 100 Epochs
ultimate_model = AttentionUNet(img_ch=4, num_classes=4)
if torch.cuda.device_count() > 1:
    ultimate_model = nn.DataParallel(ultimate_model)
ultimate_model = ultimate_model.to(device)

# Optimized DataLoader (num_workers=4, pin_memory=True)
opt_train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True
)
opt_val_loader = DataLoader(
    val_dataset,   batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)

criterion_ultimate = DiceFocalLoss()
optimizer          = torch.optim.AdamW(ultimate_model.parameters(), lr=2e-4, weight_decay=1e-5)
scheduler          = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
scaler             = amp.GradScaler()  # สำหรับ Mixed Precision Training

ultimate_epochs    = 100
ultimate_best_dice = 0.0
history            = {'train_loss': [], 'val_loss': [], 'val_dice': []}

print("⚡ เริ่ม Ultimate Training (100 Epochs + AMP + Augmentation + Focal Loss)!")

for epoch in range(ultimate_epochs):
    # --- Train Phase (พร้อม AMP) ---
    ultimate_model.train()
    train_loss = 0.0
    train_pbar = tqdm(opt_train_loader, desc=f"Epoch {epoch+1}/{ultimate_epochs} [Train]", leave=False)

    for images, masks in train_pbar:
        images, masks = apply_augmentation(images, masks)  # Random Augmentation
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision: คำนวณด้วย Float16
        with amp.autocast():
            outputs = ultimate_model(images)
            loss    = criterion_ultimate(outputs, masks)

        scaler.scale(loss).backward()  # Scale Gradient เพื่อป้องกัน Underflow
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    avg_train_loss = train_loss / len(opt_train_loader)
    scheduler.step()

    # --- Validation Phase ---
    ultimate_model.eval()
    val_loss = val_dice = 0.0
    val_pbar = tqdm(opt_val_loader, desc=f"Epoch {epoch+1}/{ultimate_epochs} [Val]", leave=False)

    with torch.no_grad():
        for images, masks in val_pbar:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with amp.autocast():
                outputs = ultimate_model(images)
                loss    = criterion_ultimate(outputs, masks)

            val_loss += loss.item()
            preds     = torch.argmax(outputs, dim=1)
            val_dice  += calculate_metrics(preds, masks)['Dice']

    avg_val_loss = val_loss / len(opt_val_loader)
    avg_val_dice = val_dice / len(opt_val_loader)

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_dice'].append(avg_val_dice)

    print(f"📊 Epoch {epoch+1:03d}/{ultimate_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f}")

    if avg_val_dice > ultimate_best_dice:
        ultimate_best_dice = avg_val_dice
        torch.save(ultimate_model.state_dict(), '/kaggle/working/best_attention_unet_100ep.pth')
        print(f"   ⭐ New SOTA! ({ultimate_best_dice:.4f}) - Saved!")

print(f"\n🏆 Ultimate Training จบ! Best Dice = {ultimate_best_dice * 100:.2f}%")


# ==============================================================================
# ส่วนที่ 12: Learning Curves และ Confusion Matrix
# ==============================================================================
"""
📦 คำอธิบาย:
    วิเคราะห์ผลลัพธ์เชิงลึกด้วยกราฟ 2 ประเภท:

    1. Learning Curves (กราฟการเรียนรู้):
       - Train Loss vs Val Loss → ดูว่า Overfit หรือไม่
         * ถ้า Train Loss ต่ำ แต่ Val Loss สูง = Overfit (จำแต่ Training data)
         * ถ้าทั้งคู่ลดพร้อมกัน = กำลังเรียนรู้ดี
       - Val Dice Score → ดูความก้าวหน้าของความแม่นยำ

    2. Pixel-wise Confusion Matrix:
       แสดงว่า AI สับสนระหว่างคลาสไหนมากที่สุด
       แกน Y = Ground Truth จริง, แกน X = AI Prediction
       
       เช่น ช่อง (2, 3) สูง = AI มักทาย Edema เป็น Enhancing Tumor
       
       ใช้ Row-Normalization เพื่อแปลงเป็นเปอร์เซ็นต์
       → อ่านง่ายขึ้น แม้แต่ละคลาสมีจำนวน pixel ต่างกันมาก
"""
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- พล็อต Learning Curves ---
epochs_range = range(1, ultimate_epochs + 1)

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], label='Train Loss',      color='blue',  linewidth=2)
plt.plot(epochs_range, history['val_loss'],   label='Validation Loss', color='red',   linewidth=2)
plt.title('Training and Validation Loss', fontsize=16)
plt.xlabel('Epochs')
plt.ylabel('Dice-Focal Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['val_dice'], label='Val Dice Score', color='green', linewidth=2)
plt.title('Validation Dice Score over 100 Epochs', fontsize=16)
plt.xlabel('Epochs')
plt.ylabel('Dice Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# --- สร้าง Confusion Matrix ---
print("🧮 สร้าง Pixel-wise Confusion Matrix จาก Test Set...")

# โหลดโมเดล 100 Epochs ที่ดีที่สุด
ultimate_model.load_state_dict(torch.load('/kaggle/working/best_attention_unet_100ep.pth'))
ultimate_model.eval()

cm       = np.zeros((4, 4), dtype=np.int64)
test_pbar = tqdm(test_loader, desc="Generating Confusion Matrix")

with torch.no_grad():
    for images, masks in test_pbar:
        images = images.to(device)
        outputs = ultimate_model(images)
        preds   = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
        trues   = masks.numpy().flatten()
        cm     += confusion_matrix(trues, preds, labels=[0, 1, 2, 3])

# แปลงเป็น % (Row Normalized)
cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
class_names = ['Background (0)', 'Necrotic Core (1)', 'Edema (2)', 'Enhancing Tumor (3)']

sns.heatmap(cm_pct, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 12})

plt.title('Pixel-wise Confusion Matrix (Test Set)', fontsize=16)
plt.xlabel('AI Prediction', fontsize=14)
plt.ylabel('Ground Truth (แพทย์)', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("✅ วิเคราะห์ผลลัพธ์เสร็จสมบูรณ์!")
print("📌 กราฟ Learning Curves และ Confusion Matrix พร้อมใส่ในรายงานโครงงาน")
