#!/usr/bin/env python3
"""
测试数据下载脚本

提供多种医学影像数据集的下载链接和下载功能
"""

import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import subprocess

def download_file(url, destination):
    """
    下载文件到指定位置

    参数:
        url (str): 下载链接
        destination (str): 保存路径
    """
    try:
        print(f"正在下载: {url}")
        urllib.request.urlretrieve(url, destination)
        print(f"下载完成: {destination}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def extract_archive(archive_path, extract_to):
    """
    解压缩文件

    参数:
        archive_path (str): 压缩文件路径
        extract_to (str): 解压目录
    """
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
        print(f"解压完成: {extract_to}")
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False

def create_data_directory():
    """创建数据目录"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def download_synthetic_data():
    """下载/生成合成数据"""
    print("\n" + "="*50)
    print("1. 生成合成CT数据")
    print("="*50)

    data_dir = create_data_directory()

    # 运行主程序生成合成数据
    try:
        from main import generate_synthetic_ct_data
        import numpy as np

        # 生成合成数据并保存
        synthetic_ct = generate_synthetic_ct_data(shape=(256, 256, 128), noise_level=0.05)

        save_path = data_dir / "synthetic_chest_ct.npy"
        np.save(save_path, synthetic_ct)

        print(f"合成数据已保存至: {save_path}")
        print(f"数据形状: {synthetic_ct.shape}")
        print(f"HU值范围: [{np.min(synthetic_ct):.1f}, {np.max(synthetic_ct):.1f}]")

        return True

    except Exception as e:
        print(f"生成合成数据失败: {e}")
        return False

def download_sample_dicom_data():
    """下载示例DICOM数据"""
    print("\n" + "="*50)
    print("2. 下载示例DICOM数据")
    print("="*50)

    data_dir = create_data_directory()
    dicom_dir = data_dir / "sample_dicom"
    dicom_dir.mkdir(exist_ok=True)

    # 示例DICOM数据链接（公开可用的小型数据集）
    datasets = [
        {
            "name": "OsiriX DICOM示例",
            "url": "https://github.com/OsiriX-Foundation/DICOM-samples/archive/refs/heads/master.zip",
            "description": "多种类型的DICOM示例文件",
            "size": "约50MB"
        },
        {
            "name": "TCIA Chest CT示例",
            "url": "https://wiki.cancerimagingarchive.net/download/attachments/22515229/CT-CHB-001.zip?version=1&modificationDate=1470748967758&api=v2",
            "description": "胸部CT扫描示例",
            "size": "约30MB"
        }
    ]

    print("可选的数据集:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset['name']}")
        print(f"   描述: {dataset['description']}")
        print(f"   大小: {dataset['size']}")
        print(f"   链接: {dataset['url']}")
        print()

    # 由于这些链接可能需要手动下载，提供手动下载指导
    print("手动下载指导:")
    print("1. 访问上述链接下载数据")
    print("2. 将下载的文件放入 'data/' 目录")
    print("3. 如果是压缩文件，会自动解压")
    print("4. 或者使用curl/wget命令下载")
    print()

    # 生成下载脚本
    script_content = """#!/bin/bash
# 数据下载脚本

echo "开始下载示例DICOM数据..."

# 创建目录
mkdir -p data/sample_dicom

# 下载OsiriX DICOM示例
echo "下载OsiriX DICOM示例..."
curl -L -o data/dicom_samples.zip "https://github.com/OsiriX-Foundation/DICOM-samples/archive/refs/heads/master.zip"

# 解压文件
echo "解压文件..."
cd data
unzip -q dicom_samples.zip
mv DICOM-samples-master/* sample_dicom/
rm -rf dicom_samples.zip DICOM-samples-master

echo "下载完成！"
"""

    script_path = data_dir / "download_dicom.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    print(f"已生成下载脚本: {script_path}")
    print("运行命令: bash download/download_dicom.sh")

def download_medical_decathon_sample():
    """下载Medical Decathlon示例数据"""
    print("\n" + "="*50)
    print("3. Medical Segmentation Decathlon示例数据")
    print("="*50)

    data_dir = create_data_directory()
    decathlon_dir = data_dir / "medical_decathlon"
    decathlon_dir.mkdir(exist_ok=True)

    print("Medical Decathlon数据集信息:")
    print("- 网址: http://medicaldecathlon.com/")
    print("- 描述: 医学影像分割挑战赛数据集")
    print("- 包含: 10个不同的医学影像分割任务")
    print("- 格式: NIfTI (.nii.gz)")
    print()

    # 生成Python下载脚本
    script_content = '''#!/usr/bin/env python3
"""
Medical Decathlon数据下载脚本
需要先注册账号获取下载链接
"""

import os
import requests
from pathlib import Path

def download_decathlon_data(task_id, save_dir):
    """
    下载指定任务的数据

    参数:
        task_id (str): 任务ID (如: Task01_BrainTumour)
        save_dir (str): 保存目录
    """
    # 这里需要实际的下载链接，需要先注册获取
    url = f"https://drive.google.com/uc?export=download&id=YOUR_DOWNLOAD_ID_FOR_{task_id}"

    try:
        print(f"下载 {task_id} 数据...")
        response = requests.get(url, stream=True)

        filename = f"{task_id}.tar"
        filepath = Path(save_dir) / filename

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"下载完成: {filepath}")
        return filepath

    except Exception as e:
        print(f"下载失败: {e}")
        return None

if __name__ == "__main__":
    # 使用示例
    data_dir = "data/medical_decathlon"

    tasks = [
        "Task01_BrainTumour",
        "Task03_Liver",
        "Task07_Colon",
        "Task09_Spleen"
    ]

    print("请先在 http://medicaldecathlon.com/ 注册获取下载链接")
    print("然后将下载链接替换到脚本中的YOUR_DOWNLOAD_ID")
'''

    script_path = data_dir / "download_decathlon.py"
    with open(script_path, 'w') as f:
        f.write(script_content)

    print(f"已生成下载脚本模板: {script_path}")
    print("需要先注册账号获取下载链接")

def download_kaggle_datasets():
    """下载Kaggle数据集"""
    print("\n" + "="*50)
    print("4. Kaggle医学影像数据集")
    print("="*50)

    print("Kaggle上的相关数据集:")
    print()

    datasets = [
        {
            "name": "RSNA Pneumonia Detection",
            "url": "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge",
            "description": "胸部X光肺炎检测",
            "type": "X-ray"
        },
        {
            "name": "RSNA Intracranial Hemorrhage Detection",
            "url": "https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection",
            "description": "脑出血检测CT",
            "type": "CT"
        },
        {
            "name": "Brain Tumor Segmentation",
            "url": "https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification",
            "description": "脑肿瘤分割MRI",
            "type": "MRI"
        },
        {
            "name": "Chest X-Ray Images (NIH)",
            "url": "https://www.kaggle.com/nih-chest-xrays/data",
            "description": "NIH胸部X光数据集",
            "type": "X-ray"
        }
    ]

    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset['name']}")
        print(f"   类型: {dataset['type']}")
        print(f"   描述: {dataset['description']}")
        print(f"   链接: {dataset['url']}")
        print()

    # 生成Kaggle下载指导
    print("Kaggle数据下载步骤:")
    print("1. 安装Kaggle API: pip install kaggle")
    print("2. 获取Kaggle API密钥: https://www.kaggle.com/account")
    print("3. 配置API密钥: mkdir ~/.kaggle && cp kaggle.json ~/.kaggle/")
    print("4. 下载数据集: kaggle competitions download -c competition-name")
    print()

    # 生成示例下载命令
    commands = """
# 示例下载命令
kaggle competitions download -c rsna-pneumonia-detection-challenge
kaggle competitions download -c rsna-intracranial-hemorrhage-detection
kaggle datasets download -d nih-chest-xrays/data
"""

    commands_path = "data/kaggle_commands.txt"
    with open(commands_path, 'w') as f:
        f.write(commands)

    print(f"已保存Kaggle下载命令: {commands_path}")

def provide_manual_alternatives():
    """提供手动数据获取方法"""
    print("\n" + "="*50)
    print("5. 手动数据获取方法")
    print("="*50)

    alternatives = [
        {
            "name": "使用PyDicom创建测试数据",
            "description": "使用PyDicom库创建模拟DICOM文件",
            "code": """
import pydicom
import numpy as np
from pydicom.dataset import Dataset, FileDataset

# 创建模拟DICOM文件
def create_test_dicom():
    # 创建基本数据集
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = '1.2.3'
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # 创建图像数据
    image_data = np.random.randint(-1000, 1000, (512, 512))

    # 创建数据集
    ds = Dataset()
    ds.file_meta = file_meta
    ds.PixelData = image_data.tobytes()
    ds.Rows, ds.Columns = image_data.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024

    # 保存文件
    ds.save_as('test_ct.dcm')
"""
        },
        {
            "name": "使用现有numpy数组",
            "description": "直接使用numpy数组作为测试数据",
            "code": """
import numpy as np

# 创建测试用的3D CT数据
def create_test_ct():
    shape = (128, 128, 64)  # (height, width, depth)

    # 模拟CT数据
    ct_data = np.random.normal(0, 100, shape)

    # 添加一些结构
    ct_data[20:40, 20:40, :] = 40    # 软组织
    ct_data[60:80, 60:80, :] = 800   # 骨骼
    ct_data[:10, :, :] = -1000       # 空气

    return ct_data

# 使用数据
ct_image = create_test_ct()
np.save('test_ct.npy', ct_image)
"""
        }
    ]

    for i, alt in enumerate(alternatives, 1):
        print(f"{i}. {alt['name']}")
        print(f"   {alt['description']}")
        print("   代码示例:")
        print("   " + alt['code'].replace('\n', '\n   '))
        print()

def main():
    """主函数：提供数据下载选项"""
    print("医学影像测试数据下载工具")
    print("="*60)

    while True:
        print("\n请选择数据获取方式:")
        print("1. 生成合成CT数据 (推荐)")
        print("2. 下载示例DICOM数据")
        print("3. Medical Decathlon数据集")
        print("4. Kaggle数据集")
        print("5. 手动获取方法指导")
        print("0. 退出")

        choice = input("\n请输入选择 (0-5): ").strip()

        if choice == '1':
            download_synthetic_data()
        elif choice == '2':
            download_sample_dicom_data()
        elif choice == '3':
            download_medical_decathon_sample()
        elif choice == '4':
            download_kaggle_datasets()
        elif choice == '5':
            provide_manual_alternatives()
        elif choice == '0':
            print("退出下载工具")
            break
        else:
            print("无效选择，请重试")

    print("\n数据获取完成！")
    print("数据文件保存在 'data/' 目录中")
    print("现在可以运行 main.py 和 test.py 进行测试")

if __name__ == "__main__":
    main()