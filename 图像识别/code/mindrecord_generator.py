import os
import sys
from mindspore.mindrecord import FileWriter
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# === 核心配置 ===
# 原始数据集根路径
RAW_DATA_ROOT = "/home/developer/lzh/potato/data_Potato"
# 输出的MindRecord文件路径
OUTPUT_DIR = "/home/developer/lzh/potato/data_Potato_mindrecord"
MINDRECORD_TRAIN = os.path.join(OUTPUT_DIR, "Potato_train.mindrecord")
MINDRECORD_VAL = os.path.join(OUTPUT_DIR, "Potato_val.mindrecord")
MINDRECORD_TEST = os.path.join(OUTPUT_DIR, "Potato_test.mindrecord")

# 数据集划分比例
TRAIN_RATIO = 0.7    # 70% 训练集
VAL_RATIO = 0.15     # 15% 验证集
TEST_RATIO = 0.15    # 15% 测试集

# 支持的图片格式
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')
# ========================

def is_valid_image(file_path):
    """验证图片文件是否有效"""
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证图片完整性
        return True
    except Exception as e:
        print(f"⚠️ 无效图片 {file_path}: {str(e)[:100]}")
        return False

def split_dataset_by_class(data_root, train_ratio=0.7, val_ratio=0.15):
    """按类别划分数据集"""
    print("正在分析和划分数据集...")
    
    # 获取所有类别
    classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    if not classes:
        print(f"❌ 错误：在 {data_root} 下未找到任何类别文件夹")
        sys.exit(1)
    
    print(f"✅ 发现 {len(classes)} 个类别: {classes}")
    
    train_files = []
    val_files = []
    test_files = []
    class_distribution = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})
    
    for cls_name in classes:
        cls_path = os.path.join(data_root, cls_name)
        # 获取该类别下的所有有效图片
        images = []
        for img_name in os.listdir(cls_path):
            if img_name.lower().endswith(SUPPORTED_FORMATS):
                img_path = os.path.join(cls_path, img_name)
                if is_valid_image(img_path):
                    images.append(img_path)
        
        num_images = len(images)
        if num_images == 0:
            print(f"⚠️ 警告：类别 {cls_name} 下没有有效图片")
            continue
        
        class_distribution[cls_name]['total'] = num_images
        
        # 计算划分数量
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)
        num_test = num_images - num_train - num_val
        
        # 更新分布统计
        class_distribution[cls_name]['train'] = num_train
        class_distribution[cls_name]['val'] = num_val
        class_distribution[cls_name]['test'] = num_test
        
        # 划分数据
        train_files.extend([(img_path, cls_name) for img_path in images[:num_train]])
        val_files.extend([(img_path, cls_name) for img_path in images[num_train:num_train+num_val]])
        test_files.extend([(img_path, cls_name) for img_path in images[num_train+num_val:]])
    
    # 打印数据分布
    print(f"\n数据集分布统计:")
    print(f"{'类别':<15} {'训练集':<8} {'验证集':<8} {'测试集':<8} {'总计':<8}")
    print("-" * 55)
    total_train, total_val, total_test = 0, 0, 0
    for cls_name in sorted(class_distribution.keys()):
        dist = class_distribution[cls_name]
        print(f"{cls_name:<15} {dist['train']:<8} {dist['val']:<8} {dist['test']:<8} {dist['total']:<8}")
        total_train += dist['train']
        total_val += dist['val']
        total_test += dist['test']
    
    total = total_train + total_val + total_test
    print("-" * 55)
    print(f"{'总计':<15} {total_train:<8} {total_val:<8} {total_test:<8} {total:<8}")
    print(f"\n划分比例: 训练集 {total_train/total*100:.1f}% | 验证集 {total_val/total*100:.1f}% | 测试集 {total_test/total*100:.1f}%")
    
    return train_files, val_files, test_files, classes

def create_mindrecord(subset, file_list, class_to_idx, output_file):
    """
    为单个子集（train/val/test）生成MindRecord
    :param subset: 子集名称（train/val/test）
    :param file_list: 该子集的文件列表
    :param class_to_idx: 类别到索引的映射
    :param output_file: 生成的MindRecord文件名
    """
    print(f"\n{'='*60}")
    print(f"开始生成 {subset} 集的MindRecord")
    print(f"{'='*60}")
    
    # 1. 定义MindRecord数据结构
    schema = {
        "file_name": {"type": "string"},
        "label": {"type": "int32"}, 
        "data": {"type": "bytes"}
    }
    
    # 初始化Writer（覆盖已存在的文件）
    try:
        writer = FileWriter(file_name=output_file, shard_num=1, overwrite=True)
        writer.add_schema(schema, "corn_dataset")
    except Exception as e:
        print(f"❌ 创建MindRecord写入器失败: {e}")
        return False
    
    # 2. 遍历图片并写入
    data_list = []
    count = 0
    failed_count = 0
    
    with tqdm(total=len(file_list), desc=f"处理 {subset} 集") as pbar:
        for img_path, cls_name in file_list:
            try:
                # 读取图片二进制数据
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                
                # 添加到数据列表
                data_list.append({
                    "file_name": os.path.basename(img_path),
                    "label": class_to_idx[cls_name],
                    "data": img_bytes
                })
                count += 1
                
                # 每1000张批量写入
                if count % 1000 == 0:
                    writer.write_raw_data(data_list)
                    data_list = []
                    pbar.set_postfix({"已写入": count, "失败": failed_count})
            
            except Exception as e:
                print(f"\n⚠️ 处理图片 {img_path} 失败: {str(e)[:100]}")
                failed_count += 1
            
            pbar.update(1)
    
    # 写入剩余的图片
    if data_list:
        writer.write_raw_data(data_list)
    
    # 完成写入并关闭
    try:
        writer.commit()
        print(f"\n✅ {subset}集MindRecord生成完成！")
        print(f"统计信息:")
        print(f"   - 总文件数: {len(file_list)}")
        print(f"   - 成功写入: {count}")
        print(f"   - 失败数量: {failed_count}")
        print(f"   - 成功率: {count/(count+failed_count)*100:.1f}%" if (count+failed_count) > 0 else "   - 无数据")
        print(f"文件保存路径: {output_file}")
        if os.path.exists(f"{output_file}.db"):
            print(f"文件保存路径: {output_file}.db")
        return True
    except Exception as e:
        print(f"❌ 提交MindRecord失败: {e}")
        return False

def main():
    """主函数：一键运行数据处理流程"""
    print("玉米数据集MindRecord生成工具")
    print("="*50)
    
    # 1. 检查输入目录
    if not os.path.exists(RAW_DATA_ROOT):
        print(f"❌ 错误：数据集路径不存在 → {RAW_DATA_ROOT}")
        print(f"   请检查路径是否正确，或修改 RAW_DATA_ROOT 变量")
        sys.exit(1)
    
    # 2. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3. 划分数据集
    try:
        train_files, val_files, test_files, classes = split_dataset_by_class(
            RAW_DATA_ROOT, TRAIN_RATIO, VAL_RATIO
        )
    except Exception as e:
        print(f"❌ 数据集划分失败: {e}")
        sys.exit(1)
    
    # 4. 创建类别映射
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(classes))}
    print(f"\n类别映射: {class_to_idx}")
    
    # 5. 生成各个子集的MindRecord
    success = True
    
    # 生成训练集
    if not create_mindrecord("train", train_files, class_to_idx, MINDRECORD_TRAIN):
        success = False
    
    # 生成验证集
    if not create_mindrecord("val", val_files, class_to_idx, MINDRECORD_VAL):
        success = False
    
    # 生成测试集
    if not create_mindrecord("test", test_files, class_to_idx, MINDRECORD_TEST):
        success = False
    
    # 6. 总结
    print(f"\n{'='*60}")
    if success:
        print("所有MindRecord文件生成完成！")
        print(f"\n输出文件列表:")
        print(f"   - 训练集: {MINDRECORD_TRAIN}")
        print(f"   - 验证集: {MINDRECORD_VAL}") 
        print(f"   - 测试集: {MINDRECORD_TEST}")
        print(f"\n数据集统计:")
        print(f"   - 训练集样本数: {len(train_files)}")
        print(f"   - 验证集样本数: {len(val_files)}")
        print(f"   - 测试集样本数: {len(test_files)}")
        print(f"   - 总样本数: {len(train_files) + len(val_files) + len(test_files)}")
        print(f"   - 类别数: {len(classes)}")
    else:
        print("⚠️ 部分MindRecord文件生成失败，请检查错误信息")
    
    print(f"\n提示: 生成的MindRecord文件可直接用于MindSpore模型训练")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n程序结束")