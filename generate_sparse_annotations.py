import json
import random
import os
import numpy as np
from collections import defaultdict

def create_sparse_annotations(input_json, output_json, labelled_percent, drop_percent):
    """
    生成稀疏标注文件
    Args:
        input_json: 输入的完整标注文件路径
        output_json: 输出的稀疏标注文件路径
        labelled_percent: 保留标签的图片百分比 (0-100)
        drop_percent: 对于有标签的图片，随机丢弃标注的百分比 (0-100)
    """
    # 读取原始标注文件
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # 按图片ID组织标注
    image_to_anns = defaultdict(list)
    for ann in data['annotations']:
        image_to_anns[ann['image_id']].append(ann)
    
    # 随机选择保留标签的图片
    all_image_ids = list(image_to_anns.keys())
    num_labelled = int(len(all_image_ids) * labelled_percent / 100)
    random.seed(42)  # 保持可重复性
    labelled_image_ids = set(random.sample(all_image_ids, num_labelled))
    
    # 创建新的标注列表
    new_annotations = []
    new_images = []
    
    # 处理每张图片
    for img in data['images']:
        img_id = img['id']
        if img_id in labelled_image_ids:
            # 对于有标签的图片，随机丢弃一部分标注
            anns = image_to_anns[img_id]
            num_keep = int(len(anns) * (100 - drop_percent) / 100)
            kept_anns = random.sample(anns, max(1, num_keep))  # 至少保留一个标注
            new_annotations.extend(kept_anns)
            new_images.append(img)
        else:
            # 对于无标签的图片，保留图片信息但不保留标注
            new_images.append(img)
    
    # 创建新的数据集
    new_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': data['categories']
    }
    
    # 保存新的标注文件
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(new_data, f)
    
    # 统计每个类别的标注数量
    class_counts = defaultdict(int)
    for ann in new_annotations:
        class_counts[ann['category_id']] += 1
    
    print(f"\n生成稀疏标注文件: {output_json}")
    print(f"保留标签的图片比例: {labelled_percent}%")
    print(f"标注丢弃比例: {drop_percent}%")
    print("\n每个类别的标注数量:")
    for cat in data['categories']:
        cat_id = cat['id']
        print(f"类别 {cat['name']}: {class_counts[cat_id]}")
    
    return list(class_counts.values())

def main():
    # 原始标注文件 - 修改为RUDO数据集路径
    input_json = "data/RUDO/annotations/instances_train.json"
    
    # 生成三种不同稀疏度的标注
    configs = [
        ("instances_train_sparse_50_30.json", 50, 30),  # 50%图片保留标签，丢弃30%标注
        ("instances_train_sparse_30_50.json", 30, 50),  # 30%图片保留标签，丢弃50%标注
        ("instances_train_sparse_80_10.json", 80, 10),  # 80%图片保留标签，丢弃10%标注
    ]
    
    class_counts_all = {}
    for filename, label_percent, drop_percent in configs:
        output_json = os.path.join("data/RUDO/annotations", filename)
        class_counts = create_sparse_annotations(input_json, output_json, 
                                              label_percent, drop_percent)
        class_counts_all[filename] = class_counts
    
    # 生成配置文件的内容
    print("\n配置文件的CLASS_COUNTS设置:")
    for filename, counts in class_counts_all.items():
        print(f"\n{filename}:")
        print(f"CLASS_COUNTS: {counts}")

if __name__ == "__main__":
    main() 