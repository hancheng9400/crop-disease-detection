import os
import re
import warnings
warnings.filterwarnings("ignore")  # 屏蔽无关警告

import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context, Model, Tensor, save_checkpoint
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import Callback, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindcv.models import create_model
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json  # 导入json库用于保存历史记录

# === Configuration Area ===
# 玉米数据集MindRecord文件路径
TRAIN_MINDRECORD = "/home/developer/Desktop/corn_train.mindrecord"
VAL_MINDRECORD = "/home/developer/Desktop/corn_val.mindrecord"
TEST_MINDRECORD = "/home/developer/Desktop/corn_test.mindrecord"  # 独立测试集
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 4  # 玉米疾病数据集：4个类别（3种疾病+1个健康类别）
LEARNING_RATE = 0.0005
CKPT_DIR = "./ckpt_corn"  # 玉米专用的checkpoint目录
PLOT_DIR = "./plots_corn"  # 训练曲线图保存目录
# ========================

# 确保所有目录存在
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# 自定义验证Callback：跟踪每轮指标+自动判定最佳训练轮次
class ValCallback(Callback):
    def __init__(self, val_dataset, network, loss_fn):
        self.val_dataset = val_dataset
        self.network = network
        self.loss_fn = loss_fn
        self.network.set_train(False)  # 验证时模型设为评估模式
        
        # 核心：初始化跟踪变量（记录每轮指标+最佳轮次）
        self.all_train_loss = []
        self.all_val_loss = []
        self.all_val_acc = []
        self.all_epoch = []
        
        # 最佳轮次判定基准（验证准确率最高优先）
        self.best_val_acc = 0.0    # 最佳验证准确率
        self.best_val_loss = float('inf')  # 最佳验证损失（越小越好）
        self.best_epoch = 0        # 最佳训练轮次
        self.best_model_params = None  # 保存最佳模型参数

    def epoch_end(self, run_context):
        """每个epoch训练完成后：评估验证集+更新最佳轮次"""
        val_loss = 0.0
        correct = 0
        total = 0
        
        # 遍历验证集计算指标
        for data in self.val_dataset.create_dict_iterator():
            images = data["data"]
            labels = data["label"]
            total += labels.shape[0]
            
            # 前向传播
            logits = self.network(images)
            # 计算损失
            loss = self.loss_fn(logits, labels)
            val_loss += loss.asnumpy() * labels.shape[0]  # 按样本数累加损失
            # 计算准确率
            pred = np.argmax(logits.asnumpy(), axis=1)
            correct += (pred == labels.asnumpy()).sum()
        
        # 计算平均指标
        avg_val_loss = val_loss / total  # 平均验证损失
        val_acc = correct / total        # 验证准确率
        
        # 记录当前轮次指标
        cb_params = run_context.original_args()
        current_epoch = cb_params.cur_epoch_num
        self.all_epoch.append(current_epoch)
        self.all_val_loss.append(avg_val_loss)
        self.all_val_acc.append(val_acc)
        
        # 记录训练损失 (注意：此处获取的是当前step的loss，用于曲线展示)
        train_loss = cb_params.net_outputs.asnumpy()
        self.all_train_loss.append(train_loss)
        
        # 核心：更新最佳轮次（优先按验证准确率，准确率相同则选损失更低的）
        if val_acc > self.best_val_acc or (val_acc == self.best_val_acc and avg_val_loss < self.best_val_loss):
            self.best_val_acc = val_acc
            self.best_val_loss = avg_val_loss
            self.best_epoch = current_epoch
            self.best_model_params = self.network.parameters_dict()  # 保存最佳模型参数
            print(f"找到更佳轮次！更新最佳轮次为 Epoch {self.best_epoch}")
        
        # 打印本轮结果+当前最佳轮次
        print(f"\n===== Epoch {current_epoch} 验证集结果 =====")
        print(f"训练损失：{train_loss:.4f} | 验证集损失：{avg_val_loss:.4f}")
        print(f"验证集准确率：{val_acc:.4f}")
        print(f"当前最佳轮次：Epoch {self.best_epoch} → 最佳准确率：{self.best_val_acc:.4f} | 最佳损失：{self.best_val_loss:.4f}")
        print("="*60 + "\n")

    def get_best_epoch_info(self):
        """返回最佳轮次的详细信息（训练结束后调用）"""
        return {
            "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "all_epochs": self.all_epoch,
            "all_train_loss": [float(l) for l in self.all_train_loss], # 转换为float以便json序列化
            "all_val_acc": [float(a) for a in self.all_val_acc],
            "all_val_loss": [float(l) for l in self.all_val_loss]
        }

# 训练曲线绘制函数
def plot_training_curves(best_info, save_dir):
    """
    绘制完整的训练曲线图
    """
    # 提取数据
    epochs = best_info['all_epochs']
    train_losses = best_info['all_train_loss']
    val_losses = best_info['all_val_loss']
    val_accs = best_info['all_val_acc']
    best_epoch = best_info['best_epoch']
    best_val_acc = best_info['best_val_acc']
    best_val_loss = best_info['best_val_loss']
    
    if not epochs or not train_losses or not val_losses or not val_accs:
        print("❌ 没有足够的训练数据来绘制曲线")
        return None
    
    # 设置中文字体（如果需要）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 训练损失 vs 验证损失
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax1.scatter(best_epoch, best_val_loss, color='green', s=100, 
               label=f'Best: {best_val_loss:.4f}', zorder=5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. 验证准确率曲线
    ax2.plot(epochs, val_accs, 'g-', linewidth=2, label='Validation Accuracy')
    ax2.scatter(best_epoch, best_val_acc, color='red', s=100, 
               label=f'Best: {best_val_acc:.4f}', zorder=5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    # 3. 训练损失曲线（单独显示）
    ax3.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training Loss', fontsize=12)
    ax3.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. 验证损失曲线（单独显示）
    ax4.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax4.scatter(best_epoch, best_val_loss, color='green', s=100, 
               label=f'Best: {best_val_loss:.4f}', zorder=5)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation Loss', fontsize=12)
    ax4.set_title('Validation Loss Curve', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f'training_curves_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线图已保存至：{plot_path}")
    return plot_path

# 保存训练历史到文件
def save_training_history(best_info, save_dir):
    """保存训练历史到JSON文件"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = os.path.join(save_dir, f'training_history_{timestamp}.json')
    
    with open(history_file, 'w', encoding='utf-8') as f:
        # 使用json.dump保存字典
        json.dump(best_info, f, indent=4)
    
    print(f"训练历史已保存至：{history_file}")
    return history_file

# 断点续训：获取最新CKPT+对应epoch
def get_latest_ckpt(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None, 0
    
    ckpt_files = [f for f in os.listdir(ckpt_dir) 
                  if f.endswith(".ckpt") and f.startswith("corn_disease")]
    if not ckpt_files:
        return None, 0
    
    def sort_key(file_name):
        # 提取文件中的数字（假设格式为 corn_disease-epoch_X-step_Y.ckpt）
        num_list = list(map(int, re.findall(r'\d+', file_name)))
        epoch = num_list[-2] if len(num_list)>=2 else 0
        step = num_list[-1] if len(num_list)>=1 else 0
        return (epoch, step)
    
    latest_ckpt = sorted(ckpt_files, key=sort_key)[-1]
    latest_ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
    num_list = list(map(int, re.findall(r'\d+', latest_ckpt)))
    latest_epoch = num_list[-2] if len(num_list)>=2 else 0
    return latest_ckpt_path, latest_epoch

# 加载训练集+验证集
def create_train_val_dataset():
    # 检查训练/验证集是否存在
    for path in [TRAIN_MINDRECORD, VAL_MINDRECORD]:
        if not os.path.exists(path):
            print(f"❌ Error: MindRecord文件不存在 → {path}")
            exit()

    # 标准化预处理（MobileNetV3要求）
    transforms_list = [
        vision.Decode(),
        vision.Resize(256),
        vision.CenterCrop(224),
        vision.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        vision.HWC2CHW()
    ]

    # 加载训练集（打乱）
    print("DataLoader: 加载玉米训练集MindRecord...")
    train_dataset = ds.MindDataset(TRAIN_MINDRECORD, columns_list=["data", "label"], shuffle=True)
    train_dataset = train_dataset.map(operations=transforms_list, input_columns=["data"])
    train_dataset = train_dataset.map(operations=transforms.TypeCast(ms.int32), input_columns=["label"])
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

    # 加载验证集（不打乱）
    print("DataLoader: 加载玉米验证集MindRecord...")
    val_dataset = ds.MindDataset(VAL_MINDRECORD, columns_list=["data", "label"], shuffle=False)
    val_dataset = val_dataset.map(operations=transforms_list, input_columns=["data"])
    val_dataset = val_dataset.map(operations=transforms.TypeCast(ms.int32), input_columns=["label"])
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

    return train_dataset, val_dataset

# 加载测试集（用于最佳轮次模型的最终评估）
def create_test_dataset():
    if not os.path.exists(TEST_MINDRECORD):
        print(f"❌ Error: 测试集MindRecord文件不存在 → {TEST_MINDRECORD}")
        return None
    
    transforms_list = [
        vision.Decode(),
        vision.Resize(256),
        vision.CenterCrop(224),
        vision.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        vision.HWC2CHW()
    ]

    print("DataLoader: 加载玉米独立测试集MindRecord...")
    test_dataset = ds.MindDataset(TEST_MINDRECORD, columns_list=["data", "label"], shuffle=False)
    test_dataset = test_dataset.map(operations=transforms_list, input_columns=["data"])
    test_dataset = test_dataset.map(operations=transforms.TypeCast(ms.int32), input_columns=["label"])
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    return test_dataset

# 用最佳模型评估测试集
def evaluate_best_model(network, test_dataset, loss_fn):
    if test_dataset is None:
        return
    
    print("\n===== 用最佳轮次模型评估玉米独立测试集 =====")
    network.set_train(False)
    test_loss = 0.0
    correct = 0
    total = 0
    
    for data in test_dataset.create_dict_iterator():
        images = data["data"]
        labels = data["label"]
        total += labels.shape[0]
        
        logits = network(images)
        loss = loss_fn(logits, labels)
        test_loss += loss.asnumpy() * labels.shape[0]
        pred = np.argmax(logits.asnumpy(), axis=1)
        correct += (pred == labels.asnumpy()).sum()
    
    avg_test_loss = test_loss / total
    test_acc = correct / total
    print(f"✅ 测试集损失：{avg_test_loss:.4f} | 测试集准确率：{test_acc:.4f}")
    print("="*60 + "\n")
    return test_acc, avg_test_loss

def train():
    # 设置运行环境
    # 优先使用GPU，如果没有，则使用CPU
    target = "GPU" if ms.context.get_context("device_target") == "GPU" else "CPU"
    ms.set_device(target)
    context.set_context(mode=context.GRAPH_MODE)
    print(f"⚙️ 训练设备：{target}")
    print(f"输出目录：")
    print(f"   - 模型文件：{CKPT_DIR}")
    print(f"   - 训练曲线图/历史：{PLOT_DIR}")

    # 加载训练集+验证集
    train_dataset, val_dataset = create_train_val_dataset()
    step_size = train_dataset.get_dataset_size()
    
    # 构建MobileNetV3模型（迁移学习）
    print("Model: 构建MobileNetV3迁移学习模型...")
    network = create_model(model_name='mobilenet_v3_small_100', 
                           num_classes=NUM_CLASSES, 
                           pretrained=True)

    # 断点续训：加载最新CKPT
    latest_ckpt_path, latest_epoch = get_latest_ckpt(CKPT_DIR)
    if latest_ckpt_path:
        print(f"从断点恢复训练：{latest_ckpt_path}（中断于epoch {latest_epoch}）")
        param_dict = load_checkpoint(latest_ckpt_path)
        load_param_into_net(network, param_dict, strict_load=False)
    else:
        print("无断点，从头开始训练")
        latest_epoch = 0

    # 损失函数+优化器
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(network.trainable_params(), learning_rate=LEARNING_RATE)

    # CKPT保存配置
    config_ck = CheckpointConfig(save_checkpoint_steps=step_size * 5, keep_checkpoint_max=10)
    ckpoint = ModelCheckpoint(prefix="corn_disease", directory=CKPT_DIR, config=config_ck)

    # 初始化Model
    model = Model(network, loss_fn=loss_fn, optimizer=optimizer, metrics={'acc'})

    # 创建验证Callback（核心：跟踪最佳轮次）
    val_callback = ValCallback(val_dataset, network, loss_fn)

    # 构建Callback列表
    callbacks = [
        TimeMonitor(),
        LossMonitor(),
        ckpoint,
        val_callback
    ]
    
    # 开始训练
    print(f"Start: 开始玉米疾病分类模型训练（总轮数：{EPOCHS}，从epoch {latest_epoch} 开始）...")
    model.train(
        EPOCHS, 
        train_dataset, 
        callbacks=callbacks,
        initial_epoch=latest_epoch
    )
    
    # 训练结束：获取并打印最佳轮次信息
    best_info = val_callback.get_best_epoch_info()
    print("\n✅ 训练完成！最佳轮次总结：")
    print(f"⭐ 最佳训练轮次：Epoch {best_info['best_epoch']}")
    print(f"⭐ 最佳验证准确率：{best_info['best_val_acc']:.4f}")
    print(f"⭐ 最佳验证损失：{best_info['best_val_loss']:.4f}")
    
    # 绘制训练曲线
    print(f"\n正在生成训练曲线图...")
    plot_path = plot_training_curves(best_info, PLOT_DIR)
    
    # 保存训练历史
    history_file = save_training_history(best_info, PLOT_DIR)
    
    # 核心修改部分：保存最佳模型，并加载它用于最终评估
    if val_callback.best_model_params:
        best_ckpt_path = os.path.join(CKPT_DIR, f"corn_best_epoch_{best_info['best_epoch']}.ckpt")
        save_checkpoint(val_callback.best_model_params, best_ckpt_path)
        print(f"最佳轮次模型已保存至：{best_ckpt_path}")
        
        # **** 关键修正：加载最佳模型参数到网络中，用于评估测试集 ****
        param_dict = load_checkpoint(best_ckpt_path)
        load_param_into_net(network, param_dict, strict_load=False)
        print("✅ 已将网络参数切换为最佳轮次模型参数，准备评估测试集。")
    
    # 用最佳模型评估独立测试集
    test_dataset = create_test_dataset()
    if test_dataset:
        evaluate_best_model(network, test_dataset, loss_fn)
    
    # 显示最终统计信息
    print(f"\n训练统计摘要：")
    if best_info['all_epochs']:
        print(f"   - 总训练轮次：{len(best_info['all_epochs'])}")
        print(f"   - 训练损失变化：{best_info['all_train_loss'][0]:.4f} → {best_info['all_train_loss'][-1]:.4f}")
        print(f"   - 验证损失变化：{best_info['all_val_loss'][0]:.4f} → {best_info['all_val_loss'][-1]:.4f}")
        print(f"   - 验证准确率变化：{best_info['all_val_acc'][0]:.4f} → {best_info['all_val_acc'][-1]:.4f}")
    print(f"   - 训练曲线图：{plot_path}")
    print(f"   - 训练历史文件：{history_file}")

if __name__ == "__main__":
    train()