import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, auc, precision_recall_curve, roc_curve
import random
import os


# 设置随机种子和设备
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

patch_size = 2
len_model = int(160 // patch_size)

# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, normal_features, anomaly_features):
        # 计算余弦相似度
        similarity = self.cos_sim(normal_features, anomaly_features) / self.temperature
        # 对比损失: 最大化异常和正常样本之间的距离
        loss = torch.mean(-torch.log(1 - torch.sigmoid(similarity) + 1e-8))
        return loss

# 位置编码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Transformer模型定义 - 带有对比学习功能
class ContrastiveTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.1):
        super(ContrastiveTimeSeriesTransformer, self).__init__()
        # 特征嵌入层
        self.embedding = nn.Linear(input_dim * patch_size, d_model)  # patching
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # 输出层
        self.output_layer = nn.Linear(d_model, 1 * patch_size)  # patching
        self.sigmoid = nn.Sigmoid()
        
        # 子片段扩展层
        self.fragment_expansion = nn.Linear(13, 160)
        
        # 特征提取器 - 用于对比学习
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
    def forward(self, x, get_features=False):
        # x形状: [batch_size, seq_len, input_dim]
        
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # patching the data
        if seq_len == 160:  # 完整序列
            x_patched = x.view(batch_size, len_model, patch_size, -1).view(batch_size, len_model, -1)
        elif seq_len == 13:  # 子片段
            # 先扩展子片段至完整序列长度
            x = self.fragment_expansion(x.transpose(1, 2)).transpose(1, 2)
            # 然后再进行patching
            x_patched = x.view(batch_size, len_model, patch_size, -1).view(batch_size, len_model, -1)
        else:
            raise ValueError(f"Unexpected sequence length: {seq_len}")
            
        # 嵌入和特征提取
        x = self.embedding(x_patched)  # [batch_size, len_model, d_model]
        x = self.pos_encoder(x)
        features = self.transformer_encoder(x)  # [batch_size, len_model, d_model]
        
        # 提取用于对比学习的特征
        if get_features:
            # 使用序列的平均特征表示
            global_features = torch.mean(features, dim=1)  # [batch_size, d_model]
            global_features = self.feature_extractor(global_features)  # [batch_size, d_model//4]
            return global_features
        
        # 输出预测
        x = self.output_layer(features)  # [batch_size, len_model, patch_size]
        
        # dispatching the data
        x = x.view(batch_size, len_model, patch_size, -1).view(batch_size, 160, -1)
        
        x = self.sigmoid(x)
        return x.squeeze(-1)  # [batch_size, seq_len]

# 提取子片段
def extract_fragments(data, labels, fragment_len=13):
    """
    从每个样本中提取正常和异常子片段
    
    参数:
    data - 形状为 (n_samples, 160, 1) 的数据
    labels - 形状为 (n_samples, 160) 的标签
    fragment_len - 子片段长度，默认为13
    
    返回:
    normal_fragments - 形状为 (n_samples, fragment_len, 1) 的正常子片段
    anomaly_fragments - 形状为 (n_samples, fragment_len, 1) 的异常子片段
    """
    n_samples = data.shape[0]
    seq_len = data.shape[1]
    normal_fragments = np.zeros((n_samples, fragment_len, data.shape[2]))
    anomaly_fragments = np.zeros((n_samples, fragment_len, data.shape[2]))
    
    for i in range(n_samples):
        # 找到异常区域
        anomaly_indices = np.where(labels[i] == 1)[0]
        
        if len(anomaly_indices) == 0:
            # 如果没有异常，随机选择一段作为"异常"片段
            anomaly_start = np.random.randint(0, seq_len - fragment_len)
            anomaly_end = anomaly_start + fragment_len
            
            # 随机选择另一段作为正常片段，确保不重叠
            normal_start = (anomaly_end + fragment_len) % seq_len
            normal_end = normal_start + fragment_len
            if normal_end > seq_len:
                normal_start = max(0, seq_len - fragment_len)
                normal_end = seq_len
        else:
            # 获取异常起始和结束位置
            anomaly_start = anomaly_indices[0]
            anomaly_end = anomaly_indices[-1] + 1
            
            # 计算异常中心点
            anomaly_center = (anomaly_start + anomaly_end) // 2
            
            # 以异常中心为中心，提取异常片段
            half_len = fragment_len // 2
            anomaly_start = max(0, anomaly_center - half_len)
            anomaly_end = min(seq_len, anomaly_start + fragment_len)
            
            # 调整以确保片段长度正确
            if anomaly_end - anomaly_start < fragment_len:
                anomaly_start = max(0, anomaly_end - fragment_len)
            
            # 选择正常片段
            if anomaly_start >= fragment_len:
                # 前面有足够空间，取前面的片段
                normal_start = anomaly_start - fragment_len
                normal_end = anomaly_start
            else:
                # 前面空间不够，取后面的片段
                normal_start = min(anomaly_end, seq_len - fragment_len)
                normal_end = normal_start + fragment_len
        
        # 确保索引在范围内
        normal_end = min(normal_end, seq_len)
        anomaly_end = min(anomaly_end, seq_len)
        
        # 截取片段
        if normal_end - normal_start == fragment_len:
            normal_fragments[i] = data[i, normal_start:normal_end]
        else:
            # 如果长度不够，填充
            actual_len = normal_end - normal_start
            normal_fragments[i, :actual_len] = data[i, normal_start:normal_end]
            
        if anomaly_end - anomaly_start == fragment_len:
            anomaly_fragments[i] = data[i, anomaly_start:anomaly_end]
        else:
            # 如果长度不够，填充
            actual_len = anomaly_end - anomaly_start
            anomaly_fragments[i, :actual_len] = data[i, anomaly_start:anomaly_end]
    
    return normal_fragments, anomaly_fragments

# 评估函数
def evaluate(model, dataloader, threshold=0.1):
    model.eval()
    all_preds = []
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = (outputs >= threshold).float()
            scores = outputs  
            all_preds.append(predictions.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    
    # 将预测结果和标签展平用于计算指标
    flat_preds = all_preds.reshape(-1)
    flat_labels = all_labels.reshape(-1)
    flat_scores = all_scores.reshape(-1)
    
    # 原有指标
    precision = precision_score(flat_labels, flat_preds, zero_division=0)
    recall = recall_score(flat_labels, flat_preds, zero_division=0)
    f1 = f1_score(flat_labels, flat_preds, zero_division=0)
    
    # 计算false alarm rate (FAR) = FP / (FP + TN)
    false_positives = np.sum((flat_preds == 1) & (flat_labels == 0))
    true_negatives = np.sum((flat_preds == 0) & (flat_labels == 0))
    false_alarm = false_positives / (false_positives + true_negatives + 1e-10)
    
    # 添加accuracy
    accuracy = np.mean(flat_preds == flat_labels)
    
    # 计算Matthews相关系数(MCC)
    mcc = matthews_corrcoef(flat_labels, flat_preds)
    
    # 计算ROC曲线下面积(AUC)
    fpr, tpr, _ = roc_curve(flat_labels, flat_scores)
    roc_auc = auc(fpr, tpr)
    
    # 计算PR曲线下面积(PR-AUC)
    precision_curve, recall_curve, _ = precision_recall_curve(flat_labels, flat_scores)
    pr_auc = auc(recall_curve, precision_curve)
    
    # 计算IoU (Intersection over Union)
    batch_size = all_preds.shape[0]
    sample_ious = []
    
    # 计算平均检测延迟和Early Detection Score
    detection_delays = []
    tp_scores = []
    
    for i in range(batch_size):
        # 对每个样本，找出异常区间
        pred_anomaly = np.where(all_preds[i] == 1)[0]
        true_anomaly = np.where(all_labels[i] == 1)[0]
        
        # 计算IoU
        if len(pred_anomaly) > 0 and len(true_anomaly) > 0:
            # 转换为区间
            pred_ranges = get_ranges(pred_anomaly)
            true_ranges = get_ranges(true_anomaly)
            
            # 计算所有区间组合的IoU，取最大值
            max_iou = 0
            for pr in pred_ranges:
                for tr in true_ranges:
                    # 计算交集
                    intersection = max(0, min(pr[1], tr[1]) - max(pr[0], tr[0]))
                    if intersection > 0:
                        # 计算并集
                        union = (pr[1] - pr[0]) + (tr[1] - tr[0]) - intersection
                        iou = intersection / union
                        max_iou = max(max_iou, iou)
            
            sample_ious.append(max_iou)
            
            # 计算检测延迟
            earliest_true = true_anomaly[0]
            earliest_pred = min(pred_anomaly) if len(pred_anomaly) > 0 else float('inf')
            
            if earliest_pred < float('inf'):
                delay = max(0, earliest_pred - earliest_true)
                detection_delays.append(delay)
                
                # 延迟越小得分越高，使用指数衰减
                alpha = 0.5  # 衰减系数
                score = np.exp(-alpha * delay / len(all_labels[i]))
                tp_scores.append(score)
        elif len(pred_anomaly) == 0 and len(true_anomaly) == 0:
            # 如果都没有异常，IoU为1
            sample_ious.append(1.0)
        else:
            # 如果只有一方有异常，IoU为0
            sample_ious.append(0.0)
            
            # 如果有真实异常但没检测到，最大延迟
            if len(true_anomaly) > 0 and len(pred_anomaly) == 0:
                detection_delays.append(len(all_labels[i]))
                tp_scores.append(0.0)  # 未检测，得分为0
    
    mean_iou = np.mean(sample_ious) if sample_ious else 0.0
    mean_detection_delay = np.mean(detection_delays) if detection_delays else float('inf')
    early_detection_score = np.mean(tp_scores) if tp_scores else 0.0
    
    # 计算覆盖率和漏报率
    total_anomalies = np.sum(flat_labels)
    correct_anomalies = np.sum((flat_preds == 1) & (flat_labels == 1))
    coverage = correct_anomalies / (total_anomalies + 1e-10)
    miss_rate = 1.0 - coverage
    
    return {
        'precision': precision,             # 范围:[0,1], 越大越好
        'recall': recall,                   # 范围:[0,1], 越大越好
        'f1_score': f1,                     # 范围:[0,1], 越大越好
        'false_alarm': false_alarm,         # 范围:[0,1], 越小越好
        'accuracy': accuracy,               # 范围:[0,1], 越大越好
        'iou': mean_iou,                    # 范围:[0,1], 越大越好
        'mcc': mcc,                         # 范围:[-1,1], 越接近1越好
        'roc_auc': roc_auc,                 # 范围:[0,1], 越大越好
        'pr_auc': pr_auc,                   # 范围:[0,1], 越大越好
        'mean_detection_delay': mean_detection_delay,  # 范围:[0,∞), 越小越好
        'early_detection_score': early_detection_score,  # 范围:[0,1], 越大越好
        'coverage': coverage,               # 范围:[0,1], 越大越好
        'miss_rate': miss_rate              # 范围:[0,1], 越小越好
    }, all_preds, all_scores

# 辅助函数：将连续的点转换为区间
def get_ranges(indices):
    if len(indices) == 0:
        return []
    
    ranges = []
    start = indices[0]
    end = indices[0]
    
    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            ranges.append((start, end + 1))  # 区间是左闭右开
            start = indices[i]
            end = indices[i]
    
    ranges.append((start, end + 1))
    return ranges
 
# 训练函数 - 对比学习
def train_model_with_contrastive(model, train_loader, valid_loader, optimizer, criterion, contrastive_criterion, 
                                normal_fragments, anomaly_fragments, num_epochs=30, contrastive_weight=0.5):
    best_f1 = 0
    best_model_state = None
    
    # 将片段数据转换为张量并创建数据加载器
    normal_tensor = torch.FloatTensor(normal_fragments)
    anomaly_tensor = torch.FloatTensor(anomaly_fragments)
    fragment_dataset = TensorDataset(normal_tensor, anomaly_tensor)
    fragment_loader = DataLoader(fragment_dataset, batch_size=16, shuffle=True)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        contrastive_loss_total = 0
        
        # 创建片段数据的迭代器
        fragment_iter = iter(fragment_loader)
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 主要预测损失
            outputs = model(inputs)
            main_loss = criterion(outputs, labels)
            
            # 对比学习损失
            try:
                normal_batch, anomaly_batch = next(fragment_iter)
            except StopIteration:
                fragment_iter = iter(fragment_loader)
                normal_batch, anomaly_batch = next(fragment_iter)
                
            normal_batch, anomaly_batch = normal_batch.to(device), anomaly_batch.to(device)
            
            # 获取特征
            normal_features = model(normal_batch, get_features=True)
            anomaly_features = model(anomaly_batch, get_features=True)
            
            # 计算对比损失
            contrast_loss = contrastive_criterion(normal_features, anomaly_features)
            
            # 总损失
            loss = main_loss + contrastive_weight * contrast_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += main_loss.item()
            contrastive_loss_total += contrast_loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_contrastive_loss = contrastive_loss_total / len(train_loader)
        
        # 验证阶段
        metrics, _, _ = evaluate(model, valid_loader)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
                  f"Valid Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                  f"F1: {metrics['f1_score']:.4f}, False Alarm: {metrics['false_alarm']:.4f}")
            
        # 保存最佳模型
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model_state = model.state_dict().copy()
            
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model

# # 主函数
# def main_contrastive(number_epoch=100, contrastive_weight=0.5):
    import math
    # 1. 加载数据
    X_train = all_data_np[:800, :, :]  # 替换为实际训练数据
    y_train = all_labels_np[:800, :]  # 替换为实际训练标签
    X_valid = all_data_np[800:1000, :, :]  # 替换为实际验证数据
    y_valid = all_labels_np[800:1000, :]  # 替换为实际验证标签
    X_test = own_test_data  # 替换为实际测试数据
    y_test = own_test_label  # 替换为实际测试标签
    
    # 提取子片段用于对比学习
    print("提取子片段用于对比学习...")
    normal_fragments, anomaly_fragments = extract_fragments(X_train, y_train, fragment_len=13)
    print(f"提取的子片段形状: 正常片段 {normal_fragments.shape}, 异常片段 {anomaly_fragments.shape}")
    
    # 将数据转换为PyTorch张量
    train_inputs = torch.FloatTensor(X_train)
    train_labels = torch.FloatTensor(y_train)
    valid_inputs = torch.FloatTensor(X_valid)
    valid_labels = torch.FloatTensor(y_valid)
    test_inputs = torch.FloatTensor(X_test)
    test_labels = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    batch_size = 16
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_inputs, valid_labels)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 2. 初始化模型
    model = ContrastiveTimeSeriesTransformer(
        input_dim=1,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.2
    ).to(device)
    
    # 3. 定义损失函数和优化器
    criterion = nn.BCELoss()
    contrastive_criterion = ContrastiveLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 训练模型
    print(f"开始训练，对比学习权重: {contrastive_weight}")
    model = train_model_with_contrastive(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        contrastive_criterion=contrastive_criterion,
        normal_fragments=normal_fragments,
        anomaly_fragments=anomaly_fragments,
        num_epochs=number_epoch,
        contrastive_weight=contrastive_weight
    )
    
    # 5. 在测试集上评估
    test_metrics, test_predictions, test_scores = evaluate(model, test_loader)
    print("\nTest Results:")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"False Alarm Rate: {test_metrics['false_alarm']:.4f}")
    
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"IoU: {test_metrics['iou']:.4f}")
    print(f"Matthews(MCC): {test_metrics['mcc']:.4f}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Mean Detection Delay: {test_metrics['mean_detection_delay']:.4f}")

    print(f"Early Detection Score: {test_metrics['early_detection_score']:.4f}")
    print(f"Coverage: {test_metrics['coverage']:.4f}")
    print(f"Miss Rate: {test_metrics['miss_rate']:.4f}")
    
    # 6. 导出预测结果
    import os
    save_path = 'own_results\contrastive_transformer/rushorder_BS_27'
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path +'/own_predictions_27.npy', test_predictions)
    np.save(save_path +'/own_scores_27.npy', test_scores)
    print(f"Predictions saved to {save_path}")
    
    return model, test_metrics
