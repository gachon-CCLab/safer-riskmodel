import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Loading data
dong = pd.read_csv('merged_data_m1_dong.csv')
yongin = pd.read_csv('merged_data_m1_yong.csv')
seoul = pd.read_csv('merged_data_m1_seoul.csv')

# Combining dataframes
data = pd.concat([yongin, seoul, dong], ignore_index=True)

# Scaling numeric features
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
scaler = MinMaxScaler(feature_range=(-1, 1))
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Filling NaN values
data = data.fillna(0)

# Convert target value
data['suicide'] = (data['suicide'] > 0).astype(float)

# Define static and sequence variables
static_variables = [
    'MED_SH','BIS_sum','BPRS_AFF','PH_tx_status','convicted','CTQ_PN','CTQ_EN','BPRS_POS','CS_status_1_score_cal','DIG_cat'
]

sequence_variables = [
    'Daily_Entropy', 'Normalized_Daily_Entropy', 'Eight_Hour_Entropy',
    'Normalized_Eight_Hour_Entropy', 'Location_Variability', 'place',
    'first_TOTAL_ACCELERATION', 'last_TOTAL_ACCELERATION', 'mean_TOTAL_ACCELERATION',
    'median_TOTAL_ACCELERATION', 'max_TOTAL_ACCELERATION', 'min_TOTAL_ACCELERATION',
    'std_TOTAL_ACCELERATION', 'nunique_TOTAL_ACCELERATION', 'first_HEARTBEAT',
    'last_HEARTBEAT', 'mean_HEARTBEAT', 'median_HEARTBEAT', 'max_HEARTBEAT',
    'min_HEARTBEAT', 'std_HEARTBEAT', 'nunique_HEARTBEAT', 'delta_DISTANCE',
    'delta_SLEEP', 'delta_STEP', 'delta_CALORIES'
]

# Ensure all static variables are numeric
for col in static_variables:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

class SuicideDataset(Dataset):
    def __init__(self, data, target_col, static_variables, sequence_variables):
        self.df = data
        self.target_col = target_col
        self.static_variables = static_variables
        self.sequence_variables = sequence_variables

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = row[self.target_col]
        static_data = row[self.static_variables].astype(np.float32)
        sequence_data = row[self.sequence_variables].astype(np.float32)
        return torch.tensor(static_data.values, dtype=torch.float32), torch.tensor(sequence_data.values, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# Create dataset instance
dataset = SuicideDataset(data, 'suicide', static_variables, sequence_variables)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VariableSelectionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.fc3 = nn.Linear(input_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, context=None):
        if context is not None:
            x = torch.cat((x, context), dim=1)
        x1 = torch.relu(self.fc1(x))
        x1 = self.fc2(x1)
        x2 = torch.relu(self.fc3(x))
        x2 = self.fc4(x2)
        x2 = self.sigmoid(x2)
        return x1 * x2

class TemporalFusionTransformer(nn.Module):
    def __init__(self, hidden_size, lstm_layers, dropout, output_size, attention_head_size, static_input_size, sequence_input_size):
        super(TemporalFusionTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.output_size = output_size
        self.attention_head_size = attention_head_size
        
        # Variable Selection Networks
        self.static_vsn = VariableSelectionNetwork(static_input_size, hidden_size)
        self.sequence_vsn = VariableSelectionNetwork(sequence_input_size, hidden_size)
        
        # LSTM layer for sequence data
        self.lstm = nn.LSTM(input_size=sequence_input_size, hidden_size=hidden_size, num_layers=lstm_layers, dropout=dropout, batch_first=True)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attention_head_size, batch_first=True)
        
        # Fully connected layer for static data
        self.static_fc = nn.Linear(static_input_size, hidden_size)
        
        # Combined fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_static, x_sequence):
        # Variable selection for static data
        static_weights = self.static_vsn(x_static)
        static_out = torch.mul(x_static, static_weights)
        static_out = self.static_fc(static_out)
        
        # Variable selection for sequence data
        sequence_weights = self.sequence_vsn(x_sequence)
        x_sequence = torch.mul(x_sequence, sequence_weights)
        
        # Processing sequence data with LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_sequence.unsqueeze(1))  # Ensure sequence input is 3D
        
        # Attention layer
        attn_output, attn_output_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine static and sequence features
        combined_out = torch.cat((static_out, attn_output[:, -1, :]), dim=1)  # (batch_size, hidden_size * 2)
        
        # Fully connected layer
        output = self.fc(combined_out)
        
        # Sigmoid activation
        output = self.sigmoid(output)
        
        return output, static_weights, sequence_weights


# 모델 클래스 및 기타 필요한 클래스/함수를 정의합니다
# 예시: class TemporalFusionTransformer(nn.Module): ...

# 모델 하이퍼파라미터 설정
hidden_size = 16
lstm_layers = 2
dropout = 0.1
output_size = 1
attention_head_size = 4
static_input_size = len(static_variables)
sequence_input_size = len(sequence_variables)

# 모델 초기화
model = TemporalFusionTransformer(hidden_size, lstm_layers, dropout, output_size, attention_head_size, static_input_size, sequence_input_size)
model = model.to(device)

# 옵티마이저 초기화
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 체크포인트 로드
checkpoint = torch.load('tft_model.pth')

# 모델과 옵티마이저의 상태 로드
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print("모델과 옵티마이저 상태가 성공적으로 로드되었습니다.")


def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            static_data, sequence_data, target = batch
            static_data, sequence_data, target = static_data.to(device), sequence_data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs, _, _ = model(static_data, sequence_data)
            loss = criterion(outputs.squeeze(), target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}')

def evaluate_model(model, dataloader):
    model.eval()
    all_targets = []
    all_predictions = []
    static_weights_all = []
    sequence_weights_all = []
    with torch.no_grad():
        for batch in dataloader:
            static_data, sequence_data, target = batch
            static_data, sequence_data, target = static_data.to(device), sequence_data.to(device), target.to(device)
            
            outputs, static_weights, sequence_weights = model(static_data, sequence_data)
            predictions = (outputs >= 0.5).float()
            
            all_targets.extend(target.tolist())
            all_predictions.extend(predictions.tolist())
            static_weights_all.extend(static_weights.cpu().numpy())
            sequence_weights_all.extend(sequence_weights.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_predictions)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Sensitivity
    sensitivity = tp / (tp + fn)
    
    # Specificity
    specificity = tn / (tn + fp)
    
    return accuracy, precision, recall, f1, auc, sensitivity, specificity, static_weights_all, sequence_weights_all

# Train model
train_model(model, train_dataloader, criterion, optimizer, num_epochs=100)

# Evaluate model
accuracy, precision, recall, f1, auc, sensitivity, specificity, static_weights_all, sequence_weights_all = evaluate_model(model, val_dataloader)
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}')

