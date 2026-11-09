import torch
import torch.nn as nn

class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=126, hidden_size=64, num_layers=2, num_classes=5, dropout=0.2):
        super(SignLanguageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Lapisan LSTM untuk menangkap pergerakan waktu (temporal dynamics)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Lapisan Fully Connected untuk klasifikasi kata
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length=30, input_size=126)
        out, (hn, cn) = self.lstm(x)
        # Ambil representasi pada frame terakhir
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
