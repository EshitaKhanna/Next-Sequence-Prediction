import os
import numpy as np
import matplotlib.pyplot as plt
from utils import Preprocess4Normalization
from scipy.ndimage import uniform_filter1d
import torch
from torch.utils.data import Dataset, DataLoader
from models import LayerNorm, Transformer
import torch.nn as nn
import torch.nn.functional as F

# Load and normalize the data
data = np.load("dataset/hhar/data_20_120.npy")
label = np.load("dataset/hhar/label_20_120.npy")

print("Loading raw data - Data:", data.shape, "Labels:", label.shape)
print("Unique activity labels - ",np.unique(label[:,0,2]))
print("Before normalization - Data shape:", data.shape, data[0][0][:])

normalizer = Preprocess4Normalization(feature_len=6)
normalized_data = np.zeros_like(data)

for i in range(len(data)):
    normalized_data[i] = normalizer(data[i])

print("After normalization - Data:", normalized_data.shape, "Labels:", label.shape, normalized_data[0][0][:])

print("\nNormalized Samples")
for i in range(117, 120):
    print(normalized_data[0][i][:])

np.save('dataset/hhar/normalized_data_20_120.npy', normalized_data)

# Parameters for filtering and training
INITIAL_REMOVE = 20 
WINDOW_SIZE = 5
THRESHOLD = 0.0275
MIN_ACTIVE = 5 # Minimum active samples to keep a segment

PREDICT_LEN = 20     # predicted steps
INPUT_LEN = 70       # input length
TRAIN_RATE = 0.8     
BATCH_SIZE = 32      
LEARNING_RATE = 1e-4  

def log(msg):
    """Print message"""
    print(f"{msg}")

def log_hyperparameters():
    """Log training and preprocessing parameters"""

    log("Training Configurations:")
    log(f"INPUT_LEN: {INPUT_LEN}, PREDICT_LEN: {PREDICT_LEN}")
    log(f"TRAIN_SIZE: {TRAIN_RATE}, BATCH_SIZE: {BATCH_SIZE}, LEARNING_RATE: {LEARNING_RATE}")
    log(f"INITIAL_REMOVE: {INITIAL_REMOVE}, WINDOW_SIZE: {WINDOW_SIZE}, THRESHOLD: {THRESHOLD}, MIN_ACTIVE: {MIN_ACTIVE}")


def preprocess_sequence(seq, pad=False, global_max_len=None):
    """Remove few initial values, filter segments and apply padding
    Returns:
    - Filtered data
    - Kept indices from original sequence
    - Removed indices
    """
    # remove initial values
    original_indices = np.arange(len(seq))
    seq = seq[INITIAL_REMOVE:]
    original_indices = original_indices[INITIAL_REMOVE:]
    
    filtered_segments = []
    kept_indices = []
    
    # window
    for w_start in range(0, 100, 20):
        w_end = w_start + 20
        window = seq[w_start:w_end]
        window_indices = original_indices[w_start:w_end]
        
        # calculate magnitude of accelerometer
        magnitudes = np.linalg.norm(window[:, :3], axis=1)
        deviations = np.abs(magnitudes - 1.0)
        smoothed = uniform_filter1d(deviations, size=WINDOW_SIZE)
        active_mask = smoothed > THRESHOLD
        
        if np.sum(active_mask) >= MIN_ACTIVE:
            filtered_segments.append(window[active_mask])
            kept_indices.extend(window_indices[active_mask])
    
    # combine active segments if any
    if filtered_segments:
        filtered = np.concatenate(filtered_segments)
    else:
        # keep entire trimmed sequence if no active windows found
        filtered = seq
        kept_indices = list(original_indices)
    
    # removed indices
    original_after_removal = set(range(INITIAL_REMOVE, 120))
    removed = list(original_after_removal - set(kept_indices))
    removed.sort()
    
    # apply padding 
    if pad:
        assert global_max_len is not None, "Must provide global_max_len when padding"
        padded = np.full((global_max_len, 6), np.nan)
        padded[:len(filtered)] = filtered
        return padded, kept_indices, removed
    
    return filtered, kept_indices, removed


def process_dataset(data, pad=True):
    """
    Apply preprocessing to all sequences in dataset.
    If pad=True, output will have uniform lengths.
    """

    # get max length
    unpadded = [preprocess_sequence(seq, pad=False)[0] for seq in data]
    max_len = max(len(seq) for seq in unpadded) if pad else None

    results = []
    for seq in data:
        result = preprocess_sequence(seq, pad=pad, global_max_len=max_len)
        results.append(result)
    
    filtered_data = [r[0] for r in results]
    kept_indices = [r[1] for r in results]
    removed_indices = [r[2] for r in results]
    
    lengths = [len(k) for k in kept_indices]
    print(f"Number of sequences: {len(lengths)}")
    print(f"Min length: {np.min(lengths)}")
    print(f"Max length: {np.max(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths)}")

    return filtered_data, kept_indices, removed_indices

#Dataset for NSP
class Dataset4NSP(Dataset):
    """Dataset for Next Sequence Prediction"""
    def __init__(self, filtered_data, kept_indices, input_len=INPUT_LEN, pred_len=PREDICT_LEN):
        self.pairs = []
        self.input_len = INPUT_LEN
        self.pred_len = PREDICT_LEN
        
        # Generate valid input-target pairs
        for inst_data, kept in zip(filtered_data, kept_indices):
            valid_length = len(kept)
            for start in range(0, valid_length - input_len - pred_len + 1):
                input_seq = inst_data[start:start+input_len] 
                target_seq = inst_data[start+input_len:start+input_len+pred_len]
                
                input_seq = np.nan_to_num(input_seq, nan=0.0)
                target_seq = np.nan_to_num(target_seq, nan=0.0)
                
                # Get original timestamps for positional encoding
                input_pos = kept[start:start+input_len]
                target_pos = kept[start+input_len:start+input_len+pred_len]
                
                self.pairs.append((
                input_seq.astype(np.float32),
                np.array(input_pos, dtype=np.int64),  
                np.array(target_pos, dtype=np.int64),
                target_seq.astype(np.float32)
                 ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, input_pos, target_pos, target_seq = self.pairs[idx]
        return (
            torch.from_numpy(input_seq),
            torch.from_numpy(input_pos),
            torch.from_numpy(target_pos),
            torch.from_numpy(target_seq)
        )
 
# DataLoader preparation
def prepare_data(filtered_data, kept_indices):
    dataset = Dataset4NSP(filtered_data, kept_indices, input_len=INPUT_LEN, pred_len=PREDICT_LEN)
    print(f"Total NSP pairs: {len(dataset)}")
    for i in range(3):
        x, x_pos, y_pos, y = dataset[i]
        print(f"Input: {x.shape}, Target: {y.shape}, Input Pos: {x_pos.shape}, Target Pos: {y_pos.shape}")

    total_sequences = len(filtered_data)
    total_pairs = len(dataset)
    avg_pairs_per_seq = total_pairs / total_sequences
    
    print(f"Total sequences: {total_sequences}")
    print(f"Total NSP pairs: {total_pairs}")
    print(f"Average NSP pairs per sequence: {avg_pairs_per_seq:.2f}")
        
    #  Train/Val/Test split (80% train, 10% val, 10% test)
    train_size = int(TRAIN_RATE * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(f"Train NSP pairs: {len(train_dataset)}")  
    print(f"Val NSP pairs: {len(val_dataset)}")  
    print(f"Test NSP pairs: {len(test_dataset)}")  
    
    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_dataset, batch_size=BATCH_SIZE),
        DataLoader(test_dataset, batch_size=BATCH_SIZE)
    )

# model config
class Config:
    def __init__(self):
        self.feature_num = 6  
        self.hidden = 256     
        self.n_layers = 4     
        self.p_drop_hidden = 0.1  
        self.max_len = 120    # Max sequence length for positional encoding (safe margin)
        self.seq_len = 85 
        self.emb_norm = True
        self.hidden_ff = 512 
        self.n_heads = 8


# modified limu-bert
class LIMUBertModel4NSP(nn.Module):
    """LIMU-BERT transformer model for Next Sequence Prediction"""
    def __init__(self, cfg, max_position=120):
        super().__init__()
        self.transformer = Transformer(cfg)
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm = LayerNorm(cfg)
        self.abs_pos_embed = nn.Embedding(max_position, cfg.hidden)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.activ = F.gelu

    def forward(self, input_seqs, input_pos, target_pos):
        h = self.transformer(input_seqs) 
        h = h + self.abs_pos_embed(input_pos)  
        h = h[:, -PREDICT_LEN:]
        h = self.activ(self.linear(h))
        h = self.norm(h)
        
        return self.decoder(h) # Predict last PRED_LEN steps

# LSTM model
class LSTMNSPModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=6, num_layers=2):
        super(LSTMNSPModel, self).__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, future_steps=5):
        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder(x)
        decoder_input = torch.zeros((batch_size, 1, 6)).to(x.device) 

        outputs = []

        for _ in range(future_steps):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.output_layer(out)  # [batch, 1, 6]
            outputs.append(pred)
            decoder_input = pred  # Use predicted output as next input

        return torch.cat(outputs, dim=1)  # [batch_size, 5, 6], return full predicted sequence


def train_model_limu():

    log_hyperparameters()  

    os.makedirs("limu_nsp", exist_ok=True)
    model_name = os.path.join("limu_nsp", f"limu_nsp_{INPUT_LEN}_{PREDICT_LEN}")

    cfg = Config()
    model = LIMUBertModel4NSP(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction='mean')

    model.train_losses = []
    model.val_losses = []
    model.test_losses = []

    filtered_data, kept_indices, removed_indices = process_dataset(normalized_data, pad=True)
    print("Filtered data shapes:")
    for i, d in enumerate(filtered_data[:5]):  # first 5 only
        print(f"Sequence {i}: {d.shape}")

    train_loader, val_loader, test_loader = prepare_data(filtered_data, kept_indices)
    
    num_epochs = 500
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # training loop
        for batch in train_loader:
            input_seq, input_pos, target_pos, target_seq = batch
            pred = model(input_seq, input_pos, target_pos)
            loss = criterion(pred, target_seq).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # validation/test 
        model.eval()
        all_preds = []
        all_actuals = []
        val_loss = 0
        test_loss = 0
        with torch.no_grad():
            # validation
            for input_seq, input_pos, target_pos, target_seq in val_loader:
                preds = model(input_seq, input_pos, target_pos)
                val_loss += F.mse_loss(preds, target_seq, reduction='mean').item()
            
            # testing
            for batch in test_loader:
                input_seq, input_pos, target_pos, target_seq = batch
                pred = model(input_seq, input_pos, target_pos)
                test_loss += F.mse_loss(pred, target_seq, reduction='mean').item()
                all_preds.append(pred.numpy())
                all_actuals.append(target_seq.numpy())

        # save losses
        model.train_losses.append(train_loss/len(train_loader))
        model.val_losses.append(val_loss/len(val_loader))
        model.test_losses.append(test_loss/len(test_loader))

        print(f"Epoch {epoch+1}: "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}", 
            f"Test Loss: {test_loss/len(test_loader):.4f}")


    # Save final predictions and model
    preds_array = np.concatenate(all_preds)
    actuals_array = np.concatenate(all_actuals)

    np.save(f"{model_name}_predictions.npy", preds_array)
    np.save(f"{model_name}_actuals.npy", actuals_array)
    print(f"Saved LIMU-NSP predictions: {preds_array.shape}, actuals: {actuals_array.shape}")

    np.save(f"{model_name}_train_losses.npy", np.array(model.train_losses))
    np.save(f"{model_name}_val_losses.npy", np.array(model.val_losses))
    np.save(f"{model_name}_test_losses.npy", np.array(model.test_losses))
    np.save(f"{model_name}_kept_indices.npy", np.array(kept_indices, dtype=object))
    np.save(f"{model_name}_removed_indices.npy", np.array(removed_indices, dtype=object))

    torch.save(model, f"{model_name}.pt")
    print(f"Saved predictions and model in {model_name}")

    return model, preds_array, actuals_array, kept_indices


def train_lstm():
    log_hyperparameters()  

    os.makedirs("lstm_nsp", exist_ok=True)
    model_name = os.path.join("lstm_nsp", f"lstm_nsp_{INPUT_LEN}_{PREDICT_LEN}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNSPModel().to(device)
    criterion = nn.MSELoss()    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train_losses = []
    model.val_losses = []
    model.test_losses = []

    filtered_data, kept_indices, removed_indices = process_dataset(normalized_data, pad=True)
    print("Filtered data shapes:")
    for i, d in enumerate(filtered_data[:5]):  # first 5 only
        print(f"Sequence {i}: {d.shape}")
    train_loader, val_loader, test_loader = prepare_data(filtered_data, kept_indices)

    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # training
        for batch in train_loader:
            input_seq, input_pos, target_pos, target_seq = batch
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            optimizer.zero_grad()
            outputs = model(input_seq, future_steps=PREDICT_LEN)
            loss = criterion(outputs, target_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()

        all_preds = []
        all_actuals = []
        val_loss = 0
        test_loss = 0

        with torch.no_grad():
            # validation
            for batch in val_loader:
                input_seq, input_pos, target_pos, target_seq = batch
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)

                outputs = model(input_seq, future_steps=PREDICT_LEN)
                loss = criterion(outputs, target_seq)
                val_loss += loss.item()

            # testing
            for batch in test_loader:
                input_seq, input_pos, target_pos, target_seq = batch
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)

                outputs = model(input_seq, future_steps=PREDICT_LEN)
                loss = criterion(outputs, target_seq)
                test_loss += loss.item()

                all_preds.append(outputs.cpu().numpy())
                all_actuals.append(target_seq.cpu().numpy()) 
        
        avg_test_loss = test_loss / len(test_loader)
        avg_val_loss = val_loss / len(val_loader)

        model.train_losses.append(avg_loss)
        model.val_losses.append(avg_val_loss)
        model.test_losses.append(avg_test_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        
    # save final predictions and model
    preds_array = np.concatenate(all_preds)
    actuals_array = np.concatenate(all_actuals)

    np.save(f"{model_name}_predictions.npy", preds_array)
    np.save(f"{model_name}_actuals.npy", actuals_array)
    print(f"Saved LSTM-NSP predictions: {preds_array.shape}, actuals: {actuals_array.shape}")

    np.save(f"{model_name}_train_losses.npy", np.array(model.train_losses))
    np.save(f"{model_name}_val_losses.npy", np.array(model.val_losses))
    np.save(f"{model_name}_test_losses.npy", np.array(model.test_losses))
    np.save(f"{model_name}_kept_indices.npy", np.array(kept_indices, dtype=object))
    np.save(f"{model_name}_removed_indices.npy", np.array(removed_indices, dtype=object))

    torch.save(model, f"{model_name}.pt")
    print(f"Training completed and saved in {model_name}")

    return model, preds_array, actuals_array, kept_indices

if __name__ == "__main__":
    # train_model_limu()
    train_lstm()
