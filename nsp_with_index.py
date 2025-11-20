import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import Preprocess4Normalization
from scipy.ndimage import uniform_filter1d
import torch
from torch.utils.data import Dataset, DataLoader
from models import LayerNorm, Transformer
import torch.nn as nn
import torch.nn.functional as F
import itertools

# Load and normalize the data
data = np.load("dataset/hhar/data_20_120.npy")
label = np.load("dataset/hhar/label_20_120.npy")

print("Loading raw data - Data:", data.shape, "Labels:", label.shape)
print("Unique activity labels - ", np.unique(label[:,0,2]))
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

# Define multiple input-output size pairs
INPUT_OUTPUT_PAIRS = [
    (85, 5),   # Original
    (80, 10),  # New pair 1
    (70, 20)   # New pair 2
]

TRAIN_RATE = 0.8      
BATCH_SIZE = 32       
SEED = 42
LEARNING_RATE = 1e-3  

def log(msg):
    """Print message"""
    print(f"{msg}")

def log_hyperparameters(input_len, predict_len):
    """Log training and preprocessing parameters"""
    log("Training Configurations:")
    log(f"INPUT_LEN: {input_len}, PREDICT_LEN: {predict_len}")
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

# Base dataset class for compatibility across different input/output sizes
class BaseDataset4NSP(Dataset):
    """Base Dataset for Next Sequence Prediction"""
    def __init__(self, filtered_data, kept_indices):
        self.filtered_data = filtered_data
        self.kept_indices = kept_indices
        self.instance_lengths = [len(k) for k in kept_indices]
    
    def __len__(self):
        return len(self.filtered_data)
    
    def get_instance(self, idx):
        """Get a full instance's data and indices"""
        return self.filtered_data[idx], self.kept_indices[idx]

# Dataset for NSP with variable input/output lengths
class Dataset4NSP(Dataset):
    """Dataset for Next Sequence Prediction with configurable input/output lengths"""
    def __init__(self, base_dataset, input_len, pred_len):
        self.base_dataset = base_dataset
        self.input_len = input_len
        self.pred_len = pred_len
        self.pairs = []
        
        # Generate valid input-target pairs
        for inst_idx in range(len(base_dataset)):
            inst_data, kept = base_dataset.get_instance(inst_idx)
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
                    target_seq.astype(np.float32),
                    inst_idx  # Track which instance this came from
                ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, input_pos, target_pos, target_seq, _ = self.pairs[idx]
        return (
            torch.from_numpy(input_seq),
            torch.from_numpy(input_pos),
            torch.from_numpy(target_pos),
            torch.from_numpy(target_seq)
        )
    
    def get_pair_instance_idx(self, idx):
        """Get the instance index for a specific pair"""
        return self.pairs[idx][4]
 
# DataLoader preparation with consistent indices across configurations
def prepare_dataset_splits(filtered_data, kept_indices, seed=SEED):
    """Create consistent dataset splits independent of input/output lengths"""
    base_dataset = BaseDataset4NSP(filtered_data, kept_indices)
    
    # Split at the instance level rather than the pair level for consistency
    instance_indices = np.arange(len(base_dataset))
    
    # Split instances into train (80%) and temp (20%)
    train_instances, temp_instances = train_test_split(
        instance_indices, test_size=0.2, random_state=seed
    )
    
    # Split temp into val (10%) and test (10%)
    val_instances, test_instances = train_test_split(
        temp_instances, test_size=0.5, random_state=seed
    )
    
    split_info = {
        'train_instances': train_instances,
        'val_instances': val_instances,
        'test_instances': test_instances,
        'seed': seed
    }
    
    # Save the instance-level splits
    os.makedirs("splits", exist_ok=True)
    np.save(f"splits/instance_level_splits.npy", split_info)
    
    print(f"Instance splits - Train: {len(train_instances)}, Val: {len(val_instances)}, Test: {len(test_instances)}")
    
    return base_dataset, split_info

def prepare_data_loaders(base_dataset, split_info, input_len, pred_len):
    """Prepare dataloaders for a specific input/output configuration using consistent splits"""
    # Create the dataset for this specific input/output config
    dataset = Dataset4NSP(base_dataset, input_len, pred_len)
    
    # Use instance-level splits to create pair-level splits
    train_instances = split_info['train_instances']
    val_instances = split_info['val_instances']
    test_instances = split_info['test_instances']
    
    # Map pair indices based on which instance they belong to
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    for pair_idx in range(len(dataset)):
        inst_idx = dataset.get_pair_instance_idx(pair_idx)
        if inst_idx in train_instances:
            train_pairs.append(pair_idx)
        elif inst_idx in val_instances:
            val_pairs.append(pair_idx)
        elif inst_idx in test_instances:
            test_pairs.append(pair_idx)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_pairs)
    val_dataset = torch.utils.data.Subset(dataset, val_pairs)
    test_dataset = torch.utils.data.Subset(dataset, test_pairs)
    
    # Save the pair indices for this configuration
    pair_splits = {
        'train_pairs': train_pairs,
        'val_pairs': val_pairs, 
        'test_pairs': test_pairs,
        'input_len': input_len,
        'pred_len': pred_len
    }
    
    config_name = f"input_{input_len}_pred_{pred_len}"
    os.makedirs(f"splits/{config_name}", exist_ok=True)
    np.save(f"splits/{config_name}/pair_splits.npy", pair_splits)
    
    print(f"\nConfiguration: {config_name}")
    print(f"Pair splits - Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    
    # Create and return the DataLoaders
    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_dataset, batch_size=BATCH_SIZE),
        DataLoader(test_dataset, batch_size=BATCH_SIZE),
        pair_splits
    )

# model config
class Config:
    def __init__(self, input_len, pred_len):
        self.feature_num = 6  
        self.hidden = 256     
        self.n_layers = 4     
        self.p_drop_hidden = 0.1  
        self.max_len = 120    # Max sequence length for positional encoding (safe margin)
        self.seq_len = input_len  # Use the specified input length 
        self.pred_len = pred_len  # Store prediction length
        self.emb_norm = True
        self.hidden_ff = 512 
        self.n_heads = 8


# modified limu-bert for nsp with flexible prediction length
class LIMUBertModel4NSP(nn.Module):
    """LIMU-BERT transformer model for Next Sequence Prediction with configurable output length"""
    def __init__(self, cfg, max_position=120):
        super().__init__()
        self.transformer = Transformer(cfg)
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm = LayerNorm(cfg)
        self.abs_pos_embed = nn.Embedding(max_position, cfg.hidden)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.activ = F.gelu
        self.pred_len = cfg.pred_len  # Store prediction length

    def forward(self, input_seqs, input_pos, target_pos):
        h = self.transformer(input_seqs) 
        h = h + self.abs_pos_embed(input_pos)  
        h = h[:, -self.pred_len:]  # Take the last pred_len steps
        h = self.activ(self.linear(h))
        h = self.norm(h)
        
        return self.decoder(h)

# LSTM model with configurable prediction length
class LSTMNSPModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=6, num_layers=2, pred_len=5):
        super(LSTMNSPModel, self).__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.pred_len = pred_len

    def forward(self, x):
        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder(x)
        decoder_input = torch.zeros((batch_size, 1, 6)).to(x.device) 

        outputs = []

        for _ in range(self.pred_len):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.output_layer(out)  # [batch, 1, 6]
            outputs.append(pred)
            decoder_input = pred  # Use predicted output as next input

        return torch.cat(outputs, dim=1)  # [batch_size, pred_len, 6]


def train_model_limu(input_len, predict_len, base_dataset, split_info):
    log_hyperparameters(input_len, predict_len)  

    config_name = f"input_{input_len}_pred_{predict_len}"
    os.makedirs(f"limu_nsp/{config_name}", exist_ok=True)
    model_name = os.path.join(f"limu_nsp/{config_name}", f"limu_nsp_{input_len}_{predict_len}")

    cfg = Config(input_len, predict_len)
    model = LIMUBertModel4NSP(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction='mean')

    model.train_losses = []
    model.val_losses = []
    model.test_losses = []

    train_loader, val_loader, test_loader, pair_splits = prepare_data_loaders(
        base_dataset, split_info, input_len, predict_len
    )
    
    # Save test indices
    test_pairs = pair_splits['test_pairs']
    np.save(f"{model_name}_test_pairs.npy", test_pairs)
    print(f"Saved test pairs: {len(test_pairs)}")
    
    num_epochs = 200  # Reduced for demonstration
    
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
                all_preds.append(pred.detach().cpu().numpy())
                all_actuals.append(target_seq.detach().cpu().numpy())

        # save losses
        model.train_losses.append(train_loss/len(train_loader))
        model.val_losses.append(val_loss/len(val_loader))
        model.test_losses.append(test_loss/len(test_loader))

        print(f"Epoch {epoch+1}: "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, " 
            f"Test Loss: {test_loss/len(test_loader):.4f}")

    # Save final predictions and model
    if all_preds:  # Check if we have predictions to save
        preds_array = np.concatenate(all_preds)
        actuals_array = np.concatenate(all_actuals)

        np.save(f"{model_name}_predictions.npy", preds_array)
        np.save(f"{model_name}_actuals.npy", actuals_array)
        print(f"Saved LIMU-NSP predictions: {preds_array.shape}, actuals: {actuals_array.shape}")

    np.save(f"{model_name}_train_losses.npy", np.array(model.train_losses))
    np.save(f"{model_name}_val_losses.npy", np.array(model.val_losses))
    np.save(f"{model_name}_test_losses.npy", np.array(model.test_losses))
    
    torch.save(model, f"{model_name}.pt")
    print(f"Saved model in {model_name}")

    return model

def train_lstm(input_len, predict_len, base_dataset, split_info):
    log_hyperparameters(input_len, predict_len)  

    config_name = f"input_{input_len}_pred_{predict_len}"
    base_dir = f"lstm_nsp/{config_name}"
    os.makedirs(base_dir, exist_ok=True)
    model_name = os.path.join(base_dir, f"lstm_nsp_{input_len}_{predict_len}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNSPModel(pred_len=predict_len).to(device)
    criterion = nn.MSELoss()    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train_losses = []
    model.val_losses = []
    model.test_losses = []

    train_loader, val_loader, test_loader, pair_splits = prepare_data_loaders(
        base_dataset, split_info, input_len, predict_len
    )

    # save test indices 
    test_pairs = pair_splits['test_pairs']
    np.save(f"{model_name}_test_pairs.npy", test_pairs)
    print(f"Saved test pairs: {len(test_pairs)}")

    num_epochs = 200  # Reduced for demonstration
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # training
        for batch in train_loader:
            input_seq, input_pos, target_pos, target_seq = batch
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            optimizer.zero_grad()
            outputs = model(input_seq)
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

                outputs = model(input_seq)
                loss = criterion(outputs, target_seq)
                val_loss += loss.item()

            # testing
            for batch in test_loader:
                input_seq, input_pos, target_pos, target_seq = batch
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)

                outputs = model(input_seq)
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
    if all_preds:  # Check if we have predictions to save
        preds_array = np.concatenate(all_preds)
        actuals_array = np.concatenate(all_actuals)

        np.save(f"{model_name}_predictions.npy", preds_array)
        np.save(f"{model_name}_actuals.npy", actuals_array)
        print(f"Saved LSTM-NSP predictions: {preds_array.shape}, actuals: {actuals_array.shape}")

    np.save(f"{model_name}_train_losses.npy", np.array(model.train_losses))
    np.save(f"{model_name}_val_losses.npy", np.array(model.val_losses))
    np.save(f"{model_name}_test_losses.npy", np.array(model.test_losses))

    torch.save(model, f"{model_name}.pt")
    print(f"Training completed and saved in {model_name}")

    return model

def compare_models(pairs):
    """Compare performance across different input/output configurations"""
    plt.figure(figsize=(15, 10))
    
    # Plot for LSTM models
    plt.subplot(1, 2, 1)
    for input_len, pred_len in pairs:
        config_name = f"input_{input_len}_pred_{pred_len}"
        test_losses = np.load(f"lstm_nsp/{config_name}/lstm_nsp_{input_len}_{pred_len}_test_losses.npy")
        plt.plot(test_losses, label=f"LSTM {input_len}->{pred_len}")
    
    plt.title('LSTM Models Test Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot for LIMU models
    plt.subplot(1, 2, 2)
    for input_len, pred_len in pairs:
        config_name = f"input_{input_len}_pred_{pred_len}"
        test_losses = np.load(f"limu_nsp/{config_name}/limu_nsp_{input_len}_{pred_len}_test_losses.npy")
        plt.plot(test_losses, label=f"LIMU {input_len}->{pred_len}")
    
    plt.title('LIMU Models Test Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    print("Model comparison saved to model_comparison.png")

if __name__ == "__main__":
    # Process dataset once
    filtered_data, kept_indices, removed_indices = process_dataset(normalized_data, pad=True)
    
    # Save the processed data
    os.makedirs("processed_data", exist_ok=True)
    np.save("processed_data/filtered_data.npy", filtered_data)
    
    # Save indices as pickle files to handle varying-length lists
    import pickle
    with open("processed_data/kept_indices.pkl", 'wb') as f:
        pickle.dump(kept_indices, f)
    with open("processed_data/removed_indices.pkl", 'wb') as f:
        pickle.dump(removed_indices, f)
    
    # Create consistent dataset splits
    base_dataset, split_info = prepare_dataset_splits(filtered_data, kept_indices)
    
    # Train models with different input/output configurations
    print("\n===== Training Models with Different Configurations =====")
    models = {}
    
    # Train LSTM models for all configurations
    for input_len, pred_len in INPUT_OUTPUT_PAIRS:
        print(f"\n===== Training LSTM Model with Input Length {input_len}, Prediction Length {pred_len} =====")
        model = train_lstm(input_len, pred_len, base_dataset, split_info)
        models[f"lstm_{input_len}_{pred_len}"] = model
    
    # Train LIMU models for all configurations
    for input_len, pred_len in INPUT_OUTPUT_PAIRS:
        print(f"\n===== Training LIMU Model with Input Length {input_len}, Prediction Length {pred_len} =====")
        model = train_model_limu(input_len, pred_len, base_dataset, split_info)
        models[f"limu_{input_len}_{pred_len}"] = model
    
    # Compare model performance
    compare_models(INPUT_OUTPUT_PAIRS)