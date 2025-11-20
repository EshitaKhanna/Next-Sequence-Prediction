import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from itertools import product

# Constants - keep consistent with nsp_with_index.py
INPUT_OUTPUT_PAIRS = [
    (85, 5),   # Original
    (80, 10),  # New pair 1
    (70, 20)   # New pair 2
]
BATCH_SIZE = 32

def load_data():
    """Load all necessary data"""
    print("Loading data...")
    
    # Load raw data and labels as per nsp_with_index.py
    data = np.load("dataset/hhar/data_20_120.npy")
    labels = np.load("dataset/hhar/label_20_120.npy")
    
    # Load normalized data saved by nsp_with_index.py
    normalized_data = np.load("dataset/hhar/normalized_data_20_120.npy")
    
    # Load processed data created during nsp_with_index.py execution
    filtered_data = np.load("processed_data/filtered_data.npy")
    
    # Load indices saved as pickle files
    with open("processed_data/kept_indices.pkl", 'rb') as f:
        kept_indices = pickle.load(f)
    
    # Load splits info saved during training
    split_info = np.load("splits/instance_level_splits.npy", allow_pickle=True).item()
    
    print("Data loaded successfully!")
    print(f"Data shapes - Normalized: {normalized_data.shape}, Labels: {labels.shape}, Filtered: {filtered_data.shape}")
    return normalized_data, labels, filtered_data, kept_indices, split_info

def get_majority_activity_labels(labels):
    """Get majority activity label for each sample"""
    num_samples = labels.shape[0]
    majority_labels = np.zeros(num_samples)
    
    # Activity labels are at index 2 of the labels array
    for i in range(num_samples):
        sample_labels = labels[i, :, 2]  # All labels for this sample
        unique_labels, counts = np.unique(sample_labels, return_counts=True)
        majority_labels[i] = unique_labels[np.argmax(counts)]
    
    # Check how many samples have multiple activity labels
    unique_counts = [len(np.unique(labels[i, :, 2])) for i in range(num_samples)]
    multi_label_count = sum(1 for count in unique_counts if count > 1)
    
    print(f"Samples with multiple activity labels: {multi_label_count} out of {num_samples}")
    print(f"Unique activity labels: {np.unique(majority_labels)}")
    
    return majority_labels

# Baseline Models
def linear_regression_predict(input_seq, pred_len):
    """Linear regression prediction for all 6 features"""
    predictions = np.zeros((pred_len, 6))
    for feature in range(6):
        x = np.arange(len(input_seq))
        y = input_seq[:, feature]
        
        # Handle NaN values that might be present in the data
        valid_indices = ~np.isnan(y)
        if np.sum(valid_indices) < 2:  # Need at least 2 points for regression
            predictions[:, feature] = np.nanmean(y) if not np.all(np.isnan(y)) else 0
            continue
            
        x_valid = x[valid_indices]
        y_valid = y[valid_indices]
        
        A = np.vstack([x_valid, np.ones(len(x_valid))]).T
        m, c = np.linalg.lstsq(A, y_valid, rcond=None)[0]
        
        for i in range(pred_len):
            predictions[i, feature] = m * (len(input_seq) + i) + c
            
    return predictions

def polynomial_regression_predict(input_seq, pred_len, degree=2):
    """Polynomial regression prediction"""
    predictions = np.zeros((pred_len, 6))
    
    for feature in range(6):
        x = np.arange(len(input_seq))
        y = input_seq[:, feature]
        
        # Handle NaN values
        valid_indices = ~np.isnan(y)
        if np.sum(valid_indices) <= degree:  # Need more points than degree
            predictions[:, feature] = np.nanmean(y) if not np.all(np.isnan(y)) else 0
            continue
            
        x_valid = x[valid_indices]
        y_valid = y[valid_indices]
        
        try:
            coeffs = np.polyfit(x_valid, y_valid, degree)
            poly = np.poly1d(coeffs)
            
            for i in range(pred_len):
                predictions[i, feature] = poly(len(input_seq) + i)
        except:
            # Fallback to mean if polynomial fit fails
            mean_val = np.nanmean(y_valid)
            predictions[:, feature] = mean_val
            
    return predictions

def kalman_predict(input_seq, pred_len, process_noise=0.01, measurement_noise=0.1):
    """Simplified Kalman filter implementation"""
    predictions = np.zeros((pred_len, 6))
    
    for feature in range(6):
        # Get the feature data
        y = input_seq[:, feature]
        
        # Handle NaN values
        valid_indices = ~np.isnan(y)
        if np.sum(valid_indices) == 0:
            predictions[:, feature] = 0
            continue
            
        y_valid = y[valid_indices]
        
        # Initialize Kalman parameters
        x = y_valid[0]  # state (position)
        v = 0           # state (velocity)
        P = np.eye(2)   # covariance matrix
        Q = np.eye(2) * process_noise  # process noise
        R = measurement_noise          # measurement noise
        
        # State transition matrix (constant velocity model)
        F = np.array([[1, 1], [0, 1]])
        
        # Measurement matrix
        H = np.array([1, 0]).reshape(1, 2)
        
        # Process data points
        for z in y_valid[1:]:
            # Predict
            x_pred = F @ np.array([x, v])
            v = x_pred[1]  # Extract velocity
            x = x_pred[0]  # Extract position
            P = F @ P @ F.T + Q
            
            # Update
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
            x_update = x_pred + K @ (np.array([z]) - H @ x_pred)
            P = (np.eye(2) - K @ H) @ P
            
            x = x_update[0]
            v = x_update[1]
        
        # Predict future steps
        for i in range(pred_len):
            x_pred = F @ np.array([x, v])
            v = x_pred[1]
            x = x_pred[0]
            P = F @ P @ F.T + Q
            
            predictions[i, feature] = x
            
    return predictions

# Base Dataset Class (reuse from nsp_with_index)
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

def prepare_test_data(base_dataset, split_info, input_len, pred_len):
    """Prepare test data for a specific input/output configuration"""
    # Create the dataset for this specific input/output config
    dataset = Dataset4NSP(base_dataset, input_len, pred_len)
    
    # Get test instances
    test_instances = split_info['test_instances']
    
    # Map test pair indices based on which instance they belong to
    test_pairs = []
    for pair_idx in range(len(dataset)):
        inst_idx = dataset.get_pair_instance_idx(pair_idx)
        if inst_idx in test_instances:
            test_pairs.append(pair_idx)
    
    # Create test dataset
    test_dataset = torch.utils.data.Subset(dataset, test_pairs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return test_loader, test_pairs, dataset

def get_saved_model_predictions(input_len, pred_len, model_type="limu"):
    """Load saved model predictions using the exact file structure from nsp_with_index.py"""
    config_name = f"input_{input_len}_pred_{pred_len}"
    base_dir = f"{model_type}_nsp/{config_name}"
    model_name = f"{model_type}_nsp_{input_len}_{pred_len}"
    
    # Load predictions and actuals saved during model training
    try:
        predictions = np.load(f"{base_dir}/{model_name}_predictions.npy")
        actuals = np.load(f"{base_dir}/{model_name}_actuals.npy")
        test_pairs = np.load(f"{base_dir}/{model_name}_test_pairs.npy")
        print(f"Successfully loaded {model_type} predictions: {predictions.shape}, actuals: {actuals.shape}")
        return predictions, actuals, test_pairs
    except FileNotFoundError as e:
        print(f"Error loading {model_type} predictions: {e}")
        print(f"Looking for files in: {base_dir}/{model_name}_predictions.npy")
        # Check what files exist in the directory
        if os.path.exists(base_dir):
            print(f"Files in {base_dir}: {os.listdir(base_dir)}")
        else:
            print(f"Directory {base_dir} does not exist")
        return None, None, None

def generate_baseline_predictions(model_func, test_loader, pred_len):
    """Generate predictions for all test sequences using a given baseline model function"""
    all_preds = []
    all_inputs = []
    all_targets = []
    
    for batch in test_loader:
        input_seq, input_pos, target_pos, target_seq = batch
        
        # Convert to numpy and process
        inputs = input_seq.numpy()
        targets = target_seq.numpy()
        
        batch_preds = []
        for seq in inputs:
            pred = model_func(seq, pred_len)
            batch_preds.append(pred)
        
        all_inputs.extend(inputs)
        all_targets.extend(targets)
        all_preds.extend(batch_preds)
    
    return np.array(all_preds), np.array(all_inputs), np.array(all_targets)

def calculate_metrics(predictions, actuals, filter_mask=None):
    """Calculate MSE, MAPE, and Pearson correlation by feature"""
    if filter_mask is not None:
        predictions = predictions[filter_mask]
        actuals = actuals[filter_mask]
    
    metrics = {
        'mse': [],
        'mape': [],
        'pearson': []
    }
    
    for feature in range(6):
        # Extract specific feature values
        pred_feature = predictions[..., feature].flatten()
        actual_feature = actuals[..., feature].flatten()
        
        # Calculate metrics
        mse = mean_squared_error(actual_feature, pred_feature)
        
        # Handle MAPE with care (avoid division by zero)
        non_zero = actual_feature != 0
        mape = np.mean(np.abs((actual_feature[non_zero] - pred_feature[non_zero]) / actual_feature[non_zero])) * 100 if np.any(non_zero) else np.nan
        
        # Calculate Pearson correlation
        try:
            pearson = pearsonr(pred_feature, actual_feature)[0]
        except:
            pearson = np.nan
        
        metrics['mse'].append(mse)
        metrics['mape'].append(mape) 
        metrics['pearson'].append(pearson)
    
    # Calculate overall metrics
    metrics['overall_mse'] = mean_squared_error(actuals.flatten(), predictions.flatten())
    
    non_zero = actuals.flatten() != 0
    metrics['overall_mape'] = np.mean(np.abs((actuals.flatten()[non_zero] - predictions.flatten()[non_zero]) / actuals.flatten()[non_zero])) * 100 if np.any(non_zero) else np.nan
    
    try:
        metrics['overall_pearson'] = pearsonr(predictions.flatten(), actuals.flatten())[0]
    except:
        metrics['overall_pearson'] = np.nan
    
    return metrics

def get_instance_activity_mapping(test_pairs, dataset, activity_labels):
    """Map test pair indices to their corresponding activity labels"""
    pair_to_activity = {}
    
    for idx in test_pairs:
        # Get the instance index for this pair
        inst_idx = dataset.get_pair_instance_idx(idx)
        
        # Get the activity label for this instance
        activity = activity_labels[inst_idx]
        
        # Store mapping
        pair_to_activity[idx] = activity
    
    return pair_to_activity

def analyze_by_activity(predictions, actuals, pair_to_activity):
    """Analyze model performance by activity class"""
    # Get unique activity classes
    activity_classes = np.unique(list(pair_to_activity.values()))
    
    results = {}
    
    for activity in activity_classes:
        # Find pairs with this activity
        activity_pairs = [idx for idx, act in pair_to_activity.items() if act == activity]
        
        # If we have predictions for these pairs
        if activity_pairs:
            # Get corresponding predictions and actuals
            activity_mask = np.array([idx in activity_pairs for idx in range(len(predictions))])
            
            # Calculate metrics for this activity
            metrics = calculate_metrics(predictions, actuals, activity_mask)
            
            results[f"Activity {int(activity)}"] = metrics
    
    return results, activity_classes

def plot_metrics_comparison(model_metrics, metric_name='overall_mse', title=None):
    """Plot a comparison of metrics across models"""
    plt.figure(figsize=(14, 8))
    
    # Prepare data for plotting
    models = list(model_metrics.keys())
    values = [model_metrics[model][metric_name] for model in models]
    
    # Create bar chart with improved styling
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(models)))
    bars = plt.bar(models, values, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', rotation=0, fontsize=9)
    
    plt.title(title or f'Comparison of {metric_name} across Models')
    plt.xlabel('Model')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure to the correct directory
    plt.savefig(f'model_comparison_plots/{metric_name}_comparison.png', dpi=300)
    plt.close()

def plot_activity_comparison(activity_results, model_names, metric_name='overall_mse'):
    """Plot metrics by activity for different models"""
    plt.figure(figsize=(15, 10))
    
    # Prepare data
    activities = list(activity_results[model_names[0]].keys())
    
    x = np.arange(len(activities))
    width = 0.8 / len(model_names)
    
    # Use a color palette
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(model_names)))
    
    for i, model in enumerate(model_names):
        model_values = [activity_results[model][activity][metric_name] for activity in activities]
        plt.bar(x + i*width - 0.4 + width/2, model_values, width, label=model, color=colors[i])
    
    plt.xlabel('Activity Class')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} by Activity Class')
    plt.xticks(x, activities, rotation=45, ha='right')
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f'model_comparison_plots/activity_{metric_name}_comparison.png', dpi=300)
    plt.close()

def plot_feature_metrics(model_metrics, metric_name='mse'):
    """Plot metrics for each feature across models"""
    plt.figure(figsize=(15, 10))
    
    # Prepare data
    models = list(model_metrics.keys())
    features = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
    
    x = np.arange(len(features))
    width = 0.8 / len(models)
    
    # Use a color palette
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(models)))
    
    for i, model in enumerate(models):
        model_values = model_metrics[model][metric_name]
        plt.bar(x + i*width - 0.4 + width/2, model_values, width, label=model, color=colors[i])
    
    plt.xlabel('Feature')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} by Feature')
    plt.xticks(x, features)
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f'model_comparison_plots/feature_{metric_name}_comparison.png', dpi=300)
    plt.close()

def visualize_predictions(model_predictions, actuals, pair_indices, model_names, num_samples=3):
    """Visualize predictions for a few samples"""
    features = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
    
    # Get a distinct color for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for sample_idx in range(min(num_samples, len(pair_indices))):
        plt.figure(figsize=(20, 15))
        
        for f_idx, feature in enumerate(features):
            plt.subplot(3, 2, f_idx+1)
            
            # Plot actual values
            plt.plot(actuals[sample_idx, :, f_idx], 'k-', label='Actual', linewidth=2)
            
            # Plot predictions for each model
            for i, model in enumerate(model_names):
                plt.plot(model_predictions[model][sample_idx, :, f_idx], '--', 
                         label=model, color=colors[i], linewidth=1.5)
            
            plt.title(f'Feature: {feature}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle(f'Sample {sample_idx} Predictions', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        plt.savefig(f'model_comparison_plots/sample_{sample_idx}_predictions.png', dpi=300)
        plt.close()

def summarize_results(model_metrics, activity_results):
    """Create summary tables of results"""
    # Overall metrics summary
    overall_summary = pd.DataFrame({
        'Model': list(model_metrics.keys()),
        'MSE': [m['overall_mse'] for m in model_metrics.values()],
        'MAPE (%)': [m['overall_mape'] for m in model_metrics.values()],
        'Pearson': [m['overall_pearson'] for m in model_metrics.values()]
    })
    
    # Sort by MSE (lower is better)
    overall_summary = overall_summary.sort_values('MSE')
    
    # Activity-wise summary
    activity_dfs = []
    
    for model, activities in activity_results.items():
        for activity, metrics in activities.items():
            activity_df = pd.DataFrame({
                'Model': [model],
                'Activity': [activity],
                'MSE': [metrics['overall_mse']],
                'MAPE (%)': [metrics['overall_mape']],
                'Pearson': [metrics['overall_pearson']]
            })
            activity_dfs.append(activity_df)
    
    activity_summary = pd.concat(activity_dfs, ignore_index=True)
    
    # Feature-wise summary
    feature_dfs = []
    features = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
    
    for model, metrics in model_metrics.items():
        for i, feature in enumerate(features):
            feature_df = pd.DataFrame({
                'Model': [model],
                'Feature': [feature],
                'MSE': [metrics['mse'][i]],
                'MAPE (%)': [metrics['mape'][i]],
                'Pearson': [metrics['pearson'][i]]
            })
            feature_dfs.append(feature_df)
    
    feature_summary = pd.concat(feature_dfs, ignore_index=True)
    
    # Create directories if they don't exist
    os.makedirs('model_comparison_results', exist_ok=True)
    
    # Save to CSV
    overall_summary.to_csv('model_comparison_results/overall_metrics.csv', index=False)
    activity_summary.to_csv('model_comparison_results/activity_metrics.csv', index=False)
    feature_summary.to_csv('model_comparison_results/feature_metrics.csv', index=False)

    
    # Additional stats: Find best model for each activity
    best_models_by_activity = activity_summary.loc[
        activity_summary.groupby('Activity')['MSE'].idxmin()
    ][['Activity', 'Model', 'MSE']]
    
    best_models_by_activity.to_csv('model_comparison_results/best_models_by_activity.csv', index=False)
    
    # Additional stats: Find best model for each feature
    best_models_by_feature = feature_summary.loc[
        feature_summary.groupby('Feature')['MSE'].idxmin()
    ][['Feature', 'Model', 'MSE']]
    
    best_models_by_feature.to_csv('model_comparison_results/best_models_by_feature.csv', index=False)
    
    return overall_summary, activity_summary, feature_summary

def main():
    # Create directories for outputs
    os.makedirs('model_comparison_plots', exist_ok=True)
    os.makedirs('model_comparison_results', exist_ok=True)
    
    # Load data
    normalized_data, labels, filtered_data, kept_indices, split_info = load_data()
    
    # Get activity labels for each instance
    activity_labels = get_majority_activity_labels(labels)
    
    # Create base dataset
    base_dataset = BaseDataset4NSP(filtered_data, kept_indices)
    
    # Dictionary to store all results
    all_predictions = {}
    all_actuals = {}
    all_metrics = {}
    all_activity_results = {}
    
    # Process each input-output pair
    for input_len, pred_len in INPUT_OUTPUT_PAIRS:
        config_name = f"input_{input_len}_pred_{pred_len}"
        print(f"\nProcessing configuration: {config_name}")
        
        # Prepare test data for this configuration
        test_loader, test_pairs, dataset = prepare_test_data(
            base_dataset, split_info, input_len, pred_len
        )
        
        # Get mapping from test pairs to activity labels
        pair_to_activity = get_instance_activity_mapping(test_pairs, dataset, activity_labels)
        
        # Get saved model predictions for LSTM and LIMU models
        lstm_preds, lstm_actuals, lstm_test_pairs = get_saved_model_predictions(input_len, pred_len, "lstm")
     
        
        # Check if we successfully loaded the predictions
        if lstm_preds is None:
            print(f"Skipping configuration {config_name} due to missing prediction files")
            continue
            

        actuals = lstm_actuals
        
        # Generate baseline predictions
        print(f"Generating linear regression predictions for {input_len}->{pred_len}...")
        linear_preds, _, _ = generate_baseline_predictions(
            linear_regression_predict, test_loader, pred_len
        )
        
        print(f"Generating polynomial regression predictions for {input_len}->{pred_len}...")
        poly_preds, _, _ = generate_baseline_predictions(
            polynomial_regression_predict, test_loader, pred_len
        )
        
        print(f"Generating Kalman filter predictions for {input_len}->{pred_len}...")
        kalman_preds, _, _ = generate_baseline_predictions(
            kalman_predict, test_loader, pred_len
        )
        
        # Store predictions
        model_names = [
            f"LSTM_{input_len}_{pred_len}",
            f"Linear_{input_len}_{pred_len}",
            f"Poly_{input_len}_{pred_len}",
            f"Kalman_{input_len}_{pred_len}"
        ]
        
        predictions = {
            model_names[0]: lstm_preds,
            model_names[1]: linear_preds,
            model_names[2]: poly_preds,
            model_names[3]: kalman_preds
        }
        
        # Calculate metrics for each model
        print("Calculating metrics...")
        metrics = {}
        activity_results = {}
        
        for model_name, preds in predictions.items():
            # Overall metrics
            metrics[model_name] = calculate_metrics(preds, actuals)
            
            # Activity-specific metrics
            act_results, activities = analyze_by_activity(preds, actuals, pair_to_activity)
            activity_results[model_name] = act_results
        
        # Store results
        all_predictions.update(predictions)
        all_actuals[f"Actuals_{input_len}_{pred_len}"] = actuals
        all_metrics.update(metrics)
        all_activity_results.update(activity_results)
        
        # Plot metrics comparison for this configuration
        plot_metrics_comparison(
            {name: metrics[name] for name in model_names}, 
            'overall_mse',
            f'MSE Comparison for Input Length {input_len}, Prediction Length {pred_len}'
        )
        
        # Visualize predictions for a few samples
        sample_indices = range(min(5, len(test_pairs)))
        visualize_predictions(
            predictions, actuals, sample_indices, model_names, num_samples=3
        )
    
    # If we didn't process any configurations, exit
    if not all_metrics:
        print("\nNo model predictions were successfully loaded. Please check the file paths and try again.")
        return None, None
    
    # Cross-configuration comparison
    print("\nGenerating cross-configuration comparisons...")
    
    # Plot feature metrics comparison
    plot_feature_metrics(all_metrics, 'mse')
    plot_feature_metrics(all_metrics, 'pearson')
    
    # Create summary tables
    print("Creating summary tables...")
    overall_summary, activity_summary, feature_summary = summarize_results(all_metrics, all_activity_results)
    
    # Calculate and report best model overall and by activity
    best_model_overall = overall_summary.sort_values('MSE').iloc[0]['Model']
    print(f"\nBest model overall (by MSE): {best_model_overall}")
    
    # Group by activity and find best model for each
    best_by_activity = activity_summary.groupby('Activity').apply(
        lambda x: x.sort_values('MSE').iloc[0]['Model']
    )
    print("\nBest model by activity class:")
    for activity, model in best_by_activity.items():
        print(f"  {activity}: {model}")
    
    print("\nAnalysis complete! Results are saved in 'model_comparison_results/' directory and plots in 'model_comparison_plots/' directory.")
    
    return all_metrics, all_activity_results

if __name__ == "__main__":
    main()