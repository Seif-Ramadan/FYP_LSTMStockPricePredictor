import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .data import preprocess_daily_data, create_sequences

def build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
    """
    Build and compile the LSTM model with configurable hyperparameters.
    """
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def train_test_split(X, y, train_size=0.85):
    """
    Split the data into training and testing sets with a given train_size ratio.
    """
    split_index = int(len(X) * train_size)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

def inverse_transform_series(series_scaled, scaler, columns_to_use):
    """
    Inverse transform for a single series using the same scaler
    that was fit on the multi-feature data.
    """
    dummy_array = np.zeros((len(series_scaled), len(columns_to_use)))
    # Place the scaled close prices in the Close column index
    close_idx = list(columns_to_use).index('Close') if 'Close' in columns_to_use else 0
    dummy_array[:, close_idx] = series_scaled
    # Inverse transform
    inv = scaler.inverse_transform(dummy_array)
    return inv[:, close_idx]  # Return the close price column

def calculate_direction_accuracy(actual, predicted):
    """
    Calculate the percentage of times the predicted price movement 
    direction matches the actual direction.
    
    """
    # Convert price series to movement directions (1 for up, -1 for down, 0 for no change)
    directions_actual = np.sign(np.diff(actual))
    directions_pred = np.sign(np.diff(predicted))
    
    # Count matches (including zeros)
    matches = np.sum(directions_actual == directions_pred)
    
    # Calculate accuracy percentage
    accuracy = (matches / len(directions_actual)) * 100
    
    return accuracy

def calculate_metrics(y_test_scaled, y_pred_scaled, scaler, columns_to_use):
    """
    Calculate performance metrics on the actual vs. predicted close prices.
    """
    # Inverse transform for actual and predicted
    y_test_inv = inverse_transform_series(y_test_scaled, scaler, columns_to_use)
    y_pred_inv = inverse_transform_series(y_pred_scaled, scaler, columns_to_use)

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    
    # Calculate direction accuracy
    direction_accuracy = calculate_direction_accuracy(y_test_inv, y_pred_inv)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'direction_accuracy': direction_accuracy,  # New metric
        'y_test_inv': y_test_inv,
        'y_pred_inv': y_pred_inv
    }

def predict_future_prices(model, data, scaler, columns_to_use, sequence_length=90, n_days=5):
    """
    Predict future prices for the next n days.
    Return both predictions and last known actual price.
    """
    # Use the last sequence_length rows from the scaled dataset
    last_sequence = data[-sequence_length:]
    last_sequence = last_sequence.reshape(1, sequence_length, data.shape[1])
    
    # Get the last actual price (for comparison with first prediction)
    close_idx = list(columns_to_use).index('Close') if 'Close' in columns_to_use else 0
    last_actual_scaled = data[-1, close_idx]
    last_actual_price = inverse_transform_series(np.array([last_actual_scaled]), scaler, columns_to_use)[0]
    
    predictions = []
    curr_sequence = last_sequence.copy()
    
    for _ in range(n_days):
        # Predict next day
        next_day_scaled = model.predict(curr_sequence, verbose=0)[0][0]
        predictions.append(next_day_scaled)
        
        # Update sequence for next prediction
        # Remove first day
        new_sequence = curr_sequence[0, 1:].copy()
        
        # Add predicted day at the end
        last_point = curr_sequence[0, -1].copy()
        last_point[close_idx] = next_day_scaled
        
        # Create new sequence
        updated_sequence = np.vstack([new_sequence, last_point])
        curr_sequence = updated_sequence.reshape(1, sequence_length, data.shape[1])
    
    # Convert scaled predictions to actual prices
    predictions_array = np.array(predictions)
    future_prices = inverse_transform_series(predictions_array, scaler, columns_to_use)
    
    return future_prices, last_actual_price

def walk_forward_validation(data, features, sequence_length=90, validation_window=30, 
                            train_min_size=365, step_size=30, lstm_units=50, 
                            dropout_rate=0.2, learning_rate=0.001, epochs=50, 
                            batch_size=32, early_stopping_patience=10):
    
    results = []
    predictions = []
    
    if len(data) < train_min_size + sequence_length + validation_window:
        raise ValueError(f"Not enough data for walk-forward validation. Need at least {train_min_size + sequence_length + validation_window} data points.")
    
    max_steps = (len(data) - train_min_size - sequence_length - validation_window) // step_size + 1
    print(f"Performing walk-forward validation with {max_steps} steps...")
    
    for step in range(max_steps):
        # Calculate indices for this step
        train_end_idx = train_min_size + (step * step_size)
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + validation_window
        if val_end_idx > len(data):
            val_end_idx = len(data)
        
        # Extract training and validation data (keeping time order)
        train_data = data.iloc[:train_end_idx].copy()
        val_data = data.iloc[val_start_idx:val_end_idx].copy()
        
        # Record date ranges for logging
        train_start_date = train_data.index[0]
        train_end_date = train_data.index[-1]
        val_start_date = val_data.index[0]
        val_end_date = val_data.index[-1] if len(val_data) > 0 else None
        
        print(f"\nStep {step+1}/{max_steps}")
        print(f"Training: {train_start_date} to {train_end_date} ({len(train_data)} days)")
        print(f"Validation: {val_start_date} to {val_end_date} ({len(val_data)} days)")
        
        if len(val_data) < 5:  # Arbitrary minimum
            print("Not enough validation data, skipping step.")
            continue
        
        # Preprocess training data: Fit scaler and create training sequences.
        X_train, y_train, scaler, columns_to_use = preprocess_daily_data(
            train_data, use_features=features, sequence_length=sequence_length
        )
        
        # Combine the last sequence_length rows of training data with the validation data.
        combined_val_data = pd.concat([train_data.iloc[-sequence_length:], val_data])
        # Generate validation sequences using the already-fitted scaler.
        X_val, y_val = create_sequences(combined_val_data, scaler, columns_to_use, sequence_length)
        
        print(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")
        
        # Build the LSTM model
        model = build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        # Set up callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # Train the model using a small validation split from the training data.
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Predict on the validation sequences
        y_pred = model.predict(X_val).flatten()
        
        # Calculate performance metrics on the validation set
        metrics = calculate_metrics(y_val, y_pred, scaler, columns_to_use)
        
        # Save results summary for this step
        results.append({
            'step': step + 1,
            'train_start_date': train_start_date,
            'train_end_date': train_end_date,
            'val_start_date': val_start_date,
            'val_end_date': val_end_date,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape'],
            'r2': metrics['r2'],
            'direction_accuracy': metrics['direction_accuracy']
        })
        
        # Save detailed predictions for later plotting
        # Here we assume the number of predictions equals len(metrics['y_test_inv'])
        actual_dates = val_data.index[:len(metrics['y_test_inv'])]
        for i in range(len(actual_dates)):
            predictions.append({
                'step': step + 1,
                'date': actual_dates[i],
                'actual': metrics['y_test_inv'][i],
                'predicted': metrics['y_pred_inv'][i],
                'error': abs(metrics['y_test_inv'][i] - metrics['y_pred_inv'][i])
            })
    
    # Convert results and predictions to DataFrames
    results_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(predictions)
    
    # Calculate summary statistics
    summary = {
        'avg_mae': results_df['mae'].mean(),
        'avg_rmse': results_df['rmse'].mean(),
        'avg_mape': results_df['mape'].mean(),
        'avg_r2': results_df['r2'].mean(),
        'avg_direction_accuracy': results_df['direction_accuracy'].mean(),
        'min_mae': results_df['mae'].min(),
        'max_mae': results_df['mae'].max()
    }
    
    return results_df, predictions_df, summary

def plot_walk_forward_results(results_df, predictions_df, summary, ticker):
    """
    Plot the results of walk-forward validation.
    
    Args:
        results_df: DataFrame with validation results by step
        predictions_df: DataFrame with detailed predictions
        summary: Dictionary with summary metrics
        ticker: Stock symbol
        
    Returns:
        Matplotlib figure with plots
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 15))
    
    # 1. Plot MAE across validation windows
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(results_df['step'], results_df['mae'], marker='o', linestyle='-')
    ax1.axhline(y=summary['avg_mae'], color='r', linestyle='--', alpha=0.7)
    ax1.set_title(f'Mean Absolute Error Across Validation Windows - {ticker}')
    ax1.set_xlabel('Validation Window')
    ax1.set_ylabel('MAE ($)')
    ax1.annotate(f"Avg MAE: ${summary['avg_mae']:.2f}", 
                xy=(0.05, 0.9), xycoords='axes fraction')
    ax1.grid(True, alpha=0.3)
    
    # 2. Plot Direction Accuracy across validation windows
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(results_df['step'], results_df['direction_accuracy'], marker='o', linestyle='-', color='green')
    ax2.axhline(y=summary['avg_direction_accuracy'], color='r', linestyle='--', alpha=0.7)
    ax2.set_title(f'Direction Accuracy Across Validation Windows - {ticker}')
    ax2.set_xlabel('Validation Window')
    ax2.set_ylabel('Direction Accuracy (%)')
    ax2.annotate(f"Avg Direction Accuracy: {summary['avg_direction_accuracy']:.1f}%", 
                xy=(0.05, 0.9), xycoords='axes fraction')
    ax2.grid(True, alpha=0.3)
    
    # 3. Plot predictions for a selected step (use the last step)
    last_step = results_df['step'].max()
    step_predictions = predictions_df[predictions_df['step'] == last_step]
    
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(step_predictions['date'], step_predictions['actual'], label='Actual', marker='o')
    ax3.plot(step_predictions['date'], step_predictions['predicted'], label='Predicted', marker='x')
    ax3.set_title(f'Actual vs Predicted - Validation Window {last_step} - {ticker}')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def calculate_permutation_importance(model, X, y, feature_names, n_repeats=3):
    """
    Calculate permutation importance for each feature in the LSTM model.
    """
    # Get baseline performance
    baseline_pred = model.predict(X, verbose=0).flatten()
    baseline_mse = mean_squared_error(y, baseline_pred)
    
    # Initialize importance scores
    n_features = X.shape[2]
    importance_scores = np.zeros(n_features)
    
    # Calculate importance for each feature with multiple repeats
    for feature_idx in range(n_features):
        feature_importance = 0
        
        for _ in range(n_repeats):
            # Create a copy of the input data
            X_permuted = X.copy()
            
            # Shuffle the feature across the entire dataset
            feature_values = X_permuted[:, :, feature_idx].flatten()
            np.random.shuffle(feature_values)
            X_permuted[:, :, feature_idx] = feature_values.reshape(X_permuted.shape[0], X_permuted.shape[1])
            
            # Predict with the permuted feature and calculate error
            perm_pred = model.predict(X_permuted, verbose=0).flatten()
            perm_mse = mean_squared_error(y, perm_pred)
            
            # Importance is the increase in error
            feature_importance += perm_mse / baseline_mse
        
        # Average across repeats
        importance_scores[feature_idx] = feature_importance / n_repeats
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    
    # Sort by importance
    results = results.sort_values('Importance', ascending=False)
    
    return results

def plot_feature_importance(importance_df, top_n=10, title="Feature Importance for Stock Prediction"):
    """
    Create a horizontal bar chart of feature importance.
    """
    import matplotlib.pyplot as plt
    
    # Ensure we don't try to plot more features than we have
    n_to_plot = min(top_n, len(importance_df))
    
    # Get top N features
    plot_df = importance_df.head(n_to_plot).copy()
    
    # Sort ascending for better visualization
    plot_df = plot_df.sort_values('Importance')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, n_to_plot * 0.4)))
    
    # Plot horizontal bars
    bars = ax.barh(plot_df['Feature'], plot_df['Importance'], color='skyblue')
    
    # Add value labels to the right of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', va='center')
    
    # Customize appearance
    ax.set_xlabel('Importance (Error Ratio)')
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig