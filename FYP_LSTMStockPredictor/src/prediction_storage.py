import os
import json
from datetime import datetime, timedelta

def save_prediction(symbol, date, prediction, actual=None):
 
    # Create predictions directory if it doesn't exist
    os.makedirs('predictions', exist_ok=True)
    
    # Create file name based on symbol
    file_path = f'predictions/{symbol}_predictions.json'
    
    # Load existing predictions if file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            predictions = json.load(f)
    else:
        predictions = {}
    
    # Add new prediction
    predictions[date] = {
        'prediction': float(prediction),
        'actual': float(actual) if actual is not None else None
    }
    
    # Save predictions
    with open(file_path, 'w') as f:
        json.dump(predictions, f)

def load_predictions(symbol):

    file_path = f'predictions/{symbol}_predictions.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def update_prediction_with_actual(symbol, date, actual):

    predictions = load_predictions(symbol)
    if date in predictions:
        predictions[date]['actual'] = float(actual)
        
        # Save updated predictions
        with open(f'predictions/{symbol}_predictions.json', 'w') as f:
            json.dump(predictions, f)

def compare_predictions(symbol, days_ago=2):

    predictions = load_predictions(symbol)
    
    comparison = []
    today = datetime.now().strftime('%Y-%m-%d')
    
    for date, values in predictions.items():
        prediction_date = datetime.strptime(date, '%Y-%m-%d')
        target_date = datetime.now() - timedelta(days=days_ago)
        
        # Only check predictions made on the target date
        if values['actual'] is not None and abs((prediction_date - target_date).days) <= 1:
            comparison.append({
                'date': date,
                'prediction': values['prediction'],
                'actual': values['actual'],
                'error': abs(values['prediction'] - values['actual']),
                'error_percent': abs(values['prediction'] - values['actual']) / values['actual'] * 100
            })
    
    return comparison