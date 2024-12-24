from django.shortcuts import render
import joblib
from datetime import datetime, timedelta
import numpy as np

# Load the ARIMA model
model = joblib.load('price_prediction_model.pkl')

def index(request):
    return render(request, 'index.html')

def predict_price(request):
    if request.method == 'POST':
        item_name = request.POST['item_name']
        days_ahead = int(request.POST['days_ahead'])

        # Forecast prices
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, days_ahead + 1)]
        predictions = model.forecast(steps=days_ahead)
        predicted_prices = np.round(predictions, 2)

        results = zip(future_dates, predicted_prices)
        return render(request, 'result.html', {'results': results, 'item_name': item_name})

    return render(request, 'index.html')
