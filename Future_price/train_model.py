import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

# Sample time-series data (replace with actual data)
data = {
    'Date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'Price': [i + (i * 0.1) for i in range(100)]
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Train an ARIMA model
model = ARIMA(df['Price'], order=(1, 1, 1))
model_fit = model.fit()

# Save the model
joblib.dump(model_fit, 'price_prediction_model.pkl')
print("Model saved as 'price_prediction_model.pkl'")
