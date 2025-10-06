# train_classification_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv('clean_data.csv')

# Preprocessing
df['created_at'] = pd.to_datetime(df['created_at'])
df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])
df['delivery_duration'] = (df['actual_delivery_time'] - df['created_at']).dt.total_seconds() / 60
df['retard'] = (df['delivery_duration'] > 60).astype(int)

# Feature engineering
df['order_hour'] = df['created_at'].dt.hour
df['order_day'] = df['created_at'].dt.dayofweek

# Select features
features = [
    'subtotal', 'total_outstanding_orders', 'max_item_price', 'total_items',
    'total_busy_partners', 'num_distinct_items', 'total_onshift_partners',
    'order_day', 'min_item_price', 'market_id', 'order_protocol', 'order_hour'
]

X = df[features]
y = df['retard']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=40, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model trained with accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'delivery_delay_model.pkl')
print("Model saved as 'delivery_delay_model.pkl'")