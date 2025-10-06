from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load artifacts: model, scaler, and feature list
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xgb_regressor_model.pkl")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
scaler = artifact["scaler"]
features = artifact["features"]
df_clustering = pd.read_csv("datadelevry_cleaned.csv")
clustering_model= joblib.load("clustering.pkl")
classification_model = joblib.load("delivery_delay_model.pkl")
clustering_scaler = joblib.load("scaler.pkl")

FEATURES = [
    'subtotal',
    'total_outstanding_orders',
    'max_item_price',
    'total_items',
    'total_busy_partners',
    'num_distinct_items',
    'total_onshift_partners',
    'order_day',
    'min_item_price',
    'market_id',
    'order_protocol',
    'order_hour'
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/predict_classification', methods=['POST'])
def predict_classification():
    try:
        # Récupérer les données du formulaire
        data = request.get_json()
        
        # Créer un DataFrame avec les données
        input_data_clf = pd.DataFrame({
            'subtotal': [float(data['subtotal'])],
            'total_outstanding_orders': [int(data['total_outstanding_orders'])],
            'max_item_price': [float(data['max_item_price'])],
            'total_items': [int(data['total_items'])],
            'total_busy_partners': [int(data['total_busy_partners'])],
            'num_distinct_items': [int(data['num_distinct_items'])],
            'total_onshift_partners': [int(data['total_onshift_partners'])],
            'order_day': [int(data['order_day'])],
            'min_item_price': [float(data['min_item_price'])],
            'market_id': [int(data['market_id'])],
            'order_protocol': [int(data['order_protocol'])],
            'order_hour': [int(data['order_hour'])]
        })
        
        # Faire la prédiction
        prediction = classification_model.predict(input_data_clf)[0]
        probabilities = classification_model.predict_proba(input_data_clf)[0]
        
        # Préparer la réponse
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Retard' if prediction == 1 else 'À l\'heure',
            'probability_on_time': round(float(probabilities[0]) * 100, 2),
            'probability_delayed': round(float(probabilities[1]) * 100, 2)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route("/clustering_index", methods=["GET"])
def clustering_index():
    stats = df_clustering.describe().to_html(classes='table table-striped', border=0)
    return render_template('clustering_index.html', tables=stats)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure all required fields are present
        missing = [f for f in features if f not in request.form]
        if missing:
            return f"Missing fields: {', '.join(missing)}", 400

        # Convert inputs safely
        try:
            form_data = {f: float(request.form[f]) for f in features}
        except ValueError as ve:
            return f"Invalid input. Please enter numeric values only. Details: {str(ve)}", 400

        # Prepare input DataFrame in correct order
        X_input = pd.DataFrame([form_data])[features]

        # Scale
        X_scaled = scaler.transform(X_input)

        # Predict
        prediction = round(model.predict(X_scaled)[0], 2)

        return render_template("result.html", prediction=prediction, inputs=form_data)

    except Exception as e:
        return f"Unexpected error: {str(e)}", 500

@app.route('/predict_clustering', methods=['GET', 'POST'])
def predict_clustering():
    if request.method == 'POST':
        # Récupérer les valeurs du formulaire
        try:
            input_data = {
    'total_orders': float(request.form['total_orders']),
    'avg_order_value': float(request.form['avg_order_value']),
    'avg_items_per_order': float(request.form['avg_items_per_order']),
    'avg_delivery_time': float(request.form['avg_delivery_time']),
    'driver_utilization_rate': float(request.form['driver_utilization_rate']),
    'price_range': float(request.form['price_range']),
    'revenue_per_item': float(request.form['revenue_per_item']),
    'delivery_time_consistency': float(request.form['delivery_time_consistency']),
    'operational_efficiency': float(request.form['operational_efficiency'])
}

            # Transformer en DataFrame
            X_new = pd.DataFrame([input_data])
            # Mise à l’échelle des nouvelles données avec le scaler entraîné
            X_scaled = scaler.transform(X_new)

            # Prédire le cluster
            cluster = clustering_model.predict(X_new)[0]
            return render_template('clustering_result.html', cluster=cluster, data=input_data)

        except Exception as e:
            return f"Erreur lors de la prédiction : {e}"

    return render_template('clustering_predict.html')

@app.route("/form", methods=["GET"])
def form():
    return render_template("form.html")

@app.route("/form_2", methods=["GET"])
def form_2():
    return render_template("form_2.html")


if __name__ == "__main__":
    app.run(debug=True)
