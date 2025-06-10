from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
from datetime import datetime

app = Flask(__name__)

# Load the model and encoders
with open('customer_churn_model.pkl', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']  # Extract the actual model from the dictionary
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Initialize feedback storage
FEEDBACK_FILE = 'feedback.json'

def load_feedback():
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_feedback(feedback):
    feedback_list = load_feedback()
    feedback_list.append(feedback)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback_list, f, indent=4)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }

        # Create a DataFrame
        df = pd.DataFrame([data])

        # Encode categorical variables
        for column in df.select_dtypes(include=['object']).columns:
            if column in encoders:
                df[column] = encoders[column].transform(df[column])

        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0]

        result = {
            'prediction': 'Churn' if prediction[0] == 1 else 'No Churn',
            'probability': f"{probability[1]*100:.2f}%" if prediction[0] == 1 else f"{probability[0]*100:.2f}%"
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        feedback_data = {
            'rating': request.form.get('rating'),
            'feedback': request.form.get('feedback'),
            'suggestions': request.form.get('suggestions'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        save_feedback(feedback_data)
        return jsonify({'message': 'Feedback submitted successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 