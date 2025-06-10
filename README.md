# Customer Churn Prediction Web Application

This is a web application that predicts whether a customer is likely to churn (leave) based on various customer attributes. The application uses a machine learning model trained on historical customer data.

## Features

- User-friendly web interface
- Real-time churn prediction
- Probability score for predictions
- Input validation
- Responsive design

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository or download the files
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure all the required files are in the same directory:
   - app.py
   - customer_churn_model.pkl
   - encoders.pkl
   - templates/index.html
   - requirements.txt

2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Fill in the customer details in the form
2. Click the "Predict Churn" button
3. View the prediction result and probability score

## Input Fields

- Gender
- Senior Citizen status
- Partner status
- Dependents
- Tenure (in months)
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract type
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

## Model Information

The prediction model is trained on historical customer data and uses various features to predict the likelihood of customer churn. The model provides both a binary prediction (Churn/No Churn) and a probability score. 