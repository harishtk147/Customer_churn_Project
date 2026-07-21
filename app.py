import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / 'Model.sav'

FEATURES = [
    'Dependents',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'Contract',
    'PaperlessBilling',
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]


def preprocess_input(values, category_maps=None):
    data = [[values['Dependents'], values['tenure'], values['OnlineSecurity'], values['OnlineBackup'], values['DeviceProtection'], values['TechSupport'], values['Contract'], values['PaperlessBilling'], values['MonthlyCharges'], values['TotalCharges']]]
    df = pd.DataFrame(data, columns=FEATURES)

    df = df[FEATURES]
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    for feature in FEATURES:
        if feature in {'tenure', 'MonthlyCharges', 'TotalCharges'}:
            continue
        df[feature] = df[feature].astype(str).str.strip()
        if category_maps and feature in category_maps:
            df[feature] = df[feature].map(category_maps[feature]).fillna(-1)
        else:
            df[feature] = df[feature].astype('category').cat.codes

    return df


@app.route("/")
def home_page():
    return render_template('home.html')


@app.route("/", methods=['POST'])
def predict():
    artifact = pickle.load(open(MODEL_PATH, 'rb'))
    if isinstance(artifact, dict):
        model = artifact['model']
        category_maps = artifact.get('category_maps', {})
    else:
        model = artifact
        category_maps = {}

    values = {
        'Dependents': request.form['Dependents'],
        'tenure': float(request.form['tenure']),
        'OnlineSecurity': request.form['OnlineSecurity'],
        'OnlineBackup': request.form['OnlineBackup'],
        'DeviceProtection': request.form['DeviceProtection'],
        'TechSupport': request.form['TechSupport'],
        'Contract': request.form['Contract'],
        'PaperlessBilling': request.form['PaperlessBilling'],
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'TotalCharges': float(request.form['TotalCharges'])
    }

    df = preprocess_input(values, category_maps)

    single = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    probability = probability * 100

    if single[0] == 1:
        op1 = "This Customer is likely to be Churned!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"
    else:
        op1 = "This Customer is likely to be Continue!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"

    return render_template("home.html", op1=op1, op2=op2,
                           Dependents=request.form['Dependents'],
                           tenure=request.form['tenure'],
                           OnlineSecurity=request.form['OnlineSecurity'],
                           OnlineBackup=request.form['OnlineBackup'],
                           DeviceProtection=request.form['DeviceProtection'],
                           TechSupport=request.form['TechSupport'],
                           Contract=request.form['Contract'],
                           PaperlessBilling=request.form['PaperlessBilling'],
                           MonthlyCharges=request.form['MonthlyCharges'],
                           TotalCharges=request.form['TotalCharges'])


if __name__ == '__main__':
    app.run()