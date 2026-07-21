# Customer Churn Prediction

![Churn](https://user-images.githubusercontent.com/90024661/135493461-457a32f2-c03a-4dfa-a9e7-1d1a362dd5f1.png)

## Overview

Churn prediction means detecting which customers are likely to cancel a subscription to a service based on how they use the service. It is a critical prediction for many businesses because acquiring new clients often costs more than retaining existing ones.

### Why is it important?
Customer churn is a common problem across businesses in many sectors. Every time a client leaves, it represents a significant investment lost. Being able to predict when a client is likely to leave and offer them incentives to stay can offer huge savings to a business.

### About This Project
- **Dataset**: 7,043 customers with 27% churn rate (1,869 customers)
- **Model**: RandomForest Classifier with 200 estimators
- **Accuracy**: ~79% on test data
- **Deployment**: Flask web application with Render deployment support

## Project Structure

```
.
├── app.py                      # Flask application
├── train_model.py              # Model training script
├── Model.sav                   # Trained model (pickle)
├── Telco-Customer-Churn.csv    # Dataset
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render deployment config
├── Procfile                    # Deployment configuration
├── templates/
│   └── home.html              # Web interface
└── TelecomCustomerChurn.ipynb # Jupyter notebook (EDA & analysis)
```

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Installation & Setup

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd End-to-end-project---Customer-churn-main
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Option 1: Run the Web Application (Recommended)

The pre-trained model (`Model.sav`) is already included, so you can run the web application directly:

```bash
python app.py
```

The application will start at `http://127.0.0.1:5000`

Open your browser and navigate to `http://127.0.0.1:5000` to use the prediction interface.

### Option 2: Train the Model from Scratch

If you want to train the model yourself:

```bash
python train_model.py
```

This will:
- Load the dataset from `Telco-Customer-Churn.csv`
- Preprocess the data
- Train a RandomForest Classifier
- Save the model to `Model.sav`
- Display the accuracy score

### Option 3: Explore the Jupyter Notebook

To explore the data analysis and model development:

```bash
jupyter notebook TelecomCustomerChurn.ipynb
```

## Using the Web Application

Once the app is running, you can predict customer churn by entering:

**Categorical Fields:**
- **Dependents**: No, Yes
- **OnlineSecurity**: No, Yes, No internet service
- **OnlineBackup**: Yes, No, No internet service
- **DeviceProtection**: No, Yes, No internet service
- **TechSupport**: No, Yes, No internet service
- **Contract**: Month-to-month, One year, Two year
- **PaperlessBilling**: Yes, No

**Numeric Fields:**
- **tenure**: 0 to 72 (months)
- **MonthlyCharges**: 18.25 to 118.75
- **TotalCharges**: 18.8 to 8684.8

## Deployment

### Deploy to Render

1. Push your code to GitHub
2. Go to [render.com](https://render.com) and create an account
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Render will automatically detect the `render.yaml` configuration
6. Click "Create Web Service"

The app will be deployed at: `https://your-app-name.onrender.com`

## Model Pipeline

1. **Data Analysis (EDA)** - Exploratory data analysis
2. **Data Preprocessing** - Handle missing values, convert data types
3. **Feature Engineering** - Create relevant features
4. **Feature Selection** - SelectKBest for optimal features
5. **Model Training** - RandomForest Classifier
6. **Model Serialization** - Save model using pickle
7. **Web Application** - Flask interface for predictions
8. **Deployment** - Cloud deployment (Render)

## Technologies Used

- **Python** - Programming language
- **Flask** - Web framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Pickle** - Model serialization
- **Gunicorn** - WSGI HTTP Server

## Objective

Predict whether a customer is likely to churn (cancel their subscription) based on their service usage patterns and demographic information.
