# Bank Marketing Model Explorer

A Streamlit web application for exploring machine learning models trained on the UCI Bank Marketing dataset.

## Features

- **Model Selection**: Choose from 5 different classifiers:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Gaussian Naive Bayes
  - Random Forest

- **Metrics Display**: View comprehensive model performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Matthews Correlation Coefficient (MCC) with interpretation
  - ROC AUC Score

- **Visualizations**:
  - Confusion Matrix heatmap
  - ROC Curves

- **Batch Predictions**: Upload CSV files with customer data to get predictions

## Deployment

This app is deployed on Streamlit Cloud. To run locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Expected CSV Format

When uploading a CSV for batch predictions, ensure it has these 16 columns:
- age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome

A sample CSV can be downloaded from the app interface.

## Project Structure

```
streamlit_deploy/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── model/              # Trained model files (.joblib)
├── results_all.json   # Model metrics
└── roc_*.png          # ROC curve images
```

## Dataset

The models are trained on the UCI Bank Marketing dataset, which contains information about bank customers and whether they subscribed to a term deposit.

## License

This project is for educational purposes.

