import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Function to get the latest run ID for a specific experiment
def get_latest_run_id(experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], limit=1)
    return runs[0].info.run_id if runs else None

# Function to load the trained model from the latest run in the specified experiment
def load_model_from_latest_run(experiment_name):
    run_id = get_latest_run_id(experiment_name)
    if run_id:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/RandomForestModel")
        return model
    else:
        return None

# Function to make predictions using the loaded model
def make_predictions(model, X_test):
    return model.predict_proba(X_test)[:, 1]

# Load pickle model
with open('RF_hypa.pkl', 'rb') as file:
    clf = pickle.load(file)

X_train = clf['X_train']
X_test = clf['X_test']
y_train = clf['y_train']
y_test = clf['y_test']

st.write("""

# Hyperparameters Viz. for Random Forest Classifier

This app helps to visualize the effects of **Hyperparameters** on the RF model!

""")

# Sidebar for hyperparameters
st.sidebar.header('User Input Parameters')

st.sidebar.header('Hyperparameters')
n_estimators = st.sidebar.slider('Number of Trees (n_estimators)', min_value=1, max_value=500, value=25, step=1)
max_depth = st.sidebar.slider('Max Depth', min_value=1, max_value=50, value=5, step=1)
criterion = st.sidebar.selectbox('Criterion', ['gini', 'entropy'], index=0)
max_features = st.sidebar.selectbox('Max Features', ['sqrt', 'log2', None], index=0)
max_samples = st.sidebar.slider('Max Samples', min_value=1, max_value=300, value=5, step=1)
bootstrap = st.sidebar.checkbox('Bootstrap', value=True)
oob_score = st.sidebar.checkbox('Out-of-Bag Score', value=False)
verbose = st.sidebar.selectbox('Verbose', [0, 1, 2], index=1)

# Create and train the RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    criterion=criterion,
    max_features=max_features,
    max_samples=max_samples,
    bootstrap=bootstrap,
    oob_score=oob_score,
    verbose=verbose,
    random_state=42
)

# Start a run
with mlflow.start_run():
    try:
        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("max_samples", max_samples)
        mlflow.log_param("bootstrap", bootstrap)
        mlflow.log_param("oob_score", oob_score)
        mlflow.log_param("verbose", verbose)

        # Fit the model
        rf.fit(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(rf, "RandomForestModel")

        # Make predictions
        y_prob = make_predictions(rf, X_test)

        # Log metrics
        accuracy = accuracy_score(y_test, (y_prob > 0.5).astype(int))
        mlflow.log_metric("accuracy", accuracy)

        # Calculating the ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        mlflow.log_metric("roc_auc", roc_auc)

        # Visualize ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')

        # Save the plot as an image
        roc_image_path = "roc_curve_plot.png"
        plt.savefig(roc_image_path)

        # Log ROC plot
        mlflow.log_artifact(roc_image_path, artifact_path="plots")

        # Display the plot in Streamlit
        st.image(roc_image_path)

    finally:
        # Explicitly end the run
        mlflow.end_run()

# Load the trained model from the latest run in the specified experiment
mlflow_model = load_model_from_latest_run("Your Experiment Name")

# Streamlit app for making predictions
st.sidebar.header('Make Predictions')
if mlflow_model:
    prediction = st.sidebar.button('Make Predictions')
    if prediction:
        # Assuming you have some input features for prediction (replace with your own)
        input_features = np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, -1)
        result = mlflow_model.predict(input_features)
        st.sidebar.write("Prediction:", result)
else:
    st.sidebar.warning("No model found. Train a model first.")
