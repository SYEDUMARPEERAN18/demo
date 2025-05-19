🌍 AI-Powered Natural Disaster Prediction System

📌 Features

1.Multi-Disaster Prediction:

Supports six disaster types:

Flood

Earthquake

Tsunami

Landslide

Storm

Volcano

2.AI Models for Each Disaster:

A separate machine learning model is trained for each disaster using sample data.

3.Real-Time Simulation:

Uses simulated sensor data (e.g., rainfall, seismic activity, wind speed) to make predictions.

4.Prediction Output:

Classifies each scenario as:

✅ SAFE — if no risk is detected

⚠ RISK — if the conditions resemble historical disaster patterns

5.Visualization:

Bar Chart shows risk status for each disaster.

Line Chart tracks one of the input sensor values for comparison across disasters.

6.Modular Structure:

Clearly separated functions for training, simulating, predicting, and visualizing.

⚙️ How It Works
1.Model Training (train_models()):

Trains a classifier for each disaster using small hardcoded datasets.

Each dataset includes 3 key environmental features.

Models used include:

RandomForest, DecisionTree, LogisticRegression, Naive Bayes, SVM, GradientBoosting

2.Data Simulation (simulate_inputs()):

Simulates sensor input data using random values within realistic ranges.

Example: Earthquake data includes seismic activity, ground displacement, and depth.

3.Prediction (predict_all()):

Uses trained models to predict the risk status (0 = safe, 1 = risk).

Returns formatted results including status and input values.

4.Visualization (plot_results()):

Displays:

A bar chart with disaster risk results.

A line graph of the first sensor input across all disaster types.

5.Main Execution:

Trains models → Simulates data → Predicts → Prints results → Plots graphs.

📊 Data Collection (Simulated for Demo)
Hardcoded Training Data:

Small samples manually provided for each model.

Each sample includes environmental conditions leading to either risk or safe outcomes.

Sensor Input Simulation:

Uses random.uniform() to simulate realistic environmental values.

This simulates a real-time sensor feed in the absence of live data.

In a real-world application, these inputs could come from APIs, IoT sensors, satellites, or weather stations.

🛠 Tools and Libraries Used
Library	Purpose
random	Generate simulated input data
pandas	Handle data in tabular format (DataFrames)
matplotlib.pyplot	Create charts and graphs
seaborn	Advanced visualization styling
warnings	Suppress unnecessary warnings
sklearn	Machine learning models and training

Machine Learning Models from sklearn:
RandomForestClassifier

DecisionTreeClassifier

LogisticRegression

GaussianNB (Naive Bayes)

SVC (Support Vector Classifier)

GradientBoostingClassifier
