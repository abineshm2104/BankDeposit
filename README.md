                         **Predicting Term Deposit Subscription: A Streamlit-based ML App**
**Project Overview**
This project aims to build a machine learning model to predict whether a bank client will subscribe to a term deposit based on historical marketing campaign data. The model is deployed as an interactive web application using Streamlit, which allows marketing teams to predict subscription likelihood and optimize campaign targeting.

**Project Structure**
app.py: Streamlit application script for creating the web interface.

requirements.txt: List of required Python packages.

model/: Folder containing the trained machine learning model (e.g., xgboost_model.pkl).

notebooks/: Jupyter notebooks for data analysis, model building, and evaluation.

data/: Folder containing the dataset (bank-additional-full.csv).

**Key Features**
**Predict Subscription**: Input client details and predict whether they are likely to subscribe to a term deposit.

**Exploratory Data Analysis (EDA)**: Visualize relationships and insights from the dataset to inform business strategy.

**Model Info**: Display model evaluation metrics, including accuracy, precision, recall, and ROC-AUC.

**Installation & Setup**
**Prerequisites**
Python 3.8 or later

Pip (Python package installer)

**Step 1: Install Dependencies**
Clone the repository and navigate to the project folder:
git clone https://github.com/abineshm2104/BankDeposit
cd term-deposit-app
Install the required dependencies:
pip install -r requirements.txt

**Step 2: Run the Streamlit Application**
To run the app locally, execute:
streamlit run app.py

The app will be accessible in your browser at http://localhost:8501.

**Model Evaluation**
The trained model was evaluated using the following metrics:

**Accuracy**: Measures the proportion of correct predictions.

**Precision**: The proportion of predicted "yes" that were actually "yes".

**Recall**: The proportion of actual "yes" that were predicted correctly.

**F1 Score**: The harmonic mean of precision and recall.

**ROC-AUC**: Measures the overall ability of the model to discriminate between classes.

**Deployment**
The Streamlit app is deployed on AWS EC2. The app can be accessed via the public IP:
http://16.170.171.203:8501

**Conclusion**
This project demonstrates the use of machine learning to improve targeted marketing efforts, reduce costs in telemarketing, and enhance customer experience through predictive analytics. The Streamlit app allows marketing teams to make data-driven decisions on which clients to target for term deposit subscriptions.

License
This project is licensed under the MIT License - see the LICENSE file for details.
