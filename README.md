# ANN Classification - Churn Prediction Project
This project leverages an Artificial Neural Network (ANN) model for customer churn prediction. The model is trained using Keras and TensorFlow, ensuring accurate classification of customers likely to churn. The project includes data preprocessing, training, evaluation, and deployment through a Streamlit web application.

### Project Structure
📂 ANN-Project
│-- 📜 app.py                       # Streamlit application for predictions
│-- 📜 experiments.ipynb            # Notebook for training and evaluating the ANN model
│-- 📜 prediction.ipynb             # Notebook for making predictions using the trained model
│-- 📜 requirements.txt             # Dependencies for the project
│-- 📂 models
│   ├── model.h5                    # Saved ANN model in .h5 format (Keras compatible)
│-- 📂 encoders_scalers
│   ├── label_encoder_gender.pkl    # Pickle file for gender encoding
│   ├── onehot_encoder_geo.pkl      # Pickle file for geography encoding
│   ├── scaler.pkl                  # Pickle file for data scaling
│-- 📂 data
│   ├── Churn_Modelling.csv         # Dataset used for training the model
│-- 📜 README.md                   # Project documentation

### Installation Guide
To set up the project, follow these steps:

### Step 1: Clone the Repository
git clone https://github.com/your-repo/ANN-Project.git
cd ANN-Project

### Step 2: Install Dependencies
Make sure you have Python installed (preferably Python 3.8 or later). Then, install the required dependencies:
pip install -r requirements.txt

### Step 3: Train & Evaluate the Model
Run the experiment.ipynb notebook to train and evaluate the ANN model.

The model will be saved in .h5 format inside the models/ directory.
Encoders and scalers for categorical variables and feature scaling will be stored as .pkl files in the encoders_scalers/ directory.

Alternatively, you can execute the training script programmatically:
python experiment.py

### Step 4: Make Predictions
Execute prediction.ipynb to load the trained model and make predictions.

### Step 5: Run the Churn Prediction App
Once the model is trained, deploy the application using Streamlit:
streamlit run app.py

Your interactive Churn Prediction App is now ready to use!

## Project Features
✅ Artificial Neural Network (ANN): A deep learning model for accurate churn prediction.
✅ Preprocessing Pipeline: Encoders and scalers stored in pickle files for seamless data transformation.
✅ Model Storage: Trained ANN model saved in .h5 format for easy reusability.
✅ User-Friendly Interface: Interactive Streamlit app for real-time customer churn prediction.
✅ Scalability & Flexibility: Easily adaptable for other classification tasks.

### Dependencies
The project relies on the following libraries:

Tensorflow==2.15.0
Pandas 
Numpy 
scikit-learn
tensorboard
matplotlib
streamlit
scikeras


### Conclusion
This project provides an end-to-end solution for customer churn prediction using ANN. With a well-defined training pipeline and a user-friendly web app, it enables businesses to take proactive actions to reduce churn.

### 🚀 Happy Coding! 🚀

