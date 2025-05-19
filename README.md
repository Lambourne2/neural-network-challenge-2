# Employee Attrition Prediction with Neural Networks

## Overview
This project implements a deep learning model to predict employee attrition (turnover) based on various employee characteristics and workplace factors. The model is built using scikit-learn's MLPClassifier and demonstrates the application of neural networks in human resources analytics.

## Features
- Data preprocessing and feature engineering for employee data
- Implementation of a neural network classifier using scikit-learn
- Model training with hyperparameter tuning
- Performance evaluation using classification metrics
- Feature importance analysis

## Dataset
The dataset includes various employee attributes such as:
- Personal details (Age, Gender, Marital Status)
- Job-related factors (Job Role, Department, Job Level)
- Work environment (Work-Life Balance, Environment Satisfaction)
- Career progression (Years at Company, Years Since Last Promotion)
- Compensation and benefits (Monthly Income, Stock Option Level)
- And more...

## Technologies Used
- Python 3.x
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Lambourne2/neural-network-challenge-2.git
   cd neural-network-challenge-2
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   If requirements.txt doesn't exist, install the following packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook attrition.ipynb
   ```

2. Run the notebook cells sequentially to:
   - Load and preprocess the data
   - Perform exploratory data analysis
   - Split the data into training and testing sets
   - Build and train the neural network model
   - Evaluate model performance
   - Analyze feature importance

## Model Architecture
The project implements a Multi-layer Perceptron (MLP) classifier with the following characteristics:
- Input layer with scaled numerical features
- Hidden layers with ReLU activation
- Output layer with sigmoid activation for binary classification
- Adam optimizer for training
- Early stopping to prevent overfitting

## Results
The model is evaluated based on:
- Accuracy
- Precision, Recall, and F1-score
- Confusion matrix
- ROC-AUC score

## Feature Importance
The project includes analysis of feature importance to identify key factors contributing to employee attrition, which can provide valuable insights for HR decision-making.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset provided by [data source]
- Built as part of a machine learning challenge