# 📚 Student Performance Prediction - Linear Regression Model

A machine learning project that predicts student exam performance using linear regression. This project includes both a Jupyter notebook for model training and a Streamlit web application for interactive predictions.

## 🎯 Project Overview

This project builds a predictive model to estimate final exam scores based on various student behavioral and academic factors. The model is trained on a dataset of 3000+ student records and achieves strong predictive performance.

### Key Features
- **Data Analysis & Visualization**: Comprehensive exploratory data analysis with matplotlib
- **Data Preprocessing**: Handles missing values, categorical encoding, and feature scaling
- **Machine Learning Model**: Linear regression with sklearn Pipeline architecture
- **Interactive Web App**: Streamlit-based interface for making predictions
- **Model Persistence**: Joblib serialization for model reuse

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.806 |
| RMSE | 8.06 |
| MAE | 6.51 |
| MSE | 64.98 |

## 🛠️ Technology Stack

- **Python 3.8+**
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib
- **Web Framework**: Streamlit
- **Model Serialization**: joblib

## 📋 Dataset

The project uses `student_performance_dataset.csv` containing:
- **Size**: 3000+ student records
- **Features**: 30+ behavioral and academic variables
- **Target**: Final exam score (0-100)

### Key Features Include
- **Personal**: Age, Gender, City Type
- **Academic**: Attendance, Assignment Completion, Study Hours
- **Behavioral**: Sleep Hours, Stress Level, Motivation Level, Focus Score
- **Health**: Caffeine Intake, Physical Activity, Mental State

## 📁 Project Structure

```
linear/
├── linear_reg.ipynb                    # Jupyter notebook for model training
├── app.py                              # Streamlit web application
├── requirements.txt                    # Project dependencies
├── readme.md                           # This file
├── student_performance_dataset.csv     # Training dataset
└── linear_regression_model.pkl         # Trained model artifact
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd linear
```

2. **Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Option 1: Interactive Web App (Recommended)

Launch the Streamlit application for an interactive prediction interface:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. You can:
- Adjust student parameters using sliders and dropdowns
- Get real-time predictions
- View performance feedback
- Learn model insights in the sidebar

### Option 2: Jupyter Notebook

For model training, evaluation, and exploration:

```bash
jupyter notebook linear_reg.ipynb
```

The notebook includes:
1. Data loading and exploration
2. Exploratory data analysis (EDA)
3. Data preprocessing and feature engineering
4. Model training and evaluation
5. Predictions and performance metrics
6. Model persistence

## 🎮 How to Use the Web App

1. **Input Student Information**
   - Adjust sliders and select options for each parameter
   - Include realistic values for stress, motivation, and study hours

2. **Make Predictions**
   - Click the "🔮 Predict Final Exam Score" button
   - Get instant predictions with performance feedback

3. **Interpret Results**
   - Score range 0-100
   - Performance levels: Excellent (85+), Good (70-84), Average (60-69), Needs Improvement (<60)

## 📈 Model Workflow

```
Raw Data → Preprocessing → Feature Engineering → Train/Test Split → Model Training → Evaluation → Predictions
```

### Preprocessing Steps
1. **Handling Missing Values**: Median imputation for numeric, mode for categorical
2. **Categorical Encoding**: One-hot encoding for categorical features
3. **Feature Scaling**: StandardScaler for numeric features
4. **Pipeline Integration**: sklearn Pipeline for reproducible preprocessing

## 📝 Model Details

The project uses a **Linear Regression** model implemented through sklearn's Pipeline architecture:

```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first'))
        ]), categorical_features)
    ])),
    ('model', LinearRegression())
])
```

## 🔍 Key Insights

- **Top Predictive Features**: Mental state, consistency score, study hours, stress level
- **Strong Predictors**: Attendance and assignment completion rate
- **Correlation**: Higher study hours and motivation correlate with better exam scores
- **Balance**: Sleep hours and stress management are crucial factors

## 📦 Dependencies

See `requirements.txt` for the complete list:
- streamlit
- scikit-learn
- pandas
- numpy
- joblib

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Fork and create your own version

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created as part of an AI/ML learning project.

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Happy Predicting! 🎓**
