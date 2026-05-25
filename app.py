import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------
# Streamlit Page Config (MUST BE FIRST)
# ---------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide"
)

# ---------------------------------------------------
# Load Saved Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model_path = 'linear_regression_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        alt_path = 'linear_regression_pipeline.joblib'
        if os.path.exists(alt_path):
            return joblib.load(alt_path)
        else:
            st.error(f"Model file not found!")
            st.stop()

model = load_model()

# Load the full dataset to understand the feature encoding
@st.cache_data
def load_training_data():
    try:
        data = pd.read_csv('student_performance_dataset.csv')
        return data
    except:
        return None

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("📚 Student Performance Predictor")
st.write(
    "Predict student exam performance using machine learning."
)

st.divider()

# ---------------------------------------------------
# User Inputs
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 15, 30, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city_type = st.selectbox("City Type", ["Urban", "Semi-Urban", "Rural"])
    study_hours = st.slider("Study Hours Per Day", 0.0, 10.0, 5.0, 0.1)
    sleep_hours = st.slider("Sleep Hours Per Night", 3.0, 12.0, 7.0, 0.1)

with col2:
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    motivation_level = st.slider("Motivation Level (1-10)", 1, 10, 6)
    focus_score = st.slider("Focus Score (1-10)", 1, 10, 6)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    assignment_completion = st.slider("Assignment Completion (%)", 0, 100, 80)

# ---------------------------------------------------
# Prepare Data for Prediction
# ---------------------------------------------------

# Create input dataframe in the same format as training data
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'city_type': [city_type],
    'study_hours_per_day': [study_hours],
    'sleep_hours': [sleep_hours],
    'stress_level': [stress_level],
    'motivation_level': [motivation_level],
    'focus_score': [focus_score],
    'attendance_percentage': [attendance],
    'assignment_completion_rate': [assignment_completion]
})

# Apply same preprocessing as training (get_dummies)
# This ensures the model gets features in the same format
try:
    input_processed = pd.get_dummies(input_data, drop_first=True)
    
    # Get the training data to understand feature alignment
    training_data = load_training_data()
    if training_data is not None:
        training_data_processed = training_data.drop('final_exam_score', axis=1)
        training_data_processed = pd.get_dummies(training_data_processed, drop_first=True)
        
        # Align input with training features (add missing columns with 0)
        for col in training_data_processed.columns:
            if col not in input_processed.columns:
                input_processed[col] = 0
        
        # Keep only the columns that exist in training data
        input_processed = input_processed[training_data_processed.columns]
except Exception as e:
    st.error(f"Data preprocessing error: {str(e)}")

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------

if st.button("🔮 Predict Final Exam Score", use_container_width=True):
    try:
        prediction = model.predict(input_processed)
        pred_score = prediction[0]
        
        # Ensure score is within valid range
        pred_score = max(0, min(100, pred_score))

        st.divider()
        st.subheader("📊 Prediction Result")
        
        # Display prediction with metrics
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Predicted Final Exam Score", f"{pred_score:.1f}/100")
        
        with col_metric2:
            # Performance category
            if pred_score >= 85:
                performance = "Excellent 🎯"
            elif pred_score >= 70:
                performance = "Good 👍"
            elif pred_score >= 60:
                performance = "Average 📊"
            else:
                performance = "Needs Improvement 📈"
            
            st.metric("Performance Level", performance)
        
        # Visual feedback
        progress_val = pred_score / 100
        st.progress(progress_val)
        
        if pred_score >= 85:
            st.balloons()
            st.success("Excellent! Keep up the great work! 🌟")
        elif pred_score >= 70:
            st.info("Good performance! Focus on consistency.")
        elif pred_score >= 60:
            st.warning("Average performance. Consider improving study habits.")
        else:
            st.error("Below average. Seek additional support or tutoring.")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info(f"Debug info - Input shape: {input_processed.shape if 'input_processed' in locals() else 'N/A'}")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.header("ℹ️ About This App")

st.sidebar.markdown("""
### How It Works:
1. **Input Features**: Enter student information (study hours, sleep, stress level, etc.)
2. **Data Processing**: Features are encoded to match the model's training format
3. **Prediction**: Linear regression model predicts final exam score
4. **Interpretation**: Results are categorized into performance levels

### Model Performance:
- Trained on 3000+ student records
- Features: 30+ behavioral and academic variables
- Target: Final exam score (0-100)

### Tips for Better Predictions:
- Provide accurate information about your study habits
- Include realistic stress and motivation levels
- Attendance and assignment completion are strong predictors

**Technology**: Streamlit + Scikit-learn
""")