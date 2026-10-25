import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Study Hours Predictor", page_icon="📚", layout="wide")

st.title("📚 Study Hours – Score Predictor")
st.write("Explore the relationship between study hours and previous scores, and estimate a score for selected study hours.")

DATA_FILE = "college_student_performance_analysis.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    required_columns = ["StudyHours", "PreviousScore"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    df = df[required_columns].copy()
    df["StudyHours"] = pd.to_numeric(df["StudyHours"], errors="coerce")
    df["PreviousScore"] = pd.to_numeric(df["PreviousScore"], errors="coerce")
    df = df.dropna(subset=required_columns)
    df = df[df["StudyHours"].between(0, 24)]
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Could not find `{DATA_FILE}`. Upload this CSV into the same GitHub repository folder as app.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if len(df) < 2:
    st.error("The dataset must contain at least two valid rows.")
    st.stop()

X = df[["StudyHours"]]
y = df["PreviousScore"]

model = LinearRegression()
model.fit(X, y)
df["PredictedScore"] = model.predict(X)

tab1, tab2, tab3 = st.tabs(["Dashboard", "Predict Score", "Dataset"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Student Records", f"{len(df):,}")
    c2.metric("Average Study Hours", f"{df['StudyHours'].mean():.2f}")
    c3.metric("Average Previous Score", f"{df['PreviousScore'].mean():.2f}")

    st.subheader("Study Hours vs Previous Score")
    fig, ax = plt.subplots()
    ax.scatter(df["StudyHours"], df["PreviousScore"], label="Actual scores")
    sorted_df = df.sort_values("StudyHours")
    ax.plot(sorted_df["StudyHours"], sorted_df["PredictedScore"], label="Regression line")
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Previous Score")
    ax.legend()
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Model Evaluation")
    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        evaluation_model = LinearRegression()
        evaluation_model.fit(X_train, y_train)
        y_pred = evaluation_model.predict(X_test)
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        m2.metric("RMSE", f"{mean_squared_error(y_test, y_pred) ** 0.5:.2f}")
        m3.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
        st.caption("Evaluation uses a held-out test set. Results may vary with different data.")
    else:
        st.info("At least 5 valid records are recommended to show a train/test evaluation.")

with tab2:
    st.subheader("Predict a Score")
    study_hours = st.number_input(
        "Enter study hours",
        min_value=0.0,
        max_value=24.0,
        value=2.0,
        step=0.5
    )
    if st.button("Predict Score", type="primary"):
        raw_prediction = float(model.predict(pd.DataFrame({"StudyHours": [study_hours]}))[0])
        prediction = max(0.0, min(100.0, raw_prediction))
        st.metric("Estimated Score", f"{prediction:.2f} / 100")
        st.caption("This is an estimate from a simple Linear Regression model trained on PreviousScore. It is not a guarantee of a student's future exam result.")

with tab3:
    st.subheader("Student Performance Dataset")
    st.dataframe(df[["StudyHours", "PreviousScore"]], use_container_width=True)
    st.subheader("Summary Statistics")
    st.dataframe(df[["StudyHours", "PreviousScore"]].describe(), use_container_width=True)
