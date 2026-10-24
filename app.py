
from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("college_student_performance_analysis.csv")

# Train the model
X = df[["StudyHours"]]
y = df["PreviousScore"]

model = LinearRegression()
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    study_hours = None
    error = None

    if request.method == "POST":
        try:
            study_hours = float(request.form["study_hours"])

            if not 0 <= study_hours <= 24:
                error = "Enter study hours between 0 and 24."
            else:
                input_data = pd.DataFrame({
                    "StudyHours": [study_hours]
                })

                result = model.predict(input_data)[0]
                prediction = round(max(0, min(100, result)), 2)

        except (ValueError, TypeError):
            error = "Please enter a valid number."

    return render_template(
        "index.html",
        prediction=prediction,
        study_hours=study_hours,
        error=error,

        # Data for the chart
        chart_hours=df["StudyHours"].tolist(),
        chart_marks=df["PreviousScore"].tolist()
    )


if __name__ == "__main__":
    app.run(debug=True)
