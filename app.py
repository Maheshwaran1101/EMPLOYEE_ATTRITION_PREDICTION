import streamlit as st
import pickle
import pandas as pd

# Load trained model and unique values
with open('model_and_key_components.pkl', 'rb') as file:
    saved_components = pickle.load(file)

model = saved_components['model']
unique_values = saved_components['unique_values']


def main():
    st.title("Employee Attrition Prediction App")
    st.sidebar.title("Model Settings")

    # Sidebar - show unique values
    with st.sidebar.expander("View Unique Values"):
        st.write("Unique values for each feature:")
        for column, values in unique_values.items():
            st.write(f"{column}: {values}")

    st.write("This app predicts employee attrition using a trained CatBoost model.")

    # User inputs
    age = st.slider("Age", 18, 70, 30)

    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )

    distance_from_home = st.slider("Distance From Home", 1, 30, 10)

    environment_satisfaction = st.slider(
        "Environment Satisfaction", 1, 4, 2
    )

    hourly_rate = st.slider("Hourly Rate", 30, 100, 65)

    job_involvement = st.slider("Job Involvement", 1, 4, 2)

    job_level = st.slider("Job Level", 1, 5, 3)

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)

    monthly_income = st.slider(
        "Monthly Income", 1000, 20000, 5000
    )

    num_companies_worked = st.slider(
        "Number of Companies Worked", 0, 10, 2
    )

    over_time = st.checkbox("Over Time")

    percent_salary_hike = st.slider(
        "Percent Salary Hike", 10, 25, 15
    )

    relationship_satisfaction = st.slider(
        "Relationship Satisfaction", 1, 4, 2
    )

   
   
   

    work_life_balance = st.slider(
        "Work Life Balance", 1, 4, 2
    )

    years_since_last_promotion = st.slider(
        "Years Since Last Promotion", 0, 15, 3
    )

    years_with_curr_manager = st.slider(
        "Years With Current Manager", 0, 15, 3
    )

    # Input dictionary
    input_dict = {
        'Age': age,
        'Department': department,
        'DistanceFromHome': distance_from_home,
        'EnvironmentSatisfaction': environment_satisfaction,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobRole': job_role,
        'JobSatisfaction': job_satisfaction,
        'MonthlyIncome': monthly_income,
        'NumCompaniesWorked': num_companies_worked,
        'OverTime': over_time,
        'PercentSalaryHike': percent_salary_hike,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': 1,
        'TrainingTimesLastYear': 2,
        'WorkLifeBalance': work_life_balance,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
        
    }

    # Create DataFrame
    input_data = pd.DataFrame([input_dict])

    # Reorder columns to match model training order
    input_data = input_data[model.feature_names_]

    # Predict button
    if st.button("Predict Attrition"):

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]

        predicted_to_leave = prediction[0] == 1

        if predicted_to_leave:
            st.error("Employee is predicted to leave (Attrition = Yes)")

            st.subheader("Suggestions for Retaining the Employee:")
            st.markdown("- Improve career development opportunities")
            st.markdown("- Increase employee engagement")
            st.markdown("- Improve work-life balance")
            st.markdown("- Offer mentorship programs")
            st.markdown("- Review promotion opportunities")

        else:
            st.success("Employee is predicted to stay (Attrition = No)")

        st.write(f"Probability of Attrition: {probability[0]:.2f}")


if __name__ == "__main__":
    main()