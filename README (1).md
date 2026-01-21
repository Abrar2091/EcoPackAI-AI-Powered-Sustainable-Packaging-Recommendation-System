🌱 EcoPackAI – AI-Powered Sustainable Packaging Recommendation System

EcoPackAI is an AI-driven web application designed to recommend sustainable and cost-efficient packaging materials based on specific product requirements. The system integrates machine learning models, sustainability metrics, business intelligence dashboards, and cloud deployment to enable data-driven and environmentally responsible packaging decisions for businesses.

📌 Problem Statement

Selecting eco-friendly packaging materials is a complex task that requires balancing multiple factors such as:

Environmental impact (CO₂ emissions, biodegradability)

Material performance (strength, weight capacity, water resistance)

Cost efficiency

Traditional packaging selection methods are largely manual, subjective, and lack quantitative evaluation. These limitations often result in higher environmental impact and inefficient material usage. EcoPackAI addresses this challenge by providing an automated, AI-based recommendation system that objectively evaluates packaging materials using data-driven intelligence.

🎯 Project Objectives

The primary objectives of EcoPackAI are to:

Collect and manage eco-friendly packaging material data

Perform data cleaning and feature engineering

Build machine learning models for cost and CO₂ impact prediction

Recommend optimal packaging materials using AI-based ranking

Develop an interactive web-based user interface

Build a Business Intelligence (BI) dashboard for sustainability insights

Deploy the application on cloud infrastructure

Provide complete technical documentation and project demonstration

🧠 System Overview

EcoPackAI follows a modular, layered architecture:

Data Layer → PostgreSQL database storing material and product data

AI Layer → Machine learning models (Random Forest, XGBoost)

Backend Layer → Flask REST APIs implementing business logic

Frontend Layer → HTML, CSS, Bootstrap, JavaScript

BI Layer → Dashboards, KPIs, sustainability reports

Deployment Layer → Render cloud platform

🛠 Technology Stack

Programming Language → Python

Backend Framework → Flask

Machine Learning → Random Forest, XGBoost

Database → PostgreSQL

Frontend → HTML, CSS, Bootstrap, JavaScript

Data Processing → Pandas, NumPy

Visualization → Chart.js, Matplotlib

Deployment → Render

⚙ Key Features

AI-based packaging material recommendation

Cost efficiency prediction

CO₂ environmental impact estimation

Sustainability scoring and ranking system

Business Intelligence dashboard

Exportable sustainability reports (Excel)

Cloud deployment with PostgreSQL integration

📊 Machine Learning Models

Random Forest Regressor

Used for predicting packaging cost efficiency

Effectively handles non-linear relationships

XGBoost Regressor

Used for predicting CO₂ environmental impact

Provides high accuracy for complex feature patterns

Evaluation Metrics Used:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² Score

🔗 API Endpoints
► /recommend

Method → POST

Purpose → Returns top recommended packaging materials

Inputs → Product weight, product fragility

Output → Ranked materials with predicted cost and CO₂ impact

► /environment-score

Method → POST

Purpose → Evaluates environmental sustainability

Output → CO₂ impact index and material suitability score

📊 Business Intelligence (BI) Dashboard & Sustainability Reporting

The BI dashboard converts AI-generated recommendations into actionable sustainability and cost-optimization insights.

🎯 Dashboard Objectives

Visualize AI recommendations

Compare baseline vs recommended materials

Measure CO₂ emission reduction

Measure cost savings

Generate exportable sustainability reports

📈 Key Metrics

CO₂ Reduction Percentage

Cost Reduction Percentage

Material Performance Score

Sustainability Score (0–100)

📋 Dashboard Modules

Baseline vs Recommended Material Comparison

Material Comparison Table

KPI Cards (Average Cost Reduction, Average CO₂ Reduction)

Material Suitability Distribution

Top Performing Materials

📊 Visualization Types

Pie charts

KPI cards

Comparison tables

📤 Report Export

Material Comparison Report (Excel)

Material Performance Report (Excel)

Sustainability Summary Report (Excel)

🚀 Deployment Details

Cloud Platform → Render

Application Type → Web Service

Database → PostgreSQL Cloud Database

🔗 Live Application:

https://ecopackai-ai-powered-sustainable.onrender.com


▶ How to Run the Project Locally

Clone the project repository

Create a virtual environment

Install required dependencies

Configure PostgreSQL environment variables

Run the Flask application using local_app.py
⚠️ Note: Do not run app.py locally as it is configured for cloud deployment.

📄 Documentation and Demo

Project Report:
https://drive.google.com/drive/folders/1we2qkOkBqkNf7RfMBDoBD1HA0du0E6ED?usp=drive_link

Video Demo:
https://drive.google.com/drive/folders/1ERiK1SKLKj4UAAHjoc1K05HVXELb7yd2?usp=sharing

PPT Presentation:
https://drive.google.com/drive/folders/1C88EpthPoXYgFPY0NxhqSyWGLcWLO9bW?usp=drive_link

📌 Results and Outcomes

Accurate AI-based packaging recommendations

Effective balance between sustainability and cost

Fully deployed cloud-based application

Modular, scalable, and industry-aligned system

Screenshots 

Web UI

<img width="1919" height="960" alt="image" src="https://github.com/user-attachments/assets/2000c6ce-5af6-444e-96c4-561988127a71" />

Recommendation Results 
<img width="1514" height="686" alt="image" src="https://github.com/user-attachments/assets/437b58e4-8aa0-4f86-b6a3-88ef5809bde9" />

Dashboard (Power BI) 
<img width="1919" height="960" alt="image" src="https://github.com/user-attachments/assets/89653d6a-df58-48fc-91be-e624b8b176a0" />

<img width="1919" height="954" alt="image" src="https://github.com/user-attachments/assets/f858fd88-9304-4bf9-b361-d7cb4a72214e" />

<img width="1916" height="960" alt="image" src="https://github.com/user-attachments/assets/cf3de865-1b1d-43c5-a848-8e2df1acfc58" />


Video Demo 

https://github.com/user-attachments/assets/ecc6d0c3-c835-4a89-b29a-33bfcb2ee171




🔮 Future Enhancements

Integration of real-time market pricing

Advanced deep learning models

User authentication and role-based access

Real-time dashboards and analytics

Enterprise-scale deployment with CI/CD pipelines

👨‍🎓 Author

Name → Abrar H
Integrated MCA Student 
Amrita Vishwa Vidyapeetham, Kochi Campus
