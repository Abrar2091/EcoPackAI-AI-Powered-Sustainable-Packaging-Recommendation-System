# EcoPackAI – AI-Powered Sustainable Packaging Recommendation System

## 1. Project Title
**EcoPackAI – AI-Powered Sustainable Packaging Recommendation System**

---

## 2. Problem Statement
Traditional packaging selection often prioritizes cost and convenience over sustainability, leading to excessive plastic usage, higher carbon footprints, and environmental harm. Businesses also struggle to balance eco-friendliness with cost efficiency due to a lack of data-driven decision tools.

**EcoPackAI** addresses this problem by providing AI-driven recommendations for sustainable packaging materials based on user requirements. The system helps organizations reduce environmental impact while maintaining cost-effectiveness and operational efficiency.

---

## 3. Objectives
- Recommend eco-friendly packaging solutions using AI-driven logic
- Reduce environmental impact by promoting sustainable materials
- Provide data-driven insights through dashboards and APIs
- Enable informed decision-making for businesses

---

## 4. Tech Stack
- **Frontend:** HTML, CSS, Bootstrap
- **Backend:** Python, Flask
- **Database:** PostgreSQL (Cloud)
- **Deployment:** Render
- **Tools:** GitHub, VS Code

---

## 5. System Architecture
**Application Flow:**

User → Web UI → Flask API → PostgreSQL (Cloud) → Recommendation Response

The user submits packaging requirements through the web interface. The Flask backend processes the input, queries the PostgreSQL cloud database, applies recommendation logic, and returns the most suitable sustainable packaging options.

---

## 6. Features
- User-friendly web interface for packaging input
- AI-based packaging recommendation engine
- BI dashboard for data visualization and insights
- Cloud-hosted PostgreSQL database integration
- RESTful API architecture

---

## 7. Database Design
**Database Name:** Postgres

**Tables:**
- **materials**
  - `id` (Primary Key)
  - `name`
  - `eco_score`
  - `cost`

- **recommendations**
  - `id` (Primary Key)
  - `user_input`
  - `result`
  - `timestamp`

---

## 8. API Endpoints

| Method | Endpoint     | Description                  |
|------|-------------|------------------------------|
| GET  | /           | Home page                    |
| POST | /predict    | Packaging recommendation     |
| GET  | /dashboard  | BI Dashboard                 |

---

## 9. Deployment Details
- **Platform:** Render
- **Database:** PostgreSQL Cloud Instance

**Environment Variables:**
- `DATABASE_URL`
- api_key:`super`

**Live URL:**  
https://your-project.onrender.com](https://ecopackai-ai-powered-sustainable.onrender.com/

---

## 10. How to Run Locally
```bash
git clone https://github.com/Abrar2091/EcoPackAI-AI-Powered-Sustainable-Packaging-Recommendation-System
cd EcoPackAI-AI-Powered-Sustainable-Packaging-Recommendation-System
pip install -r requirements.txt
python module5_api.py
```

---

## 11. Screenshots
- Home Page UI
- Recommendation Output
- BI Dashboard
- PostgreSQL Database Connection

*(Add screenshots here)*

---

## 12. Future Enhancements
- Machine learning model optimization for better recommendations
- User authentication and role-based access
- Advanced analytics and reporting dashboard
- Integration with real-time sustainability datasets

---

## 13. GitHub Repository
https://github.com/Abrar2091/EcoPackAI-AI-Powered-Sustainable-Packaging-Recommendation-System

