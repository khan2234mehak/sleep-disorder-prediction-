# 🌙 SleepSense AI — Sleep Disorder Prediction System v2

A full-stack web application using **Machine Learning** (Random Forest) to predict sleep disorders.

## 🚀 Quick Start

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
python app.py
```

Open: http://localhost:5000
Admin Panel: http://localhost:5000/admin-panel

## 🔐 Default Credentials
| Role    | Email                    | Password  |
|---------|--------------------------|-----------|
| Admin   | admin@sleepsense.ai      | Admin@123 |
| Patient | patient@demo.com         | demo123   |
| Doctor  | doctor@demo.com          | demo123   |

## 📁 Structure
```
sleepsense_ai/
├── app.py                  ← Flask backend
├── train_model.py          ← ML training (real CSV support)
├── requirements.txt
├── schema.sql
├── sleep_disorder.db       ← SQLite (auto-created)
├── templates/
│   ├── index.html          ← Main SPA
│   └── admin_panel.html    ← Admin Panel
├── model/                  ← Pre-trained model artifacts
│   ├── sleep_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── feature_cols.pkl
└── data/                   ← Place Kaggle CSV here (optional)
```

## 🧠 Real Kaggle Dataset (Optional)
Download from: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
Save as: data/sleep_data.csv
Then run: python train_model.py

## ⚙️ Admin Panel Views
- **Dashboard** — KPI cards, disorder donut chart, daily login chart
- **Login History** — Date, time, IP, status; filterable + CSV export
- **User Management** — Expandable rows with per-user login timeline
- **Disorder Summary** — Clickable filter cards + per-user breakdown
- **All Predictions** — Filter by disorder/user/date + CSV export

## 🔌 New API Endpoints (v2)
- GET /api/admin/dashboard           — KPIs + charts
- GET /api/admin/user-login-stats    — Per-user login history
- GET /api/admin/user-disorder-summary — Disorder breakdown per user
- GET /api/admin/predictions         — Filterable predictions
- GET /api/admin/login-logs          — Filterable login logs

Built with ❤️ — SleepSense AI © 2026
