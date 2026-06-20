"""
app.py - Sleep Disorder Prediction System - Flask Backend (UPGRADED v2)
New in v2:
  - /api/admin/user-login-stats  → per-user full login history (date, time, IP, status)
  - /api/admin/predictions       → filter by disorder type, date range, username
  - /api/admin/user-disorder-summary → counts: all, None, Insomnia, Sleep Apnea per user
  - /api/admin/dashboard         → unified KPI + disorder breakdown
  - Real Kaggle CSV auto-detection in train_model.py
"""

from flask import Flask, request, jsonify, render_template, session, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from datetime import datetime, timedelta
import pickle, numpy as np, os, csv, io, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────
app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = 'sleep_disorder_secret_2024_change_in_prod'

DB_USER     = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'yourpassword')
DB_HOST     = os.environ.get('DB_HOST', 'localhost')
DB_NAME     = os.environ.get('DB_NAME', 'sleep_disorder_db')

SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), 'sleep_disorder.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

def get_database_uri():
    if DB_PASSWORD and DB_PASSWORD != 'yourpassword':
        return f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    return f"sqlite:///{SQLITE_DB_PATH}"

app.config['SQLALCHEMY_DATABASE_URI'] = get_database_uri()

db     = SQLAlchemy()
bcrypt = Bcrypt()
db.init_app(app)
bcrypt.init_app(app)

# ─────────────────────────────────────────
# DATABASE MODELS
# ─────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id          = db.Column(db.Integer, primary_key=True)
    username    = db.Column(db.String(80),  unique=True, nullable=False)
    email       = db.Column(db.String(120), unique=True, nullable=False)
    password    = db.Column(db.String(255), nullable=False)
    role        = db.Column(db.String(20),  default='patient')
    is_active   = db.Column(db.Boolean,     default=True)
    created_at  = db.Column(db.DateTime,    default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    login_logs  = db.relationship('LoginLog',   backref='user', lazy=True, cascade='all, delete-orphan')

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id                = db.Column(db.Integer, primary_key=True)
    user_id           = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    age               = db.Column(db.Integer)
    gender            = db.Column(db.String(10))
    occupation        = db.Column(db.String(60))
    sleep_duration    = db.Column(db.Float)
    stress_level      = db.Column(db.Integer)
    bmi_category      = db.Column(db.String(20))
    heart_rate        = db.Column(db.Integer)
    daily_steps       = db.Column(db.Integer)
    physical_activity = db.Column(db.Integer)
    systolic_bp       = db.Column(db.Integer)
    diastolic_bp      = db.Column(db.Integer)
    prediction        = db.Column(db.String(30))
    confidence        = db.Column(db.Float)
    sleep_score       = db.Column(db.Integer)
    created_at        = db.Column(db.DateTime, default=datetime.utcnow)

class LoginLog(db.Model):
    __tablename__ = 'login_logs'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(255))
    status     = db.Column(db.String(10))   # success | failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Feedback(db.Model):
    __tablename__ = 'feedback'
    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(80))
    email      = db.Column(db.String(120))
    message    = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ─────────────────────────────────────────
# ML MODEL LOADING
# ─────────────────────────────────────────
MODEL, SCALER, ENCODERS, FEATURE_COLS = None, None, None, None

def make_default_dataset(n=1200):
    np.random.seed(42)
    ages        = np.random.randint(18, 70, n)
    genders     = np.random.choice(['Male','Female'], n)
    occupations = np.random.choice(['Engineer','Doctor','Teacher','Nurse','Accountant',
        'Lawyer','Sales Representative','Software Engineer','Scientist','Manager'], n)
    sleep_dur   = np.round(np.random.uniform(4.5, 9.5, n), 1)
    stress      = np.random.randint(1, 11, n)
    bmi_cat     = np.random.choice(['Underweight','Normal','Overweight','Obese'], n, p=[0.07,0.45,0.30,0.18])
    heart_rate  = np.random.randint(55, 100, n)
    daily_steps = np.random.randint(1500, 15000, n)
    phys_act    = np.random.randint(10, 90, n)
    sys_bp      = np.random.randint(100, 160, n)
    dia_bp      = np.random.randint(65, 105, n)
    labels = []
    for i in range(n):
        score = 0
        if sleep_dur[i] < 6: score += 2
        if stress[i] >= 7:   score += 2
        if heart_rate[i] > 85: score += 1
        if bmi_cat[i] in ['Overweight','Obese']: score += 1
        if daily_steps[i] < 4000: score += 1
        if phys_act[i] < 30: score += 1
        if sys_bp[i] > 140: score += 2
        labels.append('Sleep Apnea' if score >= 6 else ('Insomnia' if score >= 3 else 'None'))
    import pandas as pd
    return pd.DataFrame({'Age':ages,'Gender':genders,'Occupation':occupations,
        'Sleep Duration':sleep_dur,'Stress Level':stress,'BMI Category':bmi_cat,
        'Heart Rate':heart_rate,'Daily Steps':daily_steps,'Physical Activity Level':phys_act,
        'Systolic BP':sys_bp,'Diastolic BP':dia_bp,'Sleep Disorder':labels})

def train_and_save_model():
    """Train model — prefers real Kaggle CSV, falls back to synthetic data."""
    import pandas as pd
    print('Training model...')
    real_csv_paths = ['data/sleep_data.csv', 'sleep_data.csv', 'Sleep_health_and_lifestyle_dataset.csv']
    df = None
    for path in real_csv_paths:
        if os.path.exists(path):
            print(f'✅ Using real dataset: {path}')
            raw = pd.read_csv(path)
            # Normalize Kaggle column names
            col_map = {
                'Person ID': 'Person ID',
                'Age': 'Age', 'Gender': 'Gender', 'Occupation': 'Occupation',
                'Sleep Duration': 'Sleep Duration', 'Quality of Sleep': 'Quality of Sleep',
                'Physical Activity Level': 'Physical Activity Level',
                'Stress Level': 'Stress Level', 'BMI Category': 'BMI Category',
                'Blood Pressure': 'Blood Pressure', 'Heart Rate': 'Heart Rate',
                'Daily Steps': 'Daily Steps', 'Sleep Disorder': 'Sleep Disorder'
            }
            raw.rename(columns={k:v for k,v in col_map.items() if k in raw.columns}, inplace=True)
            # Split Blood Pressure into Systolic / Diastolic
            if 'Blood Pressure' in raw.columns:
                bp_split = raw['Blood Pressure'].str.split('/', expand=True)
                raw['Systolic BP']  = pd.to_numeric(bp_split[0], errors='coerce').fillna(120).astype(int)
                raw['Diastolic BP'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(80).astype(int)
            # Fill missing disorder label (NaN → 'None')
            if 'Sleep Disorder' in raw.columns:
                raw['Sleep Disorder'] = raw['Sleep Disorder'].fillna('None')
            # Normalise BMI Category
            if 'BMI Category' in raw.columns:
                raw['BMI Category'] = raw['BMI Category'].replace({'Normal Weight': 'Normal'})
            # Keep only needed columns
            need = ['Age','Gender','Occupation','Sleep Duration','Stress Level',
                    'BMI Category','Heart Rate','Daily Steps','Physical Activity Level',
                    'Systolic BP','Diastolic BP','Sleep Disorder']
            missing = [c for c in need if c not in raw.columns]
            if missing:
                print(f'  Warning: missing columns {missing} — filling with defaults')
                for m in missing:
                    raw[m] = 0
            df = raw[need].dropna(subset=['Sleep Disorder'])
            print(f'  Loaded {len(df)} rows, disorders: {df["Sleep Disorder"].value_counts().to_dict()}')
            break
    if df is None:
        print('⚠️  No real CSV found — generating synthetic data (1200 samples)')
        print('    To use real data: download from https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset')
        print('    and save as data/sleep_data.csv')
        df = make_default_dataset(1200)

    encoders = {}
    for col in ['Gender','Occupation','BMI Category']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    target_le = LabelEncoder()
    df['Sleep Disorder'] = target_le.fit_transform(df['Sleep Disorder'].astype(str))
    encoders['Sleep Disorder'] = target_le
    fc = ['Age','Gender','Occupation','Sleep Duration','Stress Level','BMI Category',
          'Heart Rate','Daily Steps','Physical Activity Level','Systolic BP','Diastolic BP']
    X = df[fc].astype(float); y = df['Sleep Disorder']
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_split=4,
                                    random_state=42,class_weight='balanced',n_jobs=-1)
    model.fit(X_scaled, y)
    os.makedirs('model', exist_ok=True)
    for name, obj in [('sleep_model',model),('scaler',scaler),('encoders',encoders),('feature_cols',fc)]:
        with open(f'model/{name}.pkl','wb') as f: pickle.dump(obj,f)
    print(f'Model saved. Classes: {list(target_le.classes_)}')

def load_model():
    global MODEL, SCALER, ENCODERS, FEATURE_COLS
    try:
        model_dir = 'model'
        # Also check uploads path if running from a different location
        if not os.path.exists(f'{model_dir}/sleep_model.pkl'):
            alt = os.path.join(os.path.dirname(__file__), 'model')
            if os.path.exists(f'{alt}/sleep_model.pkl'):
                model_dir = alt
        with open(f'{model_dir}/sleep_model.pkl','rb') as f:  MODEL        = pickle.load(f)
        with open(f'{model_dir}/scaler.pkl','rb') as f:       SCALER       = pickle.load(f)
        with open(f'{model_dir}/encoders.pkl','rb') as f:     ENCODERS     = pickle.load(f)
        with open(f'{model_dir}/feature_cols.pkl','rb') as f: FEATURE_COLS = pickle.load(f)
        print('✅ Model loaded.')
    except FileNotFoundError:
        train_and_save_model()
        load_model()

load_model()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
TIPS = {
    'Insomnia': [
        "Maintain a consistent sleep schedule daily.",
        "Avoid screens 1 hour before bed.",
        "Keep bedroom cool (18–20°C), dark, and quiet.",
        "Limit caffeine after 2 PM.",
        "Try relaxation techniques: deep breathing, meditation.",
    ],
    'Sleep Apnea': [
        "Consult an ENT or pulmonologist immediately.",
        "Sleep on your side instead of your back.",
        "Lose excess weight — even 5–10% helps.",
        "Avoid alcohol and sedatives before sleep.",
        "Ask your doctor about CPAP therapy.",
    ],
    'None': [
        "Maintain 7–9 hours of sleep per night.",
        "Exercise regularly — at least 150 min/week.",
        "Keep a sleep diary to track patterns.",
        "Stay hydrated and maintain a healthy diet.",
    ]
}

def compute_sleep_score(d):
    score = 100
    if d['sleep_duration'] < 6:   score -= 20
    elif d['sleep_duration'] < 7: score -= 10
    if d['stress_level'] >= 8:    score -= 15
    elif d['stress_level'] >= 6:  score -= 8
    if d['heart_rate'] > 90:      score -= 10
    if d['bmi_category'] in ['Overweight','Obese']: score -= 10
    if d['daily_steps'] < 4000:   score -= 10
    if d['physical_activity'] < 30: score -= 10
    if d['systolic_bp'] > 140:    score -= 15
    return max(0, min(100, score))

def encode_input(raw):
    enc = dict(raw)
    for col in ['Gender','Occupation','BMI Category']:
        val = enc.get(col)
        le  = ENCODERS.get(col)
        enc[col] = int(le.transform([val])[0]) if (le and val in list(le.classes_)) else 0
    return enc

def admin_required():
    uid = session.get('user_id')
    if not uid: return None, jsonify({'error':'Not authenticated'}), 401
    user = User.query.get(uid)
    if not user or user.role != 'admin': return None, jsonify({'error':'Admin access required'}), 403
    return user, None, None

def parse_date(s):
    """Parse YYYY-MM-DD string to datetime, returns None on failure."""
    try:
        return datetime.strptime(s, '%Y-%m-%d')
    except Exception:
        return None

# ─────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username','').strip()
    email    = data.get('email','').strip()
    password = data.get('password','')
    role     = data.get('role','patient')
    if not username or not email or not password:
        return jsonify({'error':'All fields required'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error':'Email already registered'}), 409
    if User.query.filter_by(username=username).first():
        return jsonify({'error':'Username taken'}), 409
    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    user   = User(username=username, email=email, password=hashed, role=role)
    db.session.add(user)
    db.session.commit()
    session['user_id']  = user.id
    session['username'] = user.username
    session['role']     = user.role
    return jsonify({'message':'Registered','username':username,'role':role}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data     = request.get_json()
    email    = data.get('email','').strip()
    password = data.get('password','')
    ip       = request.remote_addr
    ua       = request.user_agent.string[:255]
    user     = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        if user:
            db.session.add(LoginLog(user_id=user.id, ip_address=ip, user_agent=ua, status='failed'))
            db.session.commit()
        return jsonify({'error':'Invalid credentials'}), 401
    if not user.is_active:
        return jsonify({'error':'Account deactivated. Contact admin.'}), 403
    db.session.add(LoginLog(user_id=user.id, ip_address=ip, user_agent=ua, status='success'))
    db.session.commit()
    session['user_id']  = user.id
    session['username'] = user.username
    session['role']     = user.role
    return jsonify({'message':'Login successful','username':user.username,'role':user.role}), 200

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message':'Logged out'}), 200

@app.route('/api/me', methods=['GET'])
def me():
    uid = session.get('user_id')
    if not uid: return jsonify({'error':'Not authenticated'}), 401
    user = User.query.get(uid)
    return jsonify({'user_id':uid,'username':session['username'],'role':session.get('role','patient'),
                    'email':user.email if user else ''}), 200

# ─────────────────────────────────────────
# PREDICTION ROUTE
# ─────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error':'Model not loaded'}), 503
    data = request.get_json()
    raw = {
        'Age':                     int(data.get('age',30)),
        'Gender':                  data.get('gender','Male'),
        'Occupation':              data.get('occupation','Engineer'),
        'Sleep Duration':          float(data.get('sleep_duration',7)),
        'Stress Level':            int(data.get('stress_level',5)),
        'BMI Category':            data.get('bmi_category','Normal'),
        'Heart Rate':              int(data.get('heart_rate',70)),
        'Daily Steps':             int(data.get('daily_steps',7000)),
        'Physical Activity Level': int(data.get('physical_activity',45)),
        'Systolic BP':             int(data.get('systolic_bp',120)),
        'Diastolic BP':            int(data.get('diastolic_bp',80)),
    }
    enc  = encode_input(raw)
    row  = [enc[col] for col in FEATURE_COLS]
    X    = SCALER.transform([row])
    pred_idx   = MODEL.predict(X)[0]
    proba      = MODEL.predict_proba(X)[0]
    label      = ENCODERS['Sleep Disorder'].inverse_transform([pred_idx])[0]
    confidence = float(round(max(proba)*100,2))
    all_probs  = {cls:float(round(p*100,2)) for cls,p in zip(ENCODERS['Sleep Disorder'].classes_, proba)}
    helper     = {'sleep_duration':raw['Sleep Duration'],'stress_level':raw['Stress Level'],
                  'heart_rate':raw['Heart Rate'],'bmi_category':raw['BMI Category'],
                  'daily_steps':raw['Daily Steps'],'physical_activity':raw['Physical Activity Level'],
                  'systolic_bp':raw['Systolic BP']}
    sleep_score = compute_sleep_score(helper)
    tips        = TIPS.get(label, TIPS['None'])
    if 'user_id' in session:
        p = Prediction(user_id=session['user_id'],age=raw['Age'],gender=raw['Gender'],
            occupation=raw['Occupation'],sleep_duration=raw['Sleep Duration'],
            stress_level=raw['Stress Level'],bmi_category=raw['BMI Category'],
            heart_rate=raw['Heart Rate'],daily_steps=raw['Daily Steps'],
            physical_activity=raw['Physical Activity Level'],systolic_bp=raw['Systolic BP'],
            diastolic_bp=raw['Diastolic BP'],prediction=label,confidence=confidence,sleep_score=sleep_score)
        db.session.add(p); db.session.commit()
    return jsonify({'prediction':label,'confidence':confidence,'all_probabilities':all_probs,
                    'sleep_score':sleep_score,'tips':tips}), 200

# ─────────────────────────────────────────
# USER HISTORY
# ─────────────────────────────────────────
@app.route('/api/history', methods=['GET'])
def history():
    if 'user_id' not in session: return jsonify({'error':'Not authenticated'}), 401
    records = Prediction.query.filter_by(user_id=session['user_id'])\
                .order_by(Prediction.created_at.desc()).limit(50).all()
    return jsonify([{
        'id':r.id,'prediction':r.prediction,'confidence':r.confidence,
        'sleep_score':r.sleep_score,'sleep_duration':r.sleep_duration,
        'stress_level':r.stress_level,'age':r.age,'gender':r.gender,
        'bmi_category':r.bmi_category,'heart_rate':r.heart_rate,
        'daily_steps':r.daily_steps,'occupation':r.occupation,
        'physical_activity':r.physical_activity,'systolic_bp':r.systolic_bp,
        'diastolic_bp':r.diastolic_bp,
        'created_at':r.created_at.strftime('%Y-%m-%d %H:%M')
    } for r in records]), 200

# ─────────────────────────────────────────
# USER: DOWNLOAD OWN REPORT (CSV)
# ─────────────────────────────────────────
@app.route('/api/report/download', methods=['GET'])
def download_my_report():
    if 'user_id' not in session: return jsonify({'error':'Not authenticated'}), 401
    records = Prediction.query.filter_by(user_id=session['user_id'])\
                .order_by(Prediction.created_at.desc()).all()
    si  = io.StringIO()
    cw  = csv.writer(si)
    cw.writerow(['Date','Age','Gender','Occupation','Sleep Duration (hrs)','Stress Level',
                 'BMI Category','Heart Rate','Daily Steps','Physical Activity',
                 'Systolic BP','Diastolic BP','Prediction','Confidence (%)','Sleep Score'])
    for r in records:
        cw.writerow([r.created_at.strftime('%Y-%m-%d %H:%M'),r.age,r.gender,r.occupation,
                     r.sleep_duration,r.stress_level,r.bmi_category,r.heart_rate,
                     r.daily_steps,r.physical_activity,r.systolic_bp,r.diastolic_bp,
                     r.prediction,r.confidence,r.sleep_score])
    output = make_response(si.getvalue())
    output.headers['Content-Disposition'] = f'attachment; filename=sleep_report_{session["username"]}.csv'
    output.headers['Content-Type'] = 'text/csv'
    return output

# ─────────────────────────────────────────
# ADMIN: STATS OVERVIEW
# ─────────────────────────────────────────
@app.route('/api/admin/stats', methods=['GET'])
def admin_stats():
    user, err, code = admin_required()
    if err: return err, code
    total_users   = User.query.count()
    total_preds   = Prediction.query.count()
    total_logins  = LoginLog.query.filter_by(status='success').count()
    disorder_dist = {}
    for row in db.session.query(Prediction.prediction, db.func.count(Prediction.id))\
                         .group_by(Prediction.prediction).all():
        disorder_dist[row[0]] = row[1]
    return jsonify({
        'total_users':total_users,'total_predictions':total_preds,
        'total_logins':total_logins,'disorder_distribution':disorder_dist
    }), 200

# ─────────────────────────────────────────
# ADMIN: ALL USERS + LAST LOGIN
# ─────────────────────────────────────────
@app.route('/api/admin/users', methods=['GET'])
def admin_users():
    user, err, code = admin_required()
    if err: return err, code
    users = User.query.order_by(User.created_at.desc()).all()
    result = []
    for u in users:
        last_login = LoginLog.query.filter_by(user_id=u.id, status='success')\
                        .order_by(LoginLog.created_at.desc()).first()
        pred_count = Prediction.query.filter_by(user_id=u.id).count()
        result.append({
            'id':u.id,'username':u.username,'email':u.email,'role':u.role,
            'is_active':u.is_active,'pred_count':pred_count,
            'created_at':u.created_at.strftime('%Y-%m-%d'),
            'last_login':last_login.created_at.strftime('%Y-%m-%d %H:%M') if last_login else 'Never',
            'last_login_ip':last_login.ip_address if last_login else '—'
        })
    return jsonify(result), 200

@app.route('/api/admin/users/<int:uid>/toggle', methods=['POST'])
def admin_toggle_user(uid):
    user, err, code = admin_required()
    if err: return err, code
    target = User.query.get_or_404(uid)
    target.is_active = not target.is_active
    db.session.commit()
    return jsonify({'is_active':target.is_active}), 200

@app.route('/api/admin/users/<int:uid>/delete', methods=['DELETE'])
def admin_delete_user(uid):
    user, err, code = admin_required()
    if err: return err, code
    target = User.query.get_or_404(uid)
    db.session.delete(target); db.session.commit()
    return jsonify({'message':'User deleted'}), 200

# ─────────────────────────────────────────
# ADMIN: LOGIN LOGS (with date/time filter)
# ─────────────────────────────────────────
@app.route('/api/admin/login-logs', methods=['GET'])
def admin_login_logs():
    """
    Query params:
      username   — filter by username (partial match)
      status     — 'success' | 'failed' | '' (all)
      date_from  — YYYY-MM-DD
      date_to    — YYYY-MM-DD
      limit      — max rows (default 300)
    """
    user, err, code = admin_required()
    if err: return err, code

    username   = request.args.get('username', '').strip()
    status     = request.args.get('status', '').strip()
    date_from  = parse_date(request.args.get('date_from', ''))
    date_to    = parse_date(request.args.get('date_to', ''))
    limit      = min(int(request.args.get('limit', 300)), 1000)

    q = db.session.query(LoginLog, User.username, User.email, User.role)\
          .join(User, LoginLog.user_id == User.id)

    if username:
        q = q.filter(User.username.ilike(f'%{username}%'))
    if status:
        q = q.filter(LoginLog.status == status)
    if date_from:
        q = q.filter(LoginLog.created_at >= date_from)
    if date_to:
        q = q.filter(LoginLog.created_at < date_to + timedelta(days=1))

    logs = q.order_by(LoginLog.created_at.desc()).limit(limit).all()

    return jsonify([{
        'id':       log.id,
        'username': uname,
        'email':    email,
        'role':     role,
        'ip_address': log.ip_address,
        'status':   log.status,
        'user_agent': (log.user_agent or '')[:100],
        'date':     log.created_at.strftime('%Y-%m-%d'),
        'time':     log.created_at.strftime('%H:%M:%S'),
        'created_at': log.created_at.strftime('%Y-%m-%d %H:%M:%S')
    } for log, uname, email, role in logs]), 200

# ─────────────────────────────────────────
# ADMIN: PER-USER FULL LOGIN HISTORY  ← NEW
# ─────────────────────────────────────────
@app.route('/api/admin/user-login-stats', methods=['GET'])
def admin_user_login_stats():
    """
    Returns a list of all users with their complete login history.
    Query params:
      user_id   — restrict to a single user
      date_from — YYYY-MM-DD
      date_to   — YYYY-MM-DD
    Response:
      [
        {
          user_id, username, email, role,
          total_logins, successful_logins, failed_logins,
          first_login, last_login,
          login_history: [ {date, time, ip, status}, ... ]
        },
        ...
      ]
    """
    admin, err, code = admin_required()
    if err: return err, code

    uid_filter = request.args.get('user_id', type=int)
    date_from  = parse_date(request.args.get('date_from', ''))
    date_to    = parse_date(request.args.get('date_to', ''))

    users_q = User.query.order_by(User.username)
    if uid_filter:
        users_q = users_q.filter_by(id=uid_filter)
    users = users_q.all()

    result = []
    for u in users:
        log_q = LoginLog.query.filter_by(user_id=u.id)
        if date_from:
            log_q = log_q.filter(LoginLog.created_at >= date_from)
        if date_to:
            log_q = log_q.filter(LoginLog.created_at < date_to + timedelta(days=1))
        logs = log_q.order_by(LoginLog.created_at.desc()).all()

        history = [{
            'date':     lg.created_at.strftime('%Y-%m-%d'),
            'time':     lg.created_at.strftime('%H:%M:%S'),
            'datetime': lg.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'ip':       lg.ip_address or '—',
            'status':   lg.status,
            'user_agent': (lg.user_agent or '')[:80]
        } for lg in logs]

        success_logs = [l for l in logs if l.status == 'success']
        failed_logs  = [l for l in logs if l.status == 'failed']

        result.append({
            'user_id':          u.id,
            'username':         u.username,
            'email':            u.email,
            'role':             u.role,
            'is_active':        u.is_active,
            'registered_at':    u.created_at.strftime('%Y-%m-%d %H:%M'),
            'total_logins':     len(logs),
            'successful_logins':len(success_logs),
            'failed_logins':    len(failed_logs),
            'first_login':      logs[-1].created_at.strftime('%Y-%m-%d %H:%M:%S') if logs else None,
            'last_login':       logs[0].created_at.strftime('%Y-%m-%d %H:%M:%S')  if logs else None,
            'login_history':    history
        })

    return jsonify(result), 200

# ─────────────────────────────────────────
# ADMIN: ALL PREDICTIONS (with filters)  ← UPGRADED
# ─────────────────────────────────────────
@app.route('/api/admin/predictions', methods=['GET'])
def admin_predictions():
    """
    Query params:
      disorder   — 'None' | 'Insomnia' | 'Sleep Apnea'
      username   — partial match
      date_from  — YYYY-MM-DD
      date_to    — YYYY-MM-DD
      limit      — default 500
    """
    user, err, code = admin_required()
    if err: return err, code

    disorder  = request.args.get('disorder', '').strip()
    username  = request.args.get('username', '').strip()
    date_from = parse_date(request.args.get('date_from', ''))
    date_to   = parse_date(request.args.get('date_to', ''))
    limit     = min(int(request.args.get('limit', 500)), 2000)

    q = db.session.query(Prediction, User.username)\
          .join(User, Prediction.user_id == User.id)

    if disorder:
        q = q.filter(Prediction.prediction == disorder)
    if username:
        q = q.filter(User.username.ilike(f'%{username}%'))
    if date_from:
        q = q.filter(Prediction.created_at >= date_from)
    if date_to:
        q = q.filter(Prediction.created_at < date_to + timedelta(days=1))

    rows = q.order_by(Prediction.created_at.desc()).limit(limit).all()

    return jsonify([{
        'id':p.id,'username':uname,'prediction':p.prediction,'confidence':p.confidence,
        'sleep_score':p.sleep_score,'age':p.age,'gender':p.gender,'occupation':p.occupation,
        'sleep_duration':p.sleep_duration,'stress_level':p.stress_level,
        'bmi_category':p.bmi_category,'heart_rate':p.heart_rate,
        'daily_steps':p.daily_steps,'physical_activity':p.physical_activity,
        'systolic_bp':p.systolic_bp,'diastolic_bp':p.diastolic_bp,
        'date':p.created_at.strftime('%Y-%m-%d'),
        'time':p.created_at.strftime('%H:%M'),
        'created_at':p.created_at.strftime('%Y-%m-%d %H:%M')
    } for p,uname in rows]), 200

# ─────────────────────────────────────────
# ADMIN: USER × DISORDER SUMMARY  ← NEW
# ─────────────────────────────────────────
@app.route('/api/admin/user-disorder-summary', methods=['GET'])
def admin_user_disorder_summary():
    """
    Returns per-user prediction counts broken down by disorder type.
    Query params:
      disorder  — filter to show only users with this disorder
      username  — partial match
    Response:
      [
        {
          user_id, username, email, role,
          total, none_count, insomnia_count, sleep_apnea_count,
          last_prediction
        }
      ]
    """
    admin, err, code = admin_required()
    if err: return err, code

    disorder_filter = request.args.get('disorder', '').strip()
    username_filter = request.args.get('username', '').strip()

    users = User.query.order_by(User.username).all()
    result = []

    for u in users:
        if username_filter and username_filter.lower() not in u.username.lower():
            continue

        preds = Prediction.query.filter_by(user_id=u.id).all()
        none_c     = sum(1 for p in preds if p.prediction == 'None')
        insomnia_c = sum(1 for p in preds if p.prediction == 'Insomnia')
        apnea_c    = sum(1 for p in preds if p.prediction == 'Sleep Apnea')
        total      = len(preds)

        # Apply disorder filter — only include users who have at least one prediction of this type
        if disorder_filter:
            if disorder_filter == 'None'        and none_c == 0:     continue
            if disorder_filter == 'Insomnia'    and insomnia_c == 0: continue
            if disorder_filter == 'Sleep Apnea' and apnea_c == 0:    continue

        last_pred = max((p.created_at for p in preds), default=None)

        result.append({
            'user_id':       u.id,
            'username':      u.username,
            'email':         u.email,
            'role':          u.role,
            'is_active':     u.is_active,
            'total':         total,
            'none_count':    none_c,
            'insomnia_count':insomnia_c,
            'apnea_count':   apnea_c,
            'last_prediction': last_pred.strftime('%Y-%m-%d %H:%M') if last_pred else None
        })

    return jsonify(result), 200

# ─────────────────────────────────────────
# ADMIN: DASHBOARD (unified KPIs + charts)
# ─────────────────────────────────────────
@app.route('/api/admin/dashboard', methods=['GET'])
def admin_dashboard():
    """
    Unified endpoint for admin overview dashboard.
    Returns KPIs, disorder distribution, daily login counts (last 30 days),
    and monthly prediction counts.
    """
    admin, err, code = admin_required()
    if err: return err, code

    total_users  = User.query.count()
    total_preds  = Prediction.query.count()
    total_logins = LoginLog.query.filter_by(status='success').count()
    failed_logins= LoginLog.query.filter_by(status='failed').count()

    # Disorder distribution
    disorder_dist = {}
    for row in db.session.query(Prediction.prediction, db.func.count(Prediction.id))\
                         .group_by(Prediction.prediction).all():
        disorder_dist[row[0] or 'Unknown'] = row[1]

    # Daily logins — last 30 days
    thirty_ago = datetime.utcnow() - timedelta(days=30)
    daily_logs_raw = db.session.query(
        db.func.date(LoginLog.created_at).label('day'),
        db.func.count(LoginLog.id).label('cnt')
    ).filter(LoginLog.created_at >= thirty_ago, LoginLog.status == 'success')\
     .group_by(db.func.date(LoginLog.created_at))\
     .order_by(db.func.date(LoginLog.created_at)).all()
    daily_logins = [{'date': str(r.day), 'count': r.cnt} for r in daily_logs_raw]

    # Monthly predictions — last 6 months
    six_months_ago = datetime.utcnow() - timedelta(days=180)
    monthly_preds_raw = db.session.query(
        db.func.strftime('%Y-%m', Prediction.created_at).label('month'),
        db.func.count(Prediction.id).label('cnt')
    ).filter(Prediction.created_at >= six_months_ago)\
     .group_by(db.func.strftime('%Y-%m', Prediction.created_at))\
     .order_by(db.func.strftime('%Y-%m', Prediction.created_at)).all()
    monthly_preds = [{'month': r.month, 'count': r.cnt} for r in monthly_preds_raw]

    # Active users today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    active_today = LoginLog.query.filter(
        LoginLog.status == 'success',
        LoginLog.created_at >= today_start
    ).with_entities(LoginLog.user_id).distinct().count()

    return jsonify({
        'kpis': {
            'total_users':    total_users,
            'total_preds':    total_preds,
            'total_logins':   total_logins,
            'failed_logins':  failed_logins,
            'active_today':   active_today,
        },
        'disorder_distribution': disorder_dist,
        'daily_logins':          daily_logins,
        'monthly_predictions':   monthly_preds,
    }), 200

# ─────────────────────────────────────────
# ADMIN: DOWNLOAD CSVs
# ─────────────────────────────────────────
@app.route('/api/admin/report/download', methods=['GET'])
def admin_download_report():
    """Download all predictions as CSV. Supports same filter params as /api/admin/predictions."""
    user, err, code = admin_required()
    if err: return err, code

    disorder  = request.args.get('disorder', '').strip()
    date_from = parse_date(request.args.get('date_from', ''))
    date_to   = parse_date(request.args.get('date_to', ''))

    q = db.session.query(Prediction, User.username, User.email)\
          .join(User, Prediction.user_id == User.id)
    if disorder:
        q = q.filter(Prediction.prediction == disorder)
    if date_from:
        q = q.filter(Prediction.created_at >= date_from)
    if date_to:
        q = q.filter(Prediction.created_at < date_to + timedelta(days=1))
    rows = q.order_by(Prediction.created_at.desc()).all()

    si  = io.StringIO()
    cw  = csv.writer(si)
    cw.writerow(['Date','Time','Username','Email','Age','Gender','Occupation','Sleep Duration',
                 'Stress Level','BMI Category','Heart Rate','Daily Steps',
                 'Physical Activity','Systolic BP','Diastolic BP',
                 'Prediction','Confidence (%)','Sleep Score'])
    for p, uname, email in rows:
        cw.writerow([p.created_at.strftime('%Y-%m-%d'), p.created_at.strftime('%H:%M:%S'),
                     uname, email, p.age, p.gender, p.occupation, p.sleep_duration,
                     p.stress_level, p.bmi_category, p.heart_rate, p.daily_steps,
                     p.physical_activity, p.systolic_bp, p.diastolic_bp,
                     p.prediction, p.confidence, p.sleep_score])
    output = make_response(si.getvalue())
    output.headers['Content-Disposition'] = 'attachment; filename=all_predictions_report.csv'
    output.headers['Content-Type'] = 'text/csv'
    return output

@app.route('/api/admin/users/report/download', methods=['GET'])
def admin_download_users():
    user, err, code = admin_required()
    if err: return err, code
    users = User.query.order_by(User.created_at.desc()).all()
    si  = io.StringIO()
    cw  = csv.writer(si)
    cw.writerow(['ID','Username','Email','Role','Status','Registered On','Total Predictions',
                 'None','Insomnia','Sleep Apnea','Last Login'])
    for u in users:
        preds      = Prediction.query.filter_by(user_id=u.id).all()
        none_c     = sum(1 for p in preds if p.prediction == 'None')
        insomnia_c = sum(1 for p in preds if p.prediction == 'Insomnia')
        apnea_c    = sum(1 for p in preds if p.prediction == 'Sleep Apnea')
        last_login = LoginLog.query.filter_by(user_id=u.id, status='success')\
                        .order_by(LoginLog.created_at.desc()).first()
        cw.writerow([u.id, u.username, u.email, u.role,
                     'Active' if u.is_active else 'Inactive',
                     u.created_at.strftime('%Y-%m-%d'), len(preds),
                     none_c, insomnia_c, apnea_c,
                     last_login.created_at.strftime('%Y-%m-%d %H:%M') if last_login else 'Never'])
    output = make_response(si.getvalue())
    output.headers['Content-Disposition'] = 'attachment; filename=users_report.csv'
    output.headers['Content-Type'] = 'text/csv'
    return output

@app.route('/api/admin/login-logs/download', methods=['GET'])
def admin_download_logs():
    """Download login logs CSV. Supports same filter params as /api/admin/login-logs."""
    user, err, code = admin_required()
    if err: return err, code

    username  = request.args.get('username', '').strip()
    status    = request.args.get('status', '').strip()
    date_from = parse_date(request.args.get('date_from', ''))
    date_to   = parse_date(request.args.get('date_to', ''))

    q = db.session.query(LoginLog, User.username, User.email)\
          .join(User, LoginLog.user_id == User.id)
    if username:
        q = q.filter(User.username.ilike(f'%{username}%'))
    if status:
        q = q.filter(LoginLog.status == status)
    if date_from:
        q = q.filter(LoginLog.created_at >= date_from)
    if date_to:
        q = q.filter(LoginLog.created_at < date_to + timedelta(days=1))
    logs = q.order_by(LoginLog.created_at.desc()).all()

    si  = io.StringIO()
    cw  = csv.writer(si)
    cw.writerow(['Date','Time','Username','Email','IP Address','Status','User Agent'])
    for log, uname, email in logs:
        cw.writerow([log.created_at.strftime('%Y-%m-%d'), log.created_at.strftime('%H:%M:%S'),
                     uname, email, log.ip_address, log.status, log.user_agent or ''])
    output = make_response(si.getvalue())
    output.headers['Content-Disposition'] = 'attachment; filename=login_logs.csv'
    output.headers['Content-Type'] = 'text/csv'
    return output

@app.route('/api/admin/user-login-stats/download', methods=['GET'])
def admin_download_user_login_stats():
    """Download per-user login summary CSV."""
    admin, err, code = admin_required()
    if err: return err, code

    users = User.query.order_by(User.username).all()
    si  = io.StringIO()
    cw  = csv.writer(si)
    cw.writerow(['Username','Email','Role','Total Logins','Successful','Failed',
                 'First Login','Last Login'])
    for u in users:
        logs    = LoginLog.query.filter_by(user_id=u.id).order_by(LoginLog.created_at).all()
        success = [l for l in logs if l.status == 'success']
        failed  = [l for l in logs if l.status == 'failed']
        cw.writerow([
            u.username, u.email, u.role, len(logs), len(success), len(failed),
            logs[0].created_at.strftime('%Y-%m-%d %H:%M')  if logs else 'Never',
            logs[-1].created_at.strftime('%Y-%m-%d %H:%M') if logs else 'Never'
        ])
    output = make_response(si.getvalue())
    output.headers['Content-Disposition'] = 'attachment; filename=user_login_summary.csv'
    output.headers['Content-Type'] = 'text/csv'
    return output

# ─────────────────────────────────────────
# FEEDBACK
# ─────────────────────────────────────────
@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    fb   = Feedback(name=data.get('name',''),email=data.get('email',''),message=data.get('message',''))
    db.session.add(fb); db.session.commit()
    return jsonify({'message':'Feedback received!'}), 201

# ─────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin-panel')
def admin_panel():
    """Dedicated admin panel page — separate from main SPA."""
    return render_template('admin_panel.html')

@app.route('/<path:path>')
def catch_all(path=''):
    return render_template('index.html')

# ─────────────────────────────────────────
# INIT DB + RUN
# ─────────────────────────────────────────
def initialize_database():
    try:
        with app.app_context():
            db.create_all()
            if not User.query.filter_by(role='admin').first():
                hashed = bcrypt.generate_password_hash('Admin@123').decode('utf-8')
                db.session.add(User(username='admin',email='admin@sleepsense.ai',password=hashed,role='admin'))
                db.session.commit()
                print('Default admin created: admin@sleepsense.ai / Admin@123')
        print('Database ready.')
    except Exception as exc:
        if 'mysql' in app.config['SQLALCHEMY_DATABASE_URI']:
            print('MySQL connection failed — switching to SQLite fallback.')
            app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{SQLITE_DB_PATH}"
            with app.app_context():
                db.create_all()
        else:
            raise exc

if __name__ == '__main__':
    initialize_database()
    app.run(debug=True, host='0.0.0.0', port=5000)
