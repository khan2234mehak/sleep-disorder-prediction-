-- ═══════════════════════════════════════════════
-- Sleep Disorder Prediction System — Database Schema (UPGRADED)
-- Includes: users, predictions, login_logs, feedback, views
-- Run:  mysql -u root -p < database/schema.sql
-- ═══════════════════════════════════════════════

CREATE DATABASE IF NOT EXISTS sleep_disorder_db
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE sleep_disorder_db;

-- 1. USERS
CREATE TABLE IF NOT EXISTS users (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    username    VARCHAR(80)  NOT NULL UNIQUE,
    email       VARCHAR(120) NOT NULL UNIQUE,
    password    VARCHAR(255) NOT NULL,
    role        VARCHAR(20)  NOT NULL DEFAULT 'patient',
    is_active   TINYINT(1)   NOT NULL DEFAULT 1,
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_email(email), INDEX idx_role(role)
) ENGINE=InnoDB;

-- 2. PREDICTIONS
CREATE TABLE IF NOT EXISTS predictions (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    user_id           INT NOT NULL,
    age               INT, gender VARCHAR(10), occupation VARCHAR(60),
    sleep_duration    FLOAT, stress_level INT, bmi_category VARCHAR(20),
    heart_rate        INT, daily_steps INT, physical_activity INT,
    systolic_bp       INT, diastolic_bp INT,
    prediction        VARCHAR(30), confidence FLOAT, sleep_score INT,
    created_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id(user_id), INDEX idx_created(created_at)
) ENGINE=InnoDB;

-- 3. LOGIN LOGS
CREATE TABLE IF NOT EXISTS login_logs (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    user_id     INT         NOT NULL,
    ip_address  VARCHAR(45),
    user_agent  VARCHAR(255),
    status      VARCHAR(10) NOT NULL DEFAULT 'success',
    created_at  DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id(user_id), INDEX idx_status(status), INDEX idx_created(created_at)
) ENGINE=InnoDB;

-- 4. FEEDBACK
CREATE TABLE IF NOT EXISTS feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(80), email VARCHAR(120), message TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 5. VIEWS
CREATE OR REPLACE VIEW user_summary AS
SELECT u.id, u.username, u.email, u.role, u.is_active, u.created_at,
    COUNT(p.id) AS total_predictions,
    AVG(p.sleep_score) AS avg_sleep_score,
    MAX(p.created_at) AS last_prediction,
    (SELECT ll.created_at FROM login_logs ll WHERE ll.user_id=u.id AND ll.status='success' ORDER BY ll.created_at DESC LIMIT 1) AS last_login,
    (SELECT ll.ip_address FROM login_logs ll WHERE ll.user_id=u.id AND ll.status='success' ORDER BY ll.created_at DESC LIMIT 1) AS last_login_ip
FROM users u LEFT JOIN predictions p ON u.id=p.user_id
GROUP BY u.id, u.username, u.email, u.role, u.is_active, u.created_at;

CREATE OR REPLACE VIEW disorder_stats AS
SELECT prediction, COUNT(*) AS total,
    ROUND(AVG(confidence),2) AS avg_confidence,
    ROUND(AVG(sleep_score),1) AS avg_sleep_score,
    ROUND(AVG(sleep_duration),2) AS avg_sleep_duration
FROM predictions GROUP BY prediction;

CREATE OR REPLACE VIEW login_summary AS
SELECT u.username, u.email, u.role,
    COUNT(ll.id) AS total_logins,
    SUM(ll.status='success') AS successful,
    SUM(ll.status='failed') AS failed,
    MAX(IF(ll.status='success',ll.created_at,NULL)) AS last_success
FROM users u LEFT JOIN login_logs ll ON u.id=ll.user_id
GROUP BY u.id, u.username, u.email, u.role;

-- 6. DEFAULT ADMIN  (password: Admin@123)
INSERT IGNORE INTO users (username,email,password,role) VALUES
('admin','admin@sleepsense.ai','$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW','admin');

-- 7. DEMO USERS  (password: demo123)
INSERT IGNORE INTO users (username,email,password,role) VALUES
('rahul.sharma','patient@demo.com','$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW','patient'),
('dr.ananya',   'doctor@demo.com', '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW','doctor');

-- 8. SAMPLE PREDICTIONS
INSERT IGNORE INTO predictions
  (user_id,age,gender,occupation,sleep_duration,stress_level,bmi_category,heart_rate,daily_steps,physical_activity,systolic_bp,diastolic_bp,prediction,confidence,sleep_score)
VALUES
  (2,32,'Male','Engineer',5.5,7,'Normal',76,4200,25,122,80,'Insomnia',88.4,48),
  (2,32,'Male','Engineer',6.0,6,'Normal',74,5100,30,120,78,'Insomnia',82.1,52),
  (2,32,'Male','Engineer',7.2,4,'Normal',68,8200,45,118,76,'None',91.6,78);

-- 9. SAMPLE LOGIN LOGS
INSERT IGNORE INTO login_logs (user_id,ip_address,user_agent,status) VALUES
(1,'127.0.0.1','Mozilla/5.0 Admin Chrome/124','success'),
(2,'192.168.1.42','Mozilla/5.0 Chrome/124','success'),
(3,'10.0.0.5','Mozilla/5.0 Safari/17','success');
