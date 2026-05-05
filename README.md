# Automated Dataset Diagnostics and Repair Framework (ADRF)
### Final Year Project — Data-Centric Machine Learning

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Open in Browser
```
http://localhost:5000
```

---

## 📂 Project Structure

```
dataset_diagnostics/
├── app.py                    # Flask web application (main entry point)
├── requirements.txt          # Python dependencies
├── generate_sample_data.py   # Script to create test CSV
├── modules/
│   ├── diagnostics.py        # Dataset quality analysis engine
│   ├── repair.py             # Data repair and intervention module
│   ├── model_evaluator.py    # ML model training and comparison
│   └── visualizer.py         # Chart and visualization generator
├── templates/
│   └── index.html            # Main GUI (single-page application)
├── uploads/                  # Uploaded CSV files (auto-created)
├── versions/                 # Repaired dataset versions (auto-created)
└── reports/                  # Generated reports (auto-created)
```

---

## 🔬 Features

### Pipeline Steps
1. **Upload** — Drag & drop CSV files, instant preview
2. **Diagnostics** — 9 automated quality checks with severity scoring
3. **Repair** — 7 targeted repair operations with versioning
4. **Evaluate** — Before/after ML model performance comparison
5. **Report** — Comprehensive health report with download

### Diagnostics Checks
| Check | Description |
|-------|-------------|
| Missing Values | Per-column detection with severity levels |
| Class Imbalance | Distribution analysis and ratio calculation |
| Outliers | IQR-based detection across all numeric features |
| Duplicates | Exact duplicate row identification |
| Feature Correlation | Highly correlated feature pair detection |
| Data Leakage | Target-feature correlation analysis |
| Distribution Skew | Statistical skewness measurement |
| Constant Features | Zero-variance feature detection |
| Label Noise | KNN-based inconsistency estimation |

### Repair Operations
| Operation | Trigger Condition |
|-----------|------------------|
| Constant Feature Removal | Any constant column |
| Median/Mode Imputation | Missing values detected |
| Duplicate Removal | Duplicate rows exist |
| Outlier Capping (Winsorization) | Medium/High severity outliers |
| Log Transform | Highly skewed distributions |
| Correlated Feature Pruning | Correlation > 0.85 |
| Minority Class Oversampling | Imbalance ratio > 3x |

### ML Models Evaluated
- Random Forest Classifier/Regressor
- Logistic Regression / Linear Regression  
- Gradient Boosting Classifier

---

## 📊 Supported Dataset Formats

- **Format**: CSV only
- **Max Size**: 50MB
- **Target Column**: Auto-detected (last column, or named: target/label/class/y)
- **Task Types**: Classification and Regression

---

## 🛠️ Testing with Sample Data

```bash
python generate_sample_data.py
# Generates: sample_dataset.csv with intentional quality issues
```

---

## 🔧 Configuration

Edit `app.py` to change:
- `MAX_CONTENT_LENGTH` — File size limit (default 50MB)
- Port number — Default 5000

---

## 📋 Requirements

```
flask>=3.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
werkzeug>=3.0
```

---

## 👨‍💻 Academic Context

**Project Title**: Automated Dataset Diagnostics and Repair Framework for Data-Centric Machine Learning  
**Domain**: Data-Centric AI, Machine Learning, Software Engineering  
**Methodology**: Modular pipeline design with statistical, semantic, and model-based analysis

---

## 📄 License

Academic use only. Final Year Project submission.
