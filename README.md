# widsdatathon2025

## Project Agenda
| Part  | Date   | Agenda                          |
|-------|--------|---------------------------------|
| 1     | 4/08   | Understanding Dataset/Project   |
| 2     | 4/15   | Data Analysis & Processing      |
| 3     | 4/22   | Model Training                  |
| 4     | 4/29   | Finetuning                      |

## 📁 Project Structure & Workflow Guidelines

To collaborate effectively during the WiDS Datathon, we recommend the following development workflow using both Jupyter Notebooks and Python scripts.

### 🧪 Jupyter Notebooks
Use notebooks for:
- Exploratory Data Analysis (EDA)
- Visualizing features and distributions
- Experimenting with models and evaluation metrics
- Documenting findings with markdown cells

📄 Suggested Notebooks:
- `notebooks/eda.ipynb`
- `notebooks/model_experiments.ipynb`

---

### 🛠 Python Scripts
Use Python scripts for:
- Preprocessing pipelines
- Model training and evaluation
- Utility functions and helpers
- Submission generation

📄 Suggested Scripts:
- `src/preprocessing.py`
- `src/train_model.py`
- `src/utils.py`
- `src/create_submission.py`

---

### 📁 Recommended Directory Structure
```plaintext
/wids-datathon/
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for EDA and prototyping
├── src/                   # Source code for preprocessing, training, etc.
├── submissions/           # Output CSVs for leaderboard submission
├── README.md              # Project overview and instructions
└── requirements.txt       # List of Python dependencies
```

### 🔧 Git & Collaboration Tips
- Use `.gitignore` to avoid tracking large data files and notebook checkpoints.
