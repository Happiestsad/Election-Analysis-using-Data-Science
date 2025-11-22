<!-- Hero Section -->
<div align="center">

  <h1 style="font-size: 2.6rem; margin-bottom: 0.4rem;">🗳️ Election Analysis using Data Science</h1>
  <p style="font-size: 1.1rem; opacity: 0.9; max-width: 650px;">
    A machine learning-powered platform to predict election results and visualize key electoral patterns with Streamlit.
  </p>

  <!-- Tech Badges -->
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
    <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </p>

  <!-- Repo Stats -->
  <p>
    <img src="https://img.shields.io/github/stars/Happiestsad/Election-Analysis-using-Data-Science?style=flat-square" />
    <img src="https://img.shields.io/github/forks/Happiestsad/Election-Analysis-using-Data-Science?style=flat-square" />
    <img src="https://img.shields.io/github/issues/Happiestsad/Election-Analysis-using-Data-Science?style=flat-square" />
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" />
  </p>

</div>

---



## ✨ Overview
A data science project that uses machine learning and Streamlit to:
- Predict **election winners**
- Analyze **patterns & trends**
- Visualize model performance and feature importance

<img src="./assets/dividers/wave-red.svg" width="100%" />


## 🧩 Features

<div align="center">

| 🤖 ML Prediction | 📊 Streamlit Dashboard | 📈 Analytical Insights |
|:----------------:|:----------------------:|:-----------------------:|
| Predict outcomes using trained models | User-friendly web interface | Confusion matrix, feature importance, win rates |

</div>

---

## 📁 Project Structure

```bash
Election-Analysis-using-Data-Science/
│── app.py                 # Streamlit main file
│── requirements.txt       # Dependencies
│── LS_2.0.xls             # Dataset
│── model_rf_new.pkl       # Trained ML model
│── scaler_new.pkl         # Data scaler
│── win_rate_stats.pkl     # Win-rate stats
│── confusion_matrices.png # Model accuracy plot
│── feature_importance.png # Feature importance chart
└── README.md
```
---

## 📦 Dependencies
You’ll need:
<ul>

- Python 3.8+

- pip (Python package manager)

- All main libraries are listed in requirements.txt, including:

- ` streamlit`

- `pandas`

- `numpy`

- `scikit-learn`

- `matplotlib` / `seaborn` (for plots)

any other libs referenced in the app

</ul>

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Happiestsad/Election-Analysis-using-Data-Science.git

cd Election-Analysis-using-Data-Science
```
### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate

```
### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 🚀 Run the App
Make sure you’re in the project folder and (optionally) your virtual environment is activated.

```
streamlit run app.py
```
Streamlit will start a local server, typically at:

``` 
http://localhost:8501
 ```

<div align="center">
🧾 License & Credits
</div>

This project is licensed under the Apache-2.0 License.
See the [LICENSE](./LICENSE)
 file for more details.

Built with ❤️ by Happiestsad using Python, Machine Learning, and Streamlit.

<div align="center">
⭐ Like this project?

If this repo helped you or you find it interesting,
<strong>consider giving it a star ⭐ on GitHub</strong> — it really motivates further improvements!

</div> 
