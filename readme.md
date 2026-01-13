# UIDAI Aadhaar Analytics Dashboard 🚀



A **policy-grade, data-driven analytics dashboard** built for the UIDAI Hackathon. This project transforms Aadhaar enrollment, demographic update, and biometric update data into **actionable insights**, **early warning signals**, and **prescriptive infrastructure planning** for policymakers.



---



## 📌 Project Highlights



* 📊 **Multi-dimensional Analytics**: Enrollment, Demographic & Biometric trends

* 🚨 **ASSI – Aadhaar Service Stress Index** with district-level risk ranking

* 🔮 **Forecasting Engine** (Enrollment / Demo / Bio)

* 🧠 **Policy Engine** for early warnings & stress explanations

* 🗺️ **Interactive India Choropleth Map** (State-wise Aadhaar activity)

* 🏗️ **Strategic Resource Planning** (kits, manpower, cost estimation)

* 🤖 **AI Smart Assistant (Gemini)** for conversational policy insights



---



## 🧩 Tech Stack



* **Frontend / App**: Streamlit

* **Data Processing**: Pandas, NumPy

* **Visualization**: Plotly

* **ML / Analytics**: Scikit-learn (MinMaxScaler)

* **AI Assistant**: Google Gemini API

* **Geo Mapping**: GeoJSON + Plotly Choropleth



---



## 📁 Project Structure



```

uidai_final_project/

│

├── app.py                  # Main Streamlit dashboard

├── engine.py               # Policy & Stress Analytics Engine

├── data/                   # CSV input datasets

├── india_states.geojson    # State boundaries for India map

├── requirements.txt        # Python dependencies

└── README.md               # Project documentation

```



---



## ▶️ How to Run the Project



### 1️⃣ Install Dependencies



```bash

pip install -r requirements.txt

```



### 2️⃣ Run the Streamlit App



```bash

streamlit run app.py

```



The dashboard will open in your browser at:



```

http://localhost:8501

```



---



## 📊 app.py – Dashboard Overview



`app.py` is the **presentation and analytics layer** of the project.



### 🔹 Core Responsibilities



* Load and preprocess Aadhaar CSV datasets

* Apply global filters (State, District, Date Range)

* Compute KPIs and derived metrics

* Render visualizations and interactive tabs

* Invoke the Policy Engine (`engine.py`)

* Host the AI-powered Smart Assistant



---



## 🗂️ Data Loading



```python

def load_csv_folder(folder_path):

    ...

```



* Automatically reads and concatenates all CSVs from the `data/` folder

* Ensures scalability for multi-file UIDAI datasets



---



## 🧭 Dashboard Tabs Explained



### 1️⃣ Enrollment Trends



* Daily & Monthly enrollment analysis

* Anomaly detection using Z-score

* Short-term (7-day) and monthly forecasts



### 2️⃣ Demographic Updates



* State-wise heatmaps

* Anomaly detection & forecasting

* Identifies migration & update surges



### 3️⃣ Biometric Updates



* District-level biometric activity

* Child biometric stress detection (5–17 age group)

* Forecasting & anomaly flags



### 4️⃣ Strategic Planning



* Converts forecasts into **real-world infrastructure needs**

* Calculates:



  * Enrollment kits required

  * Personnel requirements

  * Utilization rates

  * Budget & cost efficiency

* Supports **National / State / District** planning



### 5️⃣ 🚨 Policy Alerts & Stress



* Displays **High-Risk Districts**

* Aadhaar Service Stress Index (ASSI)

* Auto-generated explanations for policymakers



### 6️⃣ 🤖 UIDAI Smart Assistant



* Gemini-powered conversational AI

* Answers policy questions using live dashboard context

* Example:



  > "Why is Bihar showing high stress this month?"



### 7️⃣ 🗺️ India Map View



* State-wise Aadhaar enrollment choropleth

* GeoJSON-based accurate mapping

* Darker states = higher Aadhaar activity



---



## ⚙️ engine.py – Policy & Stress Analytics Engine



`engine.py` is the **analytical brain** of the system.



### 🔹 What It Does



* Aggregates Aadhaar data at **State–District–Month** level

* Computes derived operational metrics

* Builds a **composite stress index (ASSI)**

* Flags early warning districts

* Explains *why* a district is under stress



---



## 🧮 Aadhaar Service Stress Index (ASSI)



```text

ASSI = 0.4 × Enrollment Load

     + 0.3 × Demographic Updates

     + 0.3 × Biometric Updates

```



All components are **Min–Max normalized** to ensure fairness across regions.



---



## 🚨 Early Warning Logic



A district is flagged if it shows:



* 🔁 **High Update Burden** (repeat updates)

* 🔄 **Biometric–Demographic Shift**

* 📉 **Volatile Demand Patterns**



Each warning includes an **auto-generated explanation** for decision-makers.



---



## 🧒 Child Biometric Hotspots



Special focus on **ages 5–17**:



* Identifies districts with unusually high biometric updates

* Helps UIDAI address child biometric failures early



---



## 🗺️ Geo Mapping



* Uses `india_states.geojson`

* Accurate state boundary mapping

* Integrated with Plotly for interactivity



---



## 🔐 Security Note



⚠️ **Important**: For production or public GitHub use:



* Move Gemini API key to environment variables

* Do **NOT** hardcode API keys



---



## 🏆 Hackathon Value Proposition



This project goes beyond visualization:



✔ Predicts demand

✔ Detects stress early

✔ Explains root causes

✔ Prescribes infrastructure & budget

✔ Enables AI-assisted policy decisions



---



## 👤 Author



**Prince Tiwari**

UIDAI Hackathon Project

*Data Analytics | Policy Intelligence | AI for Governance*



---



## 📜 License



This project is intended for **educational & hackathon use**.