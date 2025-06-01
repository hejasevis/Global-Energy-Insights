⚡ Global Energy Insights
An interactive dashboard that visualizes global energy consumption trends, growth rates, source distributions, and forecasts using machine learning models.
🌐 Live at Streamlit: global-energy-insights.streamlit.app
---

⚡ **Features**

- Visualize per capita and total energy consumption globally and by country.  
- Discover hidden relationships between energy types using **association rule mining**.  
- Analyze yearly **growth rates** in energy source consumption.  
- Compare **renewable vs non-renewable** energy mix across countries.  
- Forecast future consumption using **Prophet** and **Random Forest** models.  
- Backtest model predictions with real historical data.

---

🚀 **How It Works**

1. 📊 Load data from OWID’s global energy dataset.  
2. 🧠 Normalize, analyze, and visualize using ML and statistical tools.  
3. 🌍 Interact through a sleek **Streamlit interface** with dynamic charts and filters.  
4. 🔮 Generate predictions and evaluate model performance.

---

🛠️ **Technologies Used**

- Python (Pandas, NumPy, Scikit-learn, Prophet)  
- Plotly & Seaborn for data visualization  
- Streamlit for dashboarding  
- Apriori Algorithm (mlxtend)  
- Dataset: [Our World in Data - Energy](https://ourworldindata.org/energy)

---

📈 **Goals**

- Make global energy data more **accessible**, **interactive**, and **actionable**.  
- Showcase the power of **AI & data science** in energy policy and sustainability.  
- Help stakeholders **visualize trends**, **identify patterns**, and **make informed decisions**.

---

🚧 **Limitations**

- Forecasts are based on past consumption trends; external factors (e.g. policy shifts, conflicts) are not accounted for.  
- Dataset relies on reported country-level data, which may vary in accuracy or completeness.

---

🌟 **Getting Started**

```bash
git clone https://github.com/hejasevis/energy-dashboard.git
cd energy-dashboard
pip install -r requirements.txt
streamlit run app.py
```
🖇️ Contribution

Contributions, issues, and feature requests are welcome!
Check out the issues page or open a pull request 🤝

This dashboard was built to bridge the gap between raw energy data and meaningful, insightful decisions.
Let’s build a more sustainable future — one data point at a time. 🌍
