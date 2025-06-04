# 🚀 Core Libraries
import streamlit as st
import pandas as pd
import numpy as np

# 📊 Visualization
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns

# 🔎 Data Processing & Machine Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 📌 Association Rule Mining
from mlxtend.frequent_patterns import apriori, association_rules

# ⏳ Time Series Forecasting
from prophet import Prophet
from prophet.plot import plot_plotly

# 🖼 Image Processing (optional)
from PIL import Image

# 🌐 Streamlit Option Menu
from streamlit_option_menu import option_menu

# ⚙️ Page Configuration
st.set_page_config(layout="wide")

# 🌞 Apply Light Theme Styling Fix
st.markdown(
    """
    <style>
        body {
            background-color: white !important;
            color: black !important;
        }
        .main {
            background-color: white !important;
        }
        .css-1d391kg, .css-hxt7ib, .css-1v0mbdj, .css-18e3th9 {
            background-color: #f9f9f9 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 📋 Sidebar Navigation Menu
with st.sidebar:
    page = option_menu(
        menu_title="Dashboard Menu",
        options=[
            "Home",
            "Global Energy Map",
            "Energy Relationships",
            "Growth Rate Trends",
            "Country Energy Mix",
            "Future Energy Forecast"
        ],
        icons=["house", "globe", "bar-chart-line", "graph-up-arrow", "pie-chart", "stars"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f9f9f9"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#e0e0e0",
            },
            "nav-link-selected": {
                "background-color": "#5e60ce",
                "color": "white",
                "font-weight": "bold",
                "border-radius": "8px",
            },
            "icon": {"color": "#000000"},
        }
    )

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("owid-energy-data.csv")

df = load_data()

# 🏠 Home
if page == "Home":

    st.image("images/background.png", use_container_width=True)

    st.title("🔌 Global Energy Dashboard")
    st.markdown("""
        Welcome to the **Global Energy Dashboard**, an interactive platform designed to visualize, analyze, and forecast worldwide energy consumption patterns.  
        This project leverages data from [Our World in Data](https://ourworldindata.org/energy).
    """)

    st.markdown("### 🔍 Dashboard Overview")
    st.markdown("""
    - 🗺️ **Global Map**  
      View energy consumption per capita by country over time.

    - 🌐 **Deep Analysis**  
      Uncover meaningful associations between different energy types using Apriori-based rule mining.

    - 📈 **Growth Rates**  
      Analyze year-over-year changes and long-term trends in energy use.

    - 🗺 **Country vs Energy Type**  
      Compare countries based on specific energy type consumption and production.

    - 🔮 **Forecasting Module**  
      Predict future energy demand using machine learning models (Prophet & Random Forest).
    """)

    st.markdown("### 🧭 How to Navigate")
    st.markdown("""
        Use the sidebar on the left to select different dashboard sections.  
        Each section presents a specific aspect of the dataset, allowing you to explore patterns and predictions.
    """)

    st.markdown("---")
    st.info("📌 *This dashboard is developed using Python, Streamlit, and machine learning tools as part of a final year project.*")


# 🗺️ Page 1 -  Global Energy Map
elif page == "Global Energy Map":

    # 📌 Page Title and Introductory Info
    st.title("🗺️ Global Map of Energy Use per Capita")

    # ℹ️ Helper info box to guide users
    st.info("""
    This map illustrates per capita energy consumption (in kilowatt-hours per person) 
    across countries for a selected year.  
    Use the **slider** to pick a year and the **dropdown** to explore a specific country's values.
    """)

    # 📌 Year selection section for filtering the map data
    st.markdown("### 📅 Year Selection")

    # Select relevant columns and remove rows with missing values
    df_map = df[["iso_code", "country", "year", "energy_per_capita", "population"]].dropna()

    # Slider to pick a specific year from the available dataset
    year = st.slider("Select Year", int(df_map["year"].min()), int(df_map["year"].max()), 2023)

    # Filter the data based on selected year
    df_year = df_map[df_map["year"] == year]

    # 📌 Country selection for displaying specific data
    country_list = sorted(df_year["country"].unique())
    selected_country = st.selectbox("🌎 Select a Country to View Details", country_list)

    # Get data for the selected country
    selected_row = df_year[df_year["country"] == selected_country].iloc[0]

    # 📌 Choropleth Map
    # This visualizes per capita energy consumption using a colored world map
    fig = px.choropleth(
        df_year,
        locations="iso_code",
        color="energy_per_capita",
        hover_name="country",
        color_continuous_scale=["#76c893", "#34a0a4", "#1a759f", "#1e6091", "#184e77"],
        labels={"energy_per_capita": "kWh / person"},
        title=f"Per Capita Energy Consumption ({year})"
    )

    # Map style configuration - hides borders and sets projection
    fig.update_geos(
        showframe=False,
        showcoastlines=False,
        projection_type="natural earth"
    )

    # 📌 Layout and Theme
    # Custom styling for dark mode compatibility
    fig.update_layout(
    template="plotly_white",  # Light mode styling
    margin=dict(l=0, r=0, t=60, b=0),
    height=600,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="black", size=12),
    geo_bgcolor="white"
    )


    # 📌 Render the map
    st.plotly_chart(fig, use_container_width=True)

    # 📌 Additional analysis and country-level metrics

    # Calculate global average for comparison
    global_avg = df_year["energy_per_capita"].mean()
    diff = selected_row["energy_per_capita"] - global_avg
    comparison = "above" if diff > 0 else "below"

    # Calculate total energy consumption
    population = selected_row["population"]
    total_energy = selected_row["energy_per_capita"] * population

    # Display enhanced info box
    st.info(f"""
    #### 📄 Details for {selected_country} ({year})
    - 📊 Energy per Capita: **{selected_row['energy_per_capita']:.2f} kWh/person**
    - 👥 Population: **{population:,.0f}**
    - 🔋 Estimated Total Energy Consumption: **{total_energy:,.0f} kWh**
    - 🌍 Global Average (same year): **{global_avg:.2f} kWh/person**
    - 🔎 This is **{abs(diff):.2f} kWh/person {comparison}** the global average.
    - 📆 Year: **{selected_row['year']}**
    """)

  
# 🌐 Page 2 - Energy Relationships 
elif page == "Energy Relationships":

    # 📌 Page Title and Overview
    st.title("🌐 Country-Level Energy Pattern Discovery")

    # ℹ️ Explain the purpose of the page
    st.info("""
    This section applies **association rule mining** and **correlation analysis** to uncover hidden relationships 
    between energy consumption types in selected countries and years.  
    You can adjust thresholds, year range, and countries to explore different patterns.
    """)

    # 📌 User Inputs: countries, thresholds, year range
    selected_countries = st.multiselect(
        "Select Countries",
        sorted(df["country"].dropna().unique()),
        default=["Turkey", "Germany", "United States", "France"]
    )

    threshold = st.slider("Binary Threshold (0–1 scale)", 0.1, 0.9, 0.3)
    min_support = st.slider("Minimum Support", 0.1, 1.0, 0.4)
    min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.0)
    year_range = st.slider("Select Year Range", 1965, 2023, (2000, 2022))

    # 📌 Trigger analysis with button
    if st.button("Run Analysis"):

        # 📌 Filter the dataset based on user selection
        filtered_df = df[
            (df["country"].isin(selected_countries)) &
            (df["year"].between(year_range[0], year_range[1]))
        ].copy()

        # 📌 Select only consumption-related columns, drop others
        energy_columns = [col for col in filtered_df.columns if 'consumption' in col and 'change' not in col]
        filtered_df = filtered_df[["country", "year"] + energy_columns].dropna()

        # 📌 Normalize data between 0–1
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(filtered_df[energy_columns])
        norm_df = pd.DataFrame(normalized, columns=energy_columns)

        # 📌 Binarize based on selected threshold
        binary_df = (norm_df > threshold).astype(int)

        # 📌 Apply Apriori algorithm to discover frequent itemsets
        frequent_itemsets = apriori(binary_df, min_support=min_support, use_colnames=True)

        # 📌 Generate association rules
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        rules_sorted = rules.sort_values(by=["lift", "confidence", "support"], ascending=False)

        # 📋 1. Association Rules Table
        st.subheader("📋 Association Rules")
        st.markdown(f"📅 Showing rules for **{year_range[0]}–{year_range[1]}**")
        st.dataframe(rules_sorted)

        # ⚡️ 2. Correlation Heatmap
        st.subheader("⚡️ Correlation Heatmap")
        st.markdown("This heatmap shows normalized Pearson correlations between different energy consumption types.")

        import plotly.figure_factory as ff
        corr = norm_df.corr()
        z = corr.values
        x = list(corr.columns)
        y = list(corr.index)

        fig_heatmap = ff.create_annotated_heatmap(
            z=z,
            x=x,
            y=y,
            annotation_text=[[f"{val:.2f}" for val in row] for row in z],
            colorscale="YlGnBu",
            showscale=True
        )

        fig_heatmap.update_layout(
            title=dict(
                text="Correlation Between Energy Types",
                y=1.0,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=16)
            ),
            font=dict(size=12),
            height=600,
            margin=dict(l=60, r=60, t=80, b=60),
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickangle=45, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10))
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        # 📊 3. Top 10 Rules by Support
        st.subheader("📊 Top 10 Rules by Support")
        st.markdown("Shows the rules with the highest support values, indicating common energy patterns.")

        if not rules_sorted.empty:
            top_support = rules_sorted.nlargest(10, 'support')
            bar_data = top_support[['antecedents', 'consequents', 'support']].copy()

            def format_set(s):
                return ", ".join(sorted(list(s)))

            bar_data['rule'] = bar_data.apply(
                lambda row: f"{format_set(row['antecedents'])} → {format_set(row['consequents'])}",
                axis=1
            )

            fig2 = px.bar(
                bar_data,
                x='rule',
                y='support',
                title=f"Top Rules by Support ({year_range[0]}–{year_range[1]})",
                text='support',
                color='support',
                color_continuous_scale='Blues'
            )

            fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig2.update_layout(
                xaxis_tickangle=30,
                xaxis_title="Rule",
                yaxis_title="Support",
                title_font_size=20,
                font=dict(size=12),
                height=600,
                margin=dict(l=60, r=60, t=60, b=200),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No rules to visualize. Try adjusting thresholds or year range.")

        # 🔍 4. Insights
        st.markdown("### 🔍 Insights")
        if not rules_sorted.empty:
            avg_support = rules_sorted['support'].mean()
            avg_lift = rules_sorted['lift'].mean()
            avg_confidence = rules_sorted['confidence'].mean()

            st.markdown(f"""
            - **Average Support:** {avg_support:.2f}  
            - **Average Lift:** {avg_lift:.2f}  
            - **Average Confidence:** {avg_confidence:.2f}
            """)
        else:
            st.info("No insights available due to lack of valid rules.")

          
# 📈 Page 3 - Growth Rate Trends
elif page == "Growth Rate Trends":

    # 📌 Page Title
    st.title("📈 Annual Growth Trends in Energy Consumption")

    # ℹ️ Info box to explain this section
    st.info("""
    This section visualizes **annual growth rates** in energy consumption for various sources.  
    You can select a specific country or view global trends, and filter by year range and energy types.
    """)

    # 📊 Select energy consumption columns (e.g. coal_consumption, solar_consumption)
    energy_cols = [col for col in df.columns if col.endswith("_consumption")]

    # 📌 Drop rows with missing values in relevant columns
    df_clean = df[["country", "year"] + energy_cols].dropna()

    # 🌍 Country selection
    countries = sorted(df_clean["country"].unique())
    countries.insert(0, "World")  # Add 'World' to allow global analysis
    selected_country = st.selectbox("Select Country (or World):", countries)

    # 📆 Year range selection
    country_df = df_clean[df_clean["country"] == selected_country]
    min_year = int(country_df["year"].min())
    max_year = int(country_df["year"].max())
    year_range = st.slider("Select Year Range:", min_year, max_year, (2010, 2022))

    # 📌 Filter data for selected year range
    filtered_df = country_df[
        (country_df["year"] >= year_range[0]) &
        (country_df["year"] <= year_range[1])
    ].copy()

    # 🔢 Calculate annual % change for each energy type
    for col in energy_cols:
        filtered_df[col + "_change_%"] = filtered_df[col].pct_change() * 100

    # ⚡ Energy source selection
    selected_sources = st.multiselect(
        "Select Energy Sources:",
        energy_cols,
        default=energy_cols[:3]
    )

    # 📈 Plotly line chart
    st.markdown("### 📊 Annual Growth Rates by Source")

    fig = go.Figure()

    for col in selected_sources:
        fig.add_trace(go.Scatter(
            x=filtered_df["year"],
            y=filtered_df[col + "_change_%"],
            mode='lines+markers',
            name=col.replace("_consumption", "").title()
        ))

    fig.update_layout(
        title=f"{selected_country} – Annual Energy Consumption Growth Rates",
        xaxis_title="Year",
        yaxis_title="Change Rate (%)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 📊 Insight Section
    st.markdown("### 💡 Insights")

    # Calculate average growth rate for each selected source
    insight_df = filtered_df[[col + "_change_%" for col in selected_sources]].mean().sort_values(ascending=False)
    insight_df = insight_df.reset_index()
    insight_df.columns = ["Energy Source", "Average Annual Growth (%)"]
    insight_df["Energy Source"] = insight_df["Energy Source"].str.replace("_consumption_change_%", "").str.title()

    # Top and bottom performing sources
    top = insight_df.iloc[0]
    bottom = insight_df.iloc[-1]

    st.markdown(f"""
    - **Highest average growth:** `{top['Energy Source']}` with **{top['Average Annual Growth (%)']:.2f}%**
    - **Lowest (or negative) average growth:** `{bottom['Energy Source']}` with **{bottom['Average Annual Growth (%)']:.2f}%**
    """)

    # Optional: Show full table in expander
    with st.expander("🔍 See All Source Growth Averages"):
        st.dataframe(insight_df)

    
# 🗺 Page 4 - Country Energy Mix
elif page == "Country Energy Mix":

    # 📌 Title and Info Box
    st.title("🗺 Energy Mix Analysis by Country and Year")
    st.info("""
    This section compares how different energy sources contribute to total energy consumption for a selected country.  
    You can analyze energy share based on a specific year range and focus on selected types such as fossil, nuclear, or renewables.
    """)

    # 📌 Prepare column groups
    energy_cols = [col for col in df.columns if col.endswith("_consumption")]
    renewable_cols = [
        "solar_consumption", "wind_consumption", "biofuel_consumption",
        "hydro_consumption", "other_renewable_consumption"
    ]
    non_renewable_cols = [
        "coal_consumption", "oil_consumption", "gas_consumption", "nuclear_consumption"
    ]

    # 📌 Drop rows with missing energy values
    df_energy = df[["country", "year"] + energy_cols].dropna()

    # 🌍 Country selection
    country_list = sorted(df_energy["country"].unique())
    selected_country = st.selectbox("Select a Country:", country_list)

    # 📆 Year range selection
    min_year = int(df_energy["year"].min())
    max_year = int(df_energy["year"].max())
    year_range = st.slider("Select Year Range:", min_year, max_year, (2020, 2022))

    # 📌 Filter dataset by country and year
    country_data = df_energy[
        (df_energy["country"] == selected_country) &
        (df_energy["year"] >= year_range[0]) &
        (df_energy["year"] <= year_range[1])
    ]

    # ⚡ Energy source selection
    selected_energy = st.multiselect(
        "Select Energy Sources to Compare:",
        energy_cols,
        default=energy_cols[:5]
    )

    # 📊 Average consumption for selected energy types
    avg_data = country_data[selected_energy].mean().sort_values(ascending=False)
    avg_df = avg_data.reset_index()
    avg_df.columns = ["Energy Source", "Average Consumption"]

    # 🥧 Pie Chart
    st.markdown("### 🥧 Energy Type Share (Pie Chart)")
    fig_pie = px.pie(
        avg_df,
        names="Energy Source",
        values="Average Consumption",
        title=f"{selected_country} – Energy Type Share ({year_range[0]}–{year_range[1]})",
        hole=0.3
    )
    fig_pie.update_layout(template="plotly_white")
    st.plotly_chart(fig_pie, use_container_width=True)

    # ⚡ Insight Section
    st.markdown("### ⚡ Insights")

    total = avg_df["Average Consumption"].sum()
    avg_df["Percentage"] = (avg_df["Average Consumption"] / total * 100).round(2)

    top_row = avg_df.iloc[0]
    bottom_row = avg_df.iloc[-1]

    st.markdown(f"""
    - **Most used energy source:** `{top_row['Energy Source'].replace('_consumption', '').title()}` with **{top_row['Percentage']}%**
    - **Least used energy source:** `{bottom_row['Energy Source'].replace('_consumption', '').title()}` with **{bottom_row['Percentage']}%**
    - Total consumption (for selected sources and years): **{total:,.0f} kWh**
    """)

    # 🔍 Expandable full breakdown
    with st.expander("🔍 See Full Share Breakdown"):
        for _, row in avg_df.iterrows():
            st.markdown(f"- `{row['Energy Source'].replace('_consumption', '').title()}`: **{row['Percentage']}%**")

    # Additional Indicator: Renewable Ratio
    st.markdown("### Renewable Energy Share")

    # Calculate renewable and non-renewable sums
    renew_sum = country_data[renewable_cols].sum().sum() if set(renewable_cols).issubset(country_data.columns) else 0
    non_renew_sum = country_data[non_renewable_cols].sum().sum() if set(non_renewable_cols).issubset(country_data.columns) else 0

    if renew_sum + non_renew_sum > 0:
        renewable_ratio = (renew_sum / (renew_sum + non_renew_sum)) * 100
        st.success(f"🔋 **Estimated Renewable Share:** {renewable_ratio:.2f}% of total energy consumption")
    else:
        st.warning("Renewable/non-renewable data not sufficient to calculate ratio.")


# 🔮 Future Energy Forecast
elif page == "Future Energy Forecast":
    import streamlit as st
    import pandas as pd
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly # For Prophet's native Plotly plots
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import plotly.graph_objects as go

    st.title("🔮 Future Energy Forecast with Machine Learning")
    st.info("""
    This module compares two machine learning models – **Prophet** and **Random Forest** – to forecast future energy consumption based on historical data.
    Select a country and energy type, adjust prediction length, and validate model accuracy using backtesting and insights.
    """)

    # 📌 Load data (assuming df is already loaded globally in your Streamlit app)
    # If not, you'll need to load it here:
    # df = pd.read_csv("owid-energy-data.csv")

    # 📌 Selection and data prep
    energy_cols = sorted([col for col in df.columns if col.endswith("_consumption") and df[col].dtype != 'object'])
    
    if not energy_cols:
        st.error("No energy consumption columns found in the dataset.")
        st.stop()

    countries = sorted(df["country"].unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_country = st.selectbox("🌍 Select a Country:", countries, index=countries.index("World") if "World" in countries else 0)
    with col2:
        selected_source = st.selectbox("⚡ Select Energy Type:", energy_cols, index=energy_cols.index("primary_energy_consumption") if "primary_energy_consumption" in energy_cols else 0)
    with col3:
        future_years = st.slider("🗓️ Years to Predict:", 1, 20, 5)

    # Prepare data: Select specific country and energy source first, then drop NaNs.
    # This is more robust than dropping NaNs from all energy_cols at once.
    country_data = df[df["country"] == selected_country][["year", selected_source]].copy()
    country_data.dropna(subset=[selected_source], inplace=True)
    country_data["year"] = pd.to_numeric(country_data["year"]) # Ensure year is numeric
    country_data = country_data.sort_values(by="year")


    if country_data.empty or len(country_data) < 5: # Prophet requires at least 2 data points, more for seasonality.
        st.warning(f"⚠️ Not enough valid data points for {selected_country} and {selected_source.replace('_consumption','').title()} (found {len(country_data)}). Need at least 5 data points for reliable forecasting.")
        st.stop()

    # --- Prophet Forecast ---
    st.subheader(f"📈 Prophet Forecast for {selected_source.replace('_consumption','').title()}")
    
    prophet_df = country_data.rename(columns={"year": "ds", selected_source: "y"})
    # Ensure 'ds' is datetime. Prophet expects YYYY-MM-DD. We'll use Jan 1st for yearly data.
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"].astype(str) + '-01-01')

    try:
        prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        # Add regressors if you have relevant external data (e.g., GDP, population for this country)
        # prophet_model.add_regressor('gdp') # Example
        prophet_model.fit(prophet_df)

        future_df = prophet_model.make_future_dataframe(periods=future_years, freq="Y") # 'Y' for yearly frequency
        forecast = prophet_model.predict(future_df)

        fig_prophet_forecast = plot_plotly(prophet_model, forecast)
        fig_prophet_forecast.update_layout(
            title=f"Prophet Forecast: {selected_country} – {selected_source.replace('_consumption','').title()}",
            xaxis_title="Year",
            yaxis_title="Predicted Consumption",
            template="plotly_white"
        )
        st.plotly_chart(fig_prophet_forecast, use_container_width=True)

        show_components = st.checkbox("Show Prophet forecast components", value=False)
        if show_components:
            fig_components = plot_components_plotly(prophet_model, forecast)
            st.plotly_chart(fig_components, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during Prophet forecasting: {e}")
        st.stop()

    # --- Random Forest Forecast ---
    # Note: Random Forest is not inherently a time series model.
    # Using only 'year' as a feature is a very simple approach.
    # For better RF forecasts, consider feature engineering (lags, rolling means, etc.).
    st.subheader(f"🌲 Random Forest Forecast for {selected_source.replace('_consumption','').title()}")
    
    rf_df = country_data.copy()
    X = rf_df[["year"]]
    y = rf_df[selected_source]

    if len(rf_df) < 2: # Random Forest needs at least some data to train
        st.warning(f"⚠️ Not enough data points for Random Forest model for {selected_country} and {selected_source.replace('_consumption','').title()} after filtering (found {len(rf_df)}).")
    else:
        try:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_split=2, min_samples_leaf=1) # Basic hyperparams
            rf_model.fit(X, y)

            current_max_year = X["year"].max()
            future_years_rf_series = pd.Series(range(current_max_year + 1, current_max_year + future_years + 1), name="year")
            future_df_rf = pd.DataFrame(future_years_rf_series)
            predictions_rf = rf_model.predict(future_df_rf)

            # Combine historical and forecasted data for plotting
            plot_df_rf = pd.DataFrame({
                'Year': rf_df['year'],
                'Actual Consumption': rf_df[selected_source]
            })
            forecast_plot_df_rf = pd.DataFrame({
                'Year': future_years_rf_series,
                'RF Prediction': predictions_rf
            })

            rf_plot = go.Figure()
            rf_plot.add_trace(go.Scatter(
                x=plot_df_rf['Year'],
                y=plot_df_rf['Actual Consumption'],
                mode="lines+markers",
                name="Historical Data",
                line=dict(color="blue")
            ))
            rf_plot.add_trace(go.Scatter(
                x=forecast_plot_df_rf['Year'],
                y=forecast_plot_df_rf['RF Prediction'],
                mode="lines+markers",
                name="RF Prediction",
                line=dict(color="green")
            ))
            rf_plot.update_layout(
                title=f"Random Forest Forecast: {selected_country} – {selected_source.replace('_consumption','').title()}",
                xaxis_title="Year",
                yaxis_title="Consumption",
                template="plotly_white"
            )
            st.plotly_chart(rf_plot, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred during Random Forest forecasting: {e}")


    # --- Forecast Comparison Table ---
    st.subheader("🔍 Prophet vs Random Forest Forecast Comparison")
    # Ensure forecast and predictions_rf are available
    if 'forecast' in locals() and 'predictions_rf' in locals() and 'future_years_rf_series' in locals():
        # Prophet forecast extraction
        prophet_forecast_values = forecast[['ds', 'yhat']].tail(future_years).copy()
        prophet_forecast_values["Year"] = prophet_forecast_values["ds"].dt.year
        
        # Align years for comparison. Prophet's future_df might have slightly different end dates
        # if freq='Y' aligns to year-end, while RF is just incrementing year numbers.
        # We will use the RF years as the common base for the comparison table.
        
        comparison_df = pd.DataFrame({"Year": future_years_rf_series.values})
        
        # Merge Prophet predictions
        # Need to align prophet output years with future_years_rf_series
        prophet_comparison_data = prophet_forecast_values[prophet_forecast_values['Year'].isin(future_years_rf_series)]
        
        # Handle cases where Prophet might not have produced the exact number of future years
        # or years don't perfectly align after yearly conversion.
        # We can create a temporary mapping from Prophet's 'Year' to the RF 'Year' sequence.
        
        prophet_preds_for_table = []
        rf_preds_for_table = []

        # Get the last 'future_years' from Prophet that align with yearly frequency.
        # Prophet's 'ds' is the start of the year when freq='Y' is used for make_future_dataframe
        # So ds.dt.year should match.
        
        prophet_filtered_preds = forecast[forecast['ds'] >= pd.to_datetime(str(country_data['year'].max() + 1) + '-01-01')][['ds', 'yhat']].head(future_years)
        prophet_filtered_preds['Year'] = prophet_filtered_preds['ds'].dt.year
        
        comparison_df = pd.DataFrame({
            "Year": future_years_rf_series.values,
            "Prophet_Prediction": prophet_filtered_preds["yhat"].values if len(prophet_filtered_preds) == future_years else [np.nan]*future_years, # ensure same length
            "RF_Prediction": predictions_rf
        })
        st.dataframe(comparison_df.style.format({"Prophet_Prediction": "{:.2f}", "RF_Prediction": "{:.2f}"}))
    else:
        st.warning("Could not display forecast comparison table as one or both models did not produce predictions.")


    # --- Backtesting ---
    st.subheader("🧪 Backtesting: Prophet & Random Forest Accuracy")

    min_year_data = int(country_data["year"].min())
    max_year_data = int(country_data["year"].max())
    
    # Ensure split_year allows for at least 'future_years' for testing and some data for training
    # Min value for slider: min_year_data + (some training period, e.g., 3-5 years)
    # Max value for slider: max_year_data - future_years
    
    if (max_year_data - future_years) < (min_year_data + 4) : # Need at least 1 year for training and `future_years` for testing
         st.warning(f"Not enough historical data for backtesting with {future_years} prediction years. Minimum {future_years + 5} years of data required for this country and energy source.")
    else:
        split_year = st.slider(
            "📆 Select Last Training Year (for Backtesting):",
            min_value=min_year_data + 4, # Minimum 5 years of training data
            max_value=max_year_data - future_years,
            value=max(min_year_data + 4, max_year_data - future_years - 5) # Default to a reasonable value
        )

        df_train_bt = country_data[country_data["year"] <= split_year]
        # Test data should be the 'future_years' immediately following the split_year
        df_test_actual_bt = country_data[(country_data["year"] > split_year) & (country_data["year"] <= split_year + future_years)]

        if len(df_test_actual_bt) < future_years:
            st.warning(f"⚠️ Not enough actual data points for the selected backtesting period (need {future_years} years, found {len(df_test_actual_bt)} for years {split_year + 1}-{split_year + future_years}). Adjust slider or prediction length.")
        elif len(df_train_bt) < 2: # Need at least 2 for Prophet, more for good results
            st.warning(f"⚠️ Not enough training data points ({len(df_train_bt)}) for backtesting with split year {split_year}.")
        else:
            rmse_prophet, rmse_rf = np.nan, np.nan # Initialize with NaN
            prophet_backtest_preds_values = []
            rf_backtest_preds_values = []

            # Backtesting Prophet
            try:
                prophet_train_df_bt = df_train_bt.rename(columns={"year": "ds", selected_source: "y"})
                prophet_train_df_bt["ds"] = pd.to_datetime(prophet_train_df_bt["ds"].astype(str) + '-01-01')

                test_model_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
                test_model_prophet.fit(prophet_train_df_bt)
                
                future_test_prophet = test_model_prophet.make_future_dataframe(periods=future_years, freq="Y")
                forecast_test_prophet = test_model_prophet.predict(future_test_prophet)
                
                # Align prophet predictions with actual test years
                prophet_backtest_preds = forecast_test_prophet[forecast_test_prophet['ds'].dt.year.isin(df_test_actual_bt['year'])]['yhat']
                # Ensure the length matches df_test_actual_bt for RMSE calculation
                prophet_backtest_preds_values = prophet_backtest_preds.head(len(df_test_actual_bt)).values


                if len(prophet_backtest_preds_values) == len(df_test_actual_bt):
                     rmse_prophet = np.sqrt(mean_squared_error(df_test_actual_bt[selected_source], prophet_backtest_preds_values))
                else:
                    st.warning("Prophet backtest prediction length mismatch.")


            except Exception as e:
                st.warning(f"Error during Prophet backtesting: {e}")

            # Backtesting Random Forest
            try:
                X_train_bt = df_train_bt[["year"]]
                y_train_bt = df_train_bt[selected_source]
                X_test_bt = df_test_actual_bt[["year"]]

                if not X_train_bt.empty and not y_train_bt.empty:
                    test_model_rf = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_split=2, min_samples_leaf=1)
                    test_model_rf.fit(X_train_bt, y_train_bt)
                    rf_backtest_preds_values = test_model_rf.predict(X_test_bt)
                    
                    if len(rf_backtest_preds_values) == len(df_test_actual_bt):
                        rmse_rf = np.sqrt(mean_squared_error(df_test_actual_bt[selected_source], rf_backtest_preds_values))
                    else:
                        st.warning("Random Forest backtest prediction length mismatch.")

                else:
                    st.warning("Not enough data for Random Forest backtesting after split.")
            except Exception as e:
                st.warning(f"Error during Random Forest backtesting: {e}")

            # Display Backtesting Results
            df_compare_bt = pd.DataFrame({
                "Year": df_test_actual_bt["year"].values,
                "Actual": df_test_actual_bt[selected_source].values,
            })
            if len(prophet_backtest_preds_values) == len(df_test_actual_bt):
                 df_compare_bt["Prophet_Prediction_BT"] = prophet_backtest_preds_values
            else:
                 df_compare_bt["Prophet_Prediction_BT"] = np.nan

            if len(rf_backtest_preds_values) == len(df_test_actual_bt):
                 df_compare_bt["RF_Prediction_BT"] = rf_backtest_preds_values
            else:
                 df_compare_bt["RF_Prediction_BT"] = np.nan
            
            st.dataframe(df_compare_bt.style.format({
                "Actual": "{:.2f}", "Prophet_Prediction_BT": "{:.2f}", "RF_Prediction_BT": "{:.2f}"
            }))
            
            if not np.isnan(rmse_prophet):
                st.markdown(f"📉 **Prophet Backtesting RMSE:** {rmse_prophet:.2f}")
            if not np.isnan(rmse_rf):
                st.markdown(f"🌲 **Random Forest Backtesting RMSE:** {rmse_rf:.2f}")

            # Plot Backtesting
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=df_compare_bt["Year"], y=df_compare_bt["Actual"],
                                     mode="lines+markers", name="Actual"))
            if "Prophet_Prediction_BT" in df_compare_bt and not df_compare_bt["Prophet_Prediction_BT"].isnull().all():
                fig_bt.add_trace(go.Scatter(x=df_compare_bt["Year"], y=df_compare_bt["Prophet_Prediction_BT"],
                                         mode="lines+markers", name="Prophet (Backtest)"))
            if "RF_Prediction_BT" in df_compare_bt and not df_compare_bt["RF_Prediction_BT"].isnull().all():
                fig_bt.add_trace(go.Scatter(x=df_compare_bt["Year"], y=df_compare_bt["RF_Prediction_BT"],
                                         mode="lines+markers", name="Random Forest (Backtest)"))
            fig_bt.update_layout(
                title="📊 Backtesting: Actual vs Predicted Energy Consumption",
                xaxis_title="Year",
                yaxis_title="Energy Consumption",
                template="plotly_white"
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            # --- Insight Section ---
            st.subheader("💡 Forecasting Insights")
            if not np.isnan(rmse_prophet) and not np.isnan(rmse_rf):
                if rmse_prophet < rmse_rf:
                    stronger_model = "Prophet"
                    weaker_model = "Random Forest"
                    st.success(f"🔍 Based on RMSE, **Prophet ({rmse_prophet:.2f})** performed better than Random Forest ({rmse_rf:.2f}) in this backtesting scenario.")
                elif rmse_rf < rmse_prophet:
                    stronger_model = "Random Forest"
                    weaker_model = "Prophet"
                    st.success(f"🔍 Based on RMSE, **Random Forest ({rmse_rf:.2f})** performed better than Prophet ({rmse_prophet:.2f}) in this backtesting scenario.")
                else:
                    st.info(f"🔍 Both Prophet and Random Forest performed similarly (RMSE: {rmse_prophet:.2f}) in this backtesting scenario.")
                
                st.markdown("""
                **General Considerations:**
                * **Prophet** is generally well-suited for time series data with trends and seasonality.
                * **Random Forest**, when used with only 'year' as a feature, acts as a simple regression model. Its performance can often be improved with more sophisticated feature engineering (e.g., lagged values, rolling averages) for time series tasks.
                * The amount and quality of historical data significantly impact forecast accuracy.
                * Backtesting on a different period or with different parameters might yield different results.
                """)
            elif not np.isnan(rmse_prophet):
                 st.info("Prophet model backtested. Random Forest backtesting had issues or insufficient data.")
            elif not np.isnan(rmse_rf):
                 st.info("Random Forest model backtested. Prophet backtesting had issues or insufficient data.")
            else:
                st.warning("Could not determine a stronger model as RMSE values were not available for one or both models from backtesting.")
