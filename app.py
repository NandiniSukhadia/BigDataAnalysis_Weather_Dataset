import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit page configuration
st.set_page_config(page_title="Weather Data Dashboard", layout="wide")

# App title
st.title("🌦️ Weather Data Analysis & Forecasting Dashboard")
st.markdown("""
Explore and forecast weather patterns using regression analysis.  
Upload your dataset, visualize insights, and predict future trends 📈.
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your weather dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Show dataset preview
    st.subheader("📋 Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)

    # Basic info
    st.markdown("### 🔍 Basic Information")
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
    st.write("**Columns:**", list(df.columns))

    st.markdown("### ⚠️ Missing Values")
    st.write(df.isnull().sum())

    st.markdown("### 📊 Statistical Summary")
    st.write(df.describe())

    # Visualization
    st.markdown("## 📈 Data Visualizations")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) > 0:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols, key="vis_x")
        with col2:
            y_col = st.selectbox("Select Y-axis", numeric_cols, key="vis_y")

        # Line Chart
        st.markdown("#### 📉 Line Chart")
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.plot(df[x_col], df[y_col], color='orange', linewidth=1)
        ax.set_xlabel(x_col, fontsize=8)
        ax.set_ylabel(y_col, fontsize=8)
        ax.set_title(f"{y_col} vs {x_col}", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # Scatter Plot
        st.markdown("#### 🔸 Scatter Plot")
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax, s=25)
        ax.set_title("Scatter Plot", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # Heatmap
        st.markdown("#### 🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        sns.heatmap(
            df[numeric_cols].corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax,
            annot_kws={"size": 7},
            cbar_kws={"shrink": 0.7}
        )
        ax.set_title("Correlation Heatmap", fontsize=10)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
    else:
        st.warning("⚠️ No numeric columns found for plotting.")

    # Forecasting
    st.markdown("## 🤖 Weather Trend Forecasting (Linear Regression vs Random Forest)")
    st.markdown("""
Both models are trained and compared:  
- **Linear Regression** → simple, interpretable baseline  
- **Random Forest** → handles non-linear relationships, usually more accurate
""")

    if len(numeric_cols) >= 2:
        feature_col = st.selectbox("Select Feature (X)", numeric_cols, key="model_x")
        target_col = st.selectbox("Select Target (Y)", numeric_cols, key="model_y")

        if st.button("🚀 Train Models"):
            X = df[[feature_col]].values
            y = df[target_col].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # --- Linear Regression ---
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            y_pred_lr = lr_model.predict(X_test)
            mse_lr = mean_squared_error(y_test, y_pred_lr)
            r2_lr = r2_score(y_test, y_pred_lr)

            # --- Random Forest ---
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            mse_rf = mean_squared_error(y_test, y_pred_rf)
            r2_rf = r2_score(y_test, y_pred_rf)

            # Show results in two columns
            col1, col2 = st.columns(2)

            with col1:
                st.success("✅ Linear Regression Results")
                st.write(f"**MSE:** {mse_lr:.2f}")
                st.write(f"**R² Score:** {r2_lr:.2f}")

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.scatter(X_test, y_test, color='blue', label='Actual', s=25)
                ax.plot(X_test, y_pred_lr, color='red', linewidth=1.5, label='Predicted')
                ax.set_title("Linear Regression", fontsize=10)
                ax.set_xlabel(feature_col, fontsize=8)
                ax.set_ylabel(target_col, fontsize=8)
                ax.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

            with col2:
                st.success("✅ Random Forest Results")
                st.write(f"**MSE:** {mse_rf:.2f}")
                st.write(f"**R² Score:** {r2_rf:.2f}")

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.scatter(X_test, y_test, color='blue', label='Actual', s=25)
                ax.plot(X_test, y_pred_rf, color='green', linewidth=1.5, label='Predicted")
                ax.set_title("Random Forest", fontsize=10)
                ax.set_xlabel(feature_col, fontsize=8)
                ax.set_ylabel(target_col, fontsize=8)
                ax.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

            # Future Predictions
            st.markdown("### 🔮 Future Predictions")
            future_val = st.number_input(
                f"Future {feature_col} value:", value=float(df[feature_col].mean())
            )
            future_pred_lr = lr_model.predict([[future_val]])[0]
            future_pred_rf = rf_model.predict([[future_val]])[0]

            st.write(f"**Linear Regression → Predicted {target_col}: {future_pred_lr:.2f}**")
            st.write(f"**Random Forest → Predicted {target_col}: {future_pred_rf:.2f}**")

    else:
        st.warning("⚠️ Need at least two numeric columns for regression.")
else:
    st.info("👆 Upload a CSV file to begin.")
