import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="The Zen Ticks Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Constants
daily_risk_limit = 2000
trades_per_day_zb = 3
trades_per_day_yen = 3

# Sidebar Configuration
with st.sidebar:
    st.markdown("**Current Date and Time (UTC):** 2025-01-14 11:31:19")
    st.markdown("**Current User's Login:** Midoelafreet")
    st.markdown("---")

# Risk Manager Calculator
st.header("Risk Manager Calculator")

# Initialize session state if not exists
if 'current_daily_pnl' not in st.session_state:
    st.session_state.current_daily_pnl = 0.0
if 'zb_trades_taken' not in st.session_state:
    st.session_state.zb_trades_taken = 0
if 'yen_trades_taken' not in st.session_state:
    st.session_state.yen_trades_taken = 0
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

# Display current status
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Daily P&L", f"${st.session_state.current_daily_pnl:.2f}")
with col2:
    st.metric("ZB Trades Taken", f"{st.session_state.zb_trades_taken}/{trades_per_day_zb}")
with col3:
    st.metric("Yen Trades Taken", f"{st.session_state.yen_trades_taken}/{trades_per_day_yen}")

# Trade parameters input
market = st.selectbox("Select Market", ["ZB", "Yen"])
trade_type = st.radio("Trade Type", ["Win", "Loss"])
num_contracts = st.number_input("Number of Contracts", min_value=1, value=1)

# Calculate P&L based on market and trade type
if market == "ZB":
    pnl_per_contract = 300 if trade_type == "Win" else -600
else:  # Yen
    pnl_per_contract = 400 if trade_type == "Win" else -800

potential_pnl = pnl_per_contract * num_contracts

# Display potential P&L
st.metric("Potential P&L", f"${potential_pnl:.2f}")

# Add Trade button
if st.button("Add Trade"):
    # Check if trade is allowed
    trade_allowed = True
    if market == "ZB" and st.session_state.zb_trades_taken >= trades_per_day_zb:
        trade_allowed = False
        st.error("Maximum ZB trades reached for today!")
    elif market == "Yen" and st.session_state.yen_trades_taken >= trades_per_day_yen:
        trade_allowed = False
        st.error("Maximum Yen trades reached for today!")
    elif st.session_state.current_daily_pnl + potential_pnl < -daily_risk_limit:
        trade_allowed = False
        st.error("This trade would exceed your daily risk limit!")
    
    if trade_allowed:
        st.session_state.current_daily_pnl += potential_pnl
        if market == "ZB":
            st.session_state.zb_trades_taken += 1
        else:
            st.session_state.yen_trades_taken += 1
            
        # Add trade to history
        trade_data = {
            'market': market,
            'contracts': num_contracts,
            'result': trade_type,
            'pnl': potential_pnl,
            'timestamp': "2025-01-14 11:31:19"
        }
        st.session_state.trade_history.append(trade_data)
        
        st.success(f"Trade added successfully! New daily P&L: ${st.session_state.current_daily_pnl:.2f}")
        st.rerun()

# Reset button
if st.button("Reset Daily Stats"):
    st.session_state.current_daily_pnl = 0.0
    st.session_state.zb_trades_taken = 0
    st.session_state.yen_trades_taken = 0
    st.session_state.trade_history = []
    st.success("Daily stats reset successfully!")
    st.rerun()


# Monte Carlo Simulation
st.header("Monte Carlo Simulation")

# Simulation parameters
col1, col2 = st.columns(2)
with col1:
    win_rate = st.slider("Win Rate (%)", 0, 100, 50)
    num_simulations = st.slider("Number of Simulations", 100, 1000, 500)
with col2:
    num_trades = st.slider("Number of Trades", 1, 10, 6)
    confidence_level = st.slider("Confidence Level (%)", 90, 99, 95)

if st.button("Run Simulation"):
    # Create progress bar
    progress_bar = st.progress(0)
    simulation_results = []
    
    # Run simulations
    for i in range(num_simulations):
        daily_pnl = 0
        for _ in range(num_trades):
            # Randomly determine if trade is win or loss based on win rate
            is_win = np.random.random() < (win_rate / 100)
            
            # Randomly select market
            market = np.random.choice(['ZB', 'Yen'])
            
            # Calculate PnL based on market and outcome
            if market == 'ZB':
                pnl = 300 if is_win else -600
            else:  # Yen
                pnl = 400 if is_win else -800
                
            daily_pnl += pnl
            
        simulation_results.append(daily_pnl)
        progress_bar.progress((i + 1) / num_simulations)
    
    # Calculate statistics
    mean_pnl = np.mean(simulation_results)
    std_pnl = np.std(simulation_results)
    confidence_interval = stats.norm.interval(confidence_level/100, loc=mean_pnl, scale=std_pnl)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Daily P&L", f"${mean_pnl:.2f}")
        st.metric("Standard Deviation", f"${std_pnl:.2f}")
    with col2:
        st.metric("Worst Day", f"${min(simulation_results):.2f}")
        st.metric("Best Day", f"${max(simulation_results):.2f}")
    with col3:
        st.metric(f"{confidence_level}% CI Lower", f"${confidence_interval[0]:.2f}")
        st.metric(f"{confidence_level}% CI Upper", f"${confidence_interval[1]:.2f}")
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(simulation_results, bins=30, ax=ax)
    ax.axvline(mean_pnl, color='g', linestyle='dashed', linewidth=2, label='Mean')
    ax.axvline(confidence_interval[0], color='r', linestyle='dashed', linewidth=2, label=f'{confidence_level}% CI')
    ax.axvline(confidence_interval[1], color='r', linestyle='dashed', linewidth=2)
    ax.set_title('Distribution of Daily P&L Outcomes')
    ax.set_xlabel('Daily P&L ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)


# Export Section
st.header("Export Trading Data")
if st.session_state.trade_history:
    df_export = pd.DataFrame(st.session_state.trade_history)
    
    # Add timestamp to export
    current_time = "2025-01-14 11:31:19"
    csv = df_export.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Trading Data (CSV)",
        data=csv,
        file_name=f"trading_data_{current_time.split()[0]}.csv",
        mime="text/csv",
        key="download_csv",
        help="Click to download your trading data as a CSV file"
    )
    
    # Display current data
    st.dataframe(df_export, use_container_width=True)
else:
    st.info("No trades to export yet. Add some trades using the Risk Manager Calculator above!")
