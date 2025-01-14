import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="The Zen Ticks Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Constants
daily_risk_limit = 2000  # Daily risk limit in dollars
trades_per_day_zb = 3    # Maximum ZB trades per day
trades_per_day_yen = 3   # Maximum Yen trades per day
zb_win = 300            # ZB win amount
zb_loss = -600          # ZB loss amount
yen_win = 400           # Yen win amount
yen_loss = -800         # Yen loss amount


# Sidebar Configuration
with st.sidebar:
    st.markdown("**Current Date and Time (UTC):** 2025-01-14 11:19:22")
    st.markdown("**Current User's Login:** Midoelafreet")
    st.markdown("---")
    
    # Add any additional sidebar elements here
    st.markdown("### Trading Parameters")
    st.markdown(f"Daily Risk Limit: ${daily_risk_limit}")
    st.markdown(f"Max ZB Trades: {trades_per_day_zb}")
    st.markdown(f"Max Yen Trades: {trades_per_day_yen}")
    st.markdown("---")
    st.markdown("### P&L Parameters")
    st.markdown(f"ZB Win: ${zb_win}")
    st.markdown(f"ZB Loss: ${zb_loss}")
    st.markdown(f"Yen Win: ${yen_win}")
    st.markdown(f"Yen Loss: ${yen_loss}")

# Risk Manager Calculator
st.header("Daily Risk Manager Calculator")
with st.expander("Risk Management Calculator", expanded=True):
    # Initialize session state for tracking daily P&L if it doesn't exist
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
        st.metric("Remaining Daily Risk", f"${max(daily_risk_limit + st.session_state.current_daily_pnl, 0):.2f}")
    with col2:
        st.metric("ZB Trades Taken", f"{st.session_state.zb_trades_taken}/{trades_per_day_zb}")
        st.metric("ZB Trades Remaining", f"{max(trades_per_day_zb - st.session_state.zb_trades_taken, 0)}")
    with col3:
        st.metric("Yen Trades Taken", f"{st.session_state.yen_trades_taken}/{trades_per_day_yen}")
        st.metric("Yen Trades Remaining", f"{max(trades_per_day_yen - st.session_state.yen_trades_taken, 0)}")

    # Trade Entry Section
    st.subheader("Enter Trade Result")
    
    # Create two columns for trade entry
    trade_col1, trade_col2 = st.columns(2)
    
    with trade_col1:
        # Trade Input Fields
        market = st.selectbox("Select Market", ["ZB", "Yen"])
        num_contracts = st.number_input("Number of Contracts", min_value=1, value=1)
        trade_result = st.selectbox("Trade Result", ["Win", "Loss"])

    with trade_col2:
        # Calculate potential P&L
        if market == "ZB":
            potential_pnl = num_contracts * (zb_win if trade_result == "Win" else zb_loss)
        else:  # Yen
            potential_pnl = num_contracts * (yen_win if trade_result == "Win" else yen_loss)
        
        st.metric("Potential P&L", f"${potential_pnl:.2f}")
        
        # Show warnings if applicable
        if market == "ZB" and st.session_state.zb_trades_taken >= trades_per_day_zb:
            st.warning("Maximum ZB trades reached for today!")
        elif market == "Yen" and st.session_state.yen_trades_taken >= trades_per_day_yen:
            st.warning("Maximum Yen trades reached for today!")
        
        if st.session_state.current_daily_pnl + potential_pnl < -daily_risk_limit:
            st.warning("This trade would exceed your daily risk limit!")

# PnL Monitoring and Export Section
st.header("PnL Monitoring & Analytics")
with st.expander("Daily Performance Charts", expanded=True):
    # Initialize session state for tracking trade history if it doesn't exist
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = []
    
    # Create columns for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Cumulative PnL Chart
        if st.session_state.trade_history:
            df_trades = pd.DataFrame(st.session_state.trade_history)
            df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
            
            fig_pnl = px.line(df_trades, 
                            y='cumulative_pnl',
                            title='Cumulative PnL Throughout the Day',
                            labels={'cumulative_pnl': 'Cumulative PnL ($)',
                                   'index': 'Trade Number'})
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("No trades recorded yet today")
    
    with chart_col2:
        # Win Rate by Market
        if st.session_state.trade_history:
            win_rates = {}
            for market in ['ZB', 'Yen']:
                market_trades = [t for t in st.session_state.trade_history if t['market'] == market]
                if market_trades:
                    wins = len([t for t in market_trades if t['pnl'] > 0])
                    total = len(market_trades)
                    win_rates[market] = (wins / total) * 100
            
            fig_winrate = px.bar(
                x=list(win_rates.keys()),
                y=list(win_rates.values()),
                title='Win Rate by Market',
                labels={'x': 'Market', 'y': 'Win Rate (%)'}
            )
            fig_winrate.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_winrate, use_container_width=True)
        else:
            st.info("No trades recorded yet today")

    # Trade Statistics
    st.subheader("Today's Trading Statistics")
    if st.session_state.trade_history:
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        df_trades = pd.DataFrame(st.session_state.trade_history)
        with stats_col1:
            st.metric("Total Trades", len(df_trades))
            st.metric("Total PnL", f"${df_trades['pnl'].sum():.2f}")
        
        with stats_col2:
            wins = len(df_trades[df_trades['pnl'] > 0])
            total = len(df_trades)
            st.metric("Overall Win Rate", f"{(wins/total*100):.1f}%" if total > 0 else "N/A")
            st.metric("Average Trade PnL", f"${df_trades['pnl'].mean():.2f}" if total > 0 else "N/A")
        
        with stats_col3:
            st.metric("Largest Win", f"${df_trades['pnl'].max():.2f}" if total > 0 else "N/A")
            st.metric("Largest Loss", f"${df_trades['pnl'].min():.2f}" if total > 0 else "N/A")

# Export Section and Buttons
st.subheader("Export Daily Trading Data")
if st.session_state.trade_history:
    df_export = pd.DataFrame(st.session_state.trade_history)
    
    # Add timestamp to the export
    current_time = "2025-01-14 11:21:36"  # Current timestamp
    csv = df_export.to_csv(index=False)
    
    st.download_button(
        label="Download Trading Data (CSV)",
        data=csv,
        file_name=f"trading_data_{current_time.split()[0]}.csv",
        mime="text/csv",
    )
    
    # Display current data in a table
    st.dataframe(df_export, use_container_width=True)
else:
    st.info("No trades to export yet")


# Add Trade and Reset Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Add Trade"):
        # Check if trade is allowed
        trade_allowed = True
        if market == "ZB" and st.session_state.zb_trades_taken >= trades_per_day_zb:
            trade_allowed = False
            st.error("Cannot add trade: Maximum ZB trades reached for the day")
        elif market == "Yen" and st.session_state.yen_trades_taken >= trades_per_day_yen:
            trade_allowed = False
            st.error("Cannot add trade: Maximum Yen trades reached for the day")
        elif st.session_state.current_daily_pnl + potential_pnl < -daily_risk_limit:
            trade_allowed = False
            st.error("Cannot add trade: Would exceed daily risk limit")
        
        if trade_allowed:
            # Update P&L and trade counts
            st.session_state.current_daily_pnl += potential_pnl
            if market == "ZB":
                st.session_state.zb_trades_taken += 1
            else:
                st.session_state.yen_trades_taken += 1
                
            # Add trade to history
            trade_data = {
                'market': market,
                'contracts': num_contracts,
                'result': trade_result,
                'pnl': potential_pnl,
                'timestamp': "2025-01-14 11:21:36"  # Current timestamp
            }
            st.session_state.trade_history.append(trade_data)
            
            st.success(f"Trade added successfully! New daily P&L: ${st.session_state.current_daily_pnl:.2f}")
            st.rerun()

with col2:
    if st.button("Reset Daily Stats"):
        st.session_state.current_daily_pnl = 0.0
        st.session_state.zb_trades_taken = 0
        st.session_state.yen_trades_taken = 0
        st.session_state.trade_history = []  # Clear trade history
        st.success("Daily stats reset successfully!")
        st.rerun()


# Monte Carlo Simulation Section
st.header("Monte Carlo Simulation")
with st.expander("Trading Simulation", expanded=False):
    # Simulation parameters
    num_simulations = st.slider("Number of Simulations", 100, 1000, 500)
    num_trades = st.slider("Number of Trades per Day", 1, 10, 6)
    
    if st.button("Run Simulation"):
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Initialize results array
        simulation_results = []
        
        # Run simulations
        for i in range(num_simulations):
            daily_pnl = 0
            for _ in range(num_trades):
                # Randomly select market and outcome
                market = np.random.choice(['ZB', 'Yen'])
                outcome = np.random.choice(['Win', 'Loss'], p=[0.5, 0.5])
                
                # Calculate trade PnL
                if market == 'ZB':
                    pnl = zb_win if outcome == 'Win' else zb_loss
                else:
                    pnl = yen_win if outcome == 'Win' else yen_loss
                
                daily_pnl += pnl
            
            simulation_results.append(daily_pnl)
            progress_bar.progress((i + 1) / num_simulations)
        
        # Calculate statistics
        mean_pnl = np.mean(simulation_results)
        std_pnl = np.std(simulation_results)
        worst_day = np.min(simulation_results)
        best_day = np.max(simulation_results)
        
        # Create figure with simulation results
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(simulation_results, bins=30)
        plt.axvline(mean_pnl, color='g', linestyle='dashed', linewidth=2)
        plt.axvline(worst_day, color='r', linestyle='dashed', linewidth=2)
        plt.axvline(best_day, color='b', linestyle='dashed', linewidth=2)
        plt.title('Distribution of Daily P&L Outcomes')
        plt.xlabel('Daily P&L ($)')
        plt.ylabel('Frequency')
        st.pyplot(fig)
        
        # Display statistics
        st.subheader("Simulation Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Daily P&L", f"${mean_pnl:.2f}")
        col2.metric("Standard Deviation", f"${std_pnl:.2f}")
        col3.metric("Worst Day", f"${worst_day:.2f}")
        col4.metric("Best Day", f"${best_day:.2f}")
