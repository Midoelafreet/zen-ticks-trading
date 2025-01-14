# First section - imports and setup
import streamlit as st
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pytz
from scipy import stats

# Set page configuration
st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")

# Timestamp function
def get_current_timestamp():
    ny_tz = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_tz)
    formatted_time = ny_time.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time

# Title and main app setup
st.title("Trading Strategy Monte Carlo Simulation Dashboard")
st.markdown("Multi-Instrument Trading Strategy Analysis")

# Sidebar configurations
with st.sidebar:
    # Create a container for the clock with white text
    st.markdown(
        """
        <style>
            .clock-container { margin-bottom: 20px; }
            .clock { 
                font-family: monospace; 
                font-size: 14px;
                color: white !important;
            }
            .white-text {
                color: white !important;
            }
            .stMarkdown div {
                color: white !important;
            }
            .element-container div {
                color: white !important;
            }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Display current time and user info
    st.markdown(f'<p style="color: white;"><strong>Current Date and Time (UTC):</strong> {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: white;"><strong>Current User\'s Login:</strong> Midoelafreet</p>', unsafe_allow_html=True)
    st.markdown('<hr style="margin: 15px 0;">', unsafe_allow_html=True)
    st.header("Trading Parameters")
    
    # Monte Carlo Parameters
    with st.expander("Monte Carlo Settings", expanded=True):
        num_simulations = st.number_input(
            "Number of Simulations", 
            value=10000, 
            min_value=1000, 
            max_value=50000
        )
        starting_capital = st.number_input(
            "Starting Capital ($)", 
            value=50000, 
            min_value=1000
        )
        profit_target = st.number_input(
            "Profit Target ($)", 
            value=3000, 
            min_value=100
        )
        loss_limit = st.number_input(
            "Loss Limit ($)", 
            value=2000, 
            min_value=100
        )
        daily_risk_limit = st.number_input(
            "Daily Risk Limit ($)", 
            value=500, 
            min_value=100
        )
        win_rate = st.slider(
            "Win Rate (%)", 
            min_value=0, 
            max_value=100, 
            value=40
        ) / 100

    # Contract Specifications
    with st.expander("Contract Specifications", expanded=True):
        col1, col2 = st.columns(2)
        
        # ZB Contract Settings
        with col1:
            st.subheader("ZB Contract")
            zb_tick_value = st.number_input(
                "ZB Tick Value ($)", 
                value=31.25, 
                step=0.01
            )
            zb_contracts = st.number_input(
                "Number of ZB Contracts", 
                value=1, 
                min_value=1
            )
            zb_commission = st.number_input(
                "ZB Commission ($ per RT)", 
                value=3.25, 
                step=0.01
            )
            
        # Yen Contract Settings
        with col2:
            st.subheader("Yen Contract")
            yen_tick_value = st.number_input(
                "Yen Tick Value ($)", 
                value=6.25, 
                step=0.01
            )
            yen_contracts = st.number_input(
                "Number of Yen Contracts", 
                value=1, 
                min_value=1
            )
            yen_commission = st.number_input(
                "Yen Commission ($ per RT)", 
                value=3.25, 
                step=0.01
            )

    # Trading Parameters
    with st.expander("Trading Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            trades_per_day_zb = st.number_input(
                "ZB Trades per Day", 
                value=4, 
                min_value=1
            )
        with col2:
            trades_per_day_yen = st.number_input(
                "Yen Trades per Day", 
                value=4, 
                min_value=1
            )

# Risk Manager Calculator
st.header("Daily Risk Manager Calculator")
with st.expander("Risk Management Calculator", expanded=True):
    # Initialize session state
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
    
    trade_col1, trade_col2, trade_col3 = st.columns(3)
    
    with trade_col1:
        market = st.selectbox("Select Market", ["ZB", "Yen"])
        num_contracts = st.number_input("Number of Contracts", min_value=1, value=1)
        trade_result = st.radio("Trade Result", ["Win", "Loss"])
        
    with trade_col2:
        risk_ticks = st.number_input(
            "Risk (Ticks)", 
            min_value=1, 
            value=3,
            help="Number of ticks risking on this trade"
        )
        profit_ticks = st.number_input(
            "Profit Target (Ticks)", 
            min_value=1, 
            value=6,
            help="Number of ticks targeting for profit"
        )
        
        tick_value = zb_tick_value if market == "ZB" else yen_tick_value
        commission = zb_commission if market == "ZB" else yen_commission
        
        if trade_result == "Win":
            potential_pnl = (profit_ticks * tick_value * num_contracts) - (commission * num_contracts)
        else:
            potential_pnl = (-risk_ticks * tick_value * num_contracts) - (commission * num_contracts)
            
    with trade_col3:
        st.metric("Tick Value", f"${tick_value:.2f}")
        st.metric("Potential P&L", f"${potential_pnl:.2f}")
        
        if (market == "ZB" and st.session_state.zb_trades_taken >= trades_per_day_zb) or \
           (market == "Yen" and st.session_state.yen_trades_taken >= trades_per_day_yen):
            st.warning(f"⚠️ Maximum {market} trades for the day reached!")
        
        if st.session_state.current_daily_pnl + potential_pnl < -daily_risk_limit:
            st.warning("⚠️ This trade would exceed your daily risk limit!")

    # Add Trade Button
    if st.button("Add Trade"):
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
            st.session_state.current_daily_pnl += potential_pnl
            if market == "ZB":
                st.session_state.zb_trades_taken += 1
            else:
                st.session_state.yen_trades_taken += 1
                
            trade_data = {
                'market': market,
                'contracts': num_contracts,
                'result': trade_result,
                'risk_ticks': risk_ticks,
                'profit_ticks': profit_ticks,
                'pnl': potential_pnl,
                'timestamp': get_current_timestamp()
            }
            st.session_state.trade_history.append(trade_data)
            
            st.success(f"Trade added successfully! New daily P&L: ${st.session_state.current_daily_pnl:.2f}")
            st.rerun()

    # Reset Daily Stats Button
    if st.button("Reset Daily Stats"):
        st.session_state.current_daily_pnl = 0.0
        st.session_state.zb_trades_taken = 0
        st.session_state.yen_trades_taken = 0
        st.session_state.trade_history = []
        st.success("Daily stats reset successfully!")
        st.rerun()

    # Trade History Section
    if st.session_state.trade_history:
        st.subheader("Today's Trade History")
        df_trades = pd.DataFrame(st.session_state.trade_history)
        st.dataframe(df_trades, use_container_width=True)
        
        # Download button for trade history
        csv = df_trades.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History",
            data=csv,
            file_name=f"trade_history_{datetime.now().strftime('%Y-%m-%d')}.csv",
            mime="text/csv"
        )
        
        # Display current data
        st.dataframe(df_export, use_container_width=True)
    else:
        st.info("No trades to export yet")

    
# Main content area - Simulation execution
if st.button("Run Monte Carlo Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        # Run the simulation
        final_capitals, daily_returns, days_to_complete, \
        max_losing_streaks, max_winning_streaks, all_trades = run_monte_carlo(
            starting_capital=starting_capital,
            profit_target=profit_target,
            loss_limit=loss_limit,
            daily_risk_limit=daily_risk_limit,
            win_rate=win_rate,
            zb_trades_per_day=trades_per_day_zb,
            yen_trades_per_day=trades_per_day_yen,
            zb_win=zb_win,
            zb_loss=zb_loss,
            yen_win=yen_win,
            yen_loss=yen_loss,
            zb_contracts=zb_contracts,
            yen_contracts=yen_contracts,
            num_simulations=num_simulations
        )
        
        # Calculate key statistics
        profitable_sims = sum(1 for cap in final_capitals if cap > starting_capital)
        profit_probability = profitable_sims / num_simulations * 100
        
        avg_days = sum(days_to_complete) / len(days_to_complete)
        max_days = max(days_to_complete)
        min_days = min(days_to_complete)
        
        risk_of_ruin = calculate_risk_of_ruin(final_capitals, starting_capital, loss_limit)
        
        # Display summary statistics
        st.header("Simulation Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Profit Probability", f"{profit_probability:.1f}%")
            st.metric("Risk of Ruin", f"{risk_of_ruin:.1f}%")
        with col2:
            st.metric("Average Days", f"{avg_days:.1f}")
            st.metric("Maximum Days", f"{max_days}")
        with col3:
            st.metric("Minimum Days", f"{min_days}")
            st.metric("Total Simulations", f"{num_simulations:,}")

# Visualization section
if 'final_capitals' in locals():
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Capital Distribution", 
        "Daily Returns", 
        "Trading Streaks", 
        "Detailed Statistics"
    ])

    with tab1:
        st.subheader("Final Capital Distribution")
        
        # Create distribution plot using plotly
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=final_capitals,
            nbinsx=50,
            name="Capital Distribution",
            opacity=0.75
        ))
        
        # Add vertical lines for key metrics
        fig.add_vline(x=starting_capital, line_dash="dash", line_color="yellow", annotation_text="Starting Capital")
        fig.add_vline(x=starting_capital + profit_target, line_dash="dash", line_color="green", annotation_text="Profit Target")
        fig.add_vline(x=starting_capital - loss_limit, line_dash="dash", line_color="red", annotation_text="Loss Limit")
        
        fig.update_layout(
            title="Distribution of Final Capital Across All Simulations",
            xaxis_title="Final Capital ($)",
            yaxis_title="Frequency",
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Daily Returns Analysis")
        
        # Create daily returns distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=daily_returns,
            nbinsx=50,
            name="Daily Returns",
            opacity=0.75
        ))
        
        fig.update_layout(
            title="Distribution of Daily Returns",
            xaxis_title="Daily Return ($)",
            yaxis_title="Frequency",
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Trading Streaks Analysis")
        
        # Calculate streak statistics
        streak_stats = calculate_streak_stats(all_trades)
        
        # Create streak distribution plots
        fig = go.Figure()
        
        # Winning streaks
        fig.add_trace(go.Histogram(
            x=streak_stats['winning'],
            name="Winning Streaks",
            opacity=0.75,
            marker_color='green'
        ))
        
        # Losing streaks
        fig.add_trace(go.Histogram(
            x=streak_stats['losing'],
            name="Losing Streaks",
            opacity=0.75,
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Distribution of Trading Streaks",
            xaxis_title="Streak Length",
            yaxis_title="Frequency",
            barmode='overlay',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display streak statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Winning Streak", f"{max(max_winning_streaks)}")
            st.metric("Avg Winning Streak", f"{sum(streak_stats['winning']) / len(streak_stats['winning']):.2f}")
        with col2:
            st.metric("Max Losing Streak", f"{max(max_losing_streaks)}")
            st.metric("Avg Losing Streak", f"{sum(streak_stats['losing']) / len(streak_stats['losing']):.2f}")

    with tab4:
        st.subheader("Detailed Statistics")
        
        # Calculate additional statistics
        final_capitals_array = np.array(final_capitals)
        daily_returns_array = np.array(daily_returns)
        
        stats_data = {
            "Metric": [
                "Mean Final Capital",
                "Median Final Capital",
                "Std Dev Final Capital",
                "Skewness Final Capital",
                "Kurtosis Final Capital",
                "Mean Daily Return",
                "Median Daily Return",
                "Std Dev Daily Return",
                "Sharpe Ratio (0% risk-free)",
                "Profit Factor"
            ],
            "Value": [
                f"${np.mean(final_capitals_array):.2f}",
                f"${np.median(final_capitals_array):.2f}",
                f"${np.std(final_capitals_array):.2f}",
                f"{stats.skew(final_capitals_array):.3f}",
                f"{stats.kurtosis(final_capitals_array):.3f}",
                f"${np.mean(daily_returns_array):.2f}",
                f"${np.median(daily_returns_array):.2f}",
                f"${np.std(daily_returns_array):.2f}",
                f"{np.mean(daily_returns_array) / np.std(daily_returns_array):.3f}",
                f"{abs(sum(r for r in daily_returns if r > 0)) / abs(sum(r for r in daily_returns if r < 0)):.3f}"
            ]
        }
        
        # Create and display the statistics table
        stats_df = pd.DataFrame(stats_data)
        st.table(stats_df.set_index('Metric'))
