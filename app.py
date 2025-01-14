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
from scipy import stats  # Add this line

# Set page configuration
st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")

# Timestamp function
def get_current_timestamp():
    ny_tz = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_tz)
    formatted_time = ny_time.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time



# Auto-refresh script
st.markdown(
    """
    <script>
        function updateClock() {
            const options = {
                timeZone: 'America/New_York',
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false
            };
            const now = new Date();
            const timeStr = now.toLocaleString('en-US', options)
                .replace(',', '')
                .replace(/(\d+)\/(\d+)\/(\d+)/, '$3-$1-$2');
            document.getElementById('clock').innerHTML = timeStr;
            requestAnimationFrame(updateClock);
        }
        if (document.getElementById('clock')) {
            updateClock();
        } else {
            document.addEventListener('DOMContentLoaded', updateClock);
        }
    </script>
    """,
    unsafe_allow_html=True
)


# Second section - simulation function
def simulate_trading_session(
    starting_capital: float,
    profit_target: float,
    loss_limit: float,
    daily_risk_limit: float,
    win_rate: float,
    zb_trades_per_day: int,
    yen_trades_per_day: int,
    zb_win: float,
    zb_loss: float,
    yen_win: float,
    yen_loss: float,
    zb_contracts: int,
    yen_contracts: int,
    max_trading_days: int = 90,
    max_consecutive_losses: int = 5,
    position_reduction: float = 0.5
) -> Tuple[float, List[float], int, int, int, List[bool]]:
    
    capital = starting_capital
    trading_days = 0
    daily_returns = []
    current_losing_streak = 0
    current_winning_streak = 0
    max_losing_streak = 0
    max_winning_streak = 0
    all_trades_results = []
    
    zb_position_multiplier = 1.0
    yen_position_multiplier = 1.0

    while trading_days < max_trading_days:
        trading_days += 1
        daily_trades_result = []
        
        total_trades = zb_trades_per_day + yen_trades_per_day
        for _ in range(total_trades):
            is_zb = len(daily_trades_result) < zb_trades_per_day
            
            if np.random.random() < win_rate:
                if is_zb:
                    trade_result = zb_win * zb_contracts * zb_position_multiplier
                else:
                    trade_result = yen_win * yen_contracts * yen_position_multiplier
                all_trades_results.append(True)
                current_winning_streak += 1
                current_losing_streak = 0
            else:
                if is_zb:
                    trade_result = zb_loss * zb_contracts * zb_position_multiplier
                else:
                    trade_result = yen_loss * yen_contracts * yen_position_multiplier
                all_trades_results.append(False)
                current_losing_streak += 1
                current_winning_streak = 0
            
            max_winning_streak = max(max_winning_streak, current_winning_streak)
            max_losing_streak = max(max_losing_streak, current_losing_streak)
            daily_trades_result.append(trade_result)
        
        daily_pnl = sum(daily_trades_result)
        if daily_pnl < -daily_risk_limit:
            daily_pnl = -daily_risk_limit
        
        capital += daily_pnl
        daily_returns.append(daily_pnl)
        
        if capital >= starting_capital + profit_target or capital <= starting_capital - loss_limit:
            break
    
    return capital, daily_returns, trading_days, max_losing_streak, max_winning_streak, all_trades_results

def run_monte_carlo(
    starting_capital: float,
    profit_target: float,
    loss_limit: float,
    daily_risk_limit: float,
    win_rate: float,
    zb_trades_per_day: int,
    yen_trades_per_day: int,
    zb_win: float,
    zb_loss: float,
    yen_win: float,
    yen_loss: float,
    zb_contracts: int,
    yen_contracts: int,
    num_simulations: int
) -> Tuple[List[float], List[float], List[int], List[int], List[int], List[bool]]:
    
    final_capitals = []
    all_daily_returns = []
    days_to_complete = []
    max_losing_streaks = []
    max_winning_streaks = []
    all_trades = []
    
    simulation_params = {
        'starting_capital': starting_capital,
        'profit_target': profit_target,
        'loss_limit': loss_limit,
        'daily_risk_limit': daily_risk_limit,
        'win_rate': win_rate,
        'zb_trades_per_day': zb_trades_per_day,
        'yen_trades_per_day': yen_trades_per_day,
        'zb_win': zb_win,
        'zb_loss': zb_loss,
        'yen_win': yen_win,
        'yen_loss': yen_loss,
        'zb_contracts': zb_contracts,
        'yen_contracts': yen_contracts
    }
    
    for _ in range(num_simulations):
        final_cap, daily_returns, days, max_lose_streak, max_win_streak, trades = simulate_trading_session(**simulation_params)
        final_capitals.append(final_cap)
        all_daily_returns.extend(daily_returns)
        days_to_complete.append(days)
        max_losing_streaks.append(max_lose_streak)
        max_winning_streaks.append(max_win_streak)
        all_trades.extend(trades)
    
    return final_capitals, all_daily_returns, days_to_complete, max_losing_streaks, max_winning_streaks, all_trades

def calculate_risk_of_ruin(final_capitals: List[float], starting_capital: float, max_loss: float) -> float:
    ruin_level = starting_capital - max_loss
    ruin_count = sum(1 for cap in final_capitals if cap <= ruin_level)
    return ruin_count / len(final_capitals)


def calculate_streak_stats(trades: List[bool]) -> dict:
    current_streak = 0
    current_type = None
    streaks = {'winning': [], 'losing': []}
    
    for trade in trades:
        if current_type is None:
            current_type = trade
            current_streak = 1
        elif trade == current_type:
            current_streak += 1
        else:
            if current_type:
                streaks['winning'].append(current_streak)
            else:
                streaks['losing'].append(current_streak)
            current_type = trade
            current_streak = 1
    
    # Add the last streak
    if current_type is not None:
        if current_type:
            streaks['winning'].append(current_streak)
        else:
            streaks['losing'].append(current_streak)
    
    return streaks

# Title and main app setup
st.title("Trading Strategy Monte Carlo Simulation Dashboard")
st.markdown("Multi-Instrument Trading Strategy Analysis")

# Sidebar configurations
with st.sidebar:
    # Create a container for the clock
    st.markdown(
        """
        <style>
            .clock-container { margin-bottom: 20px; }
            .clock { font-family: monospace; font-size: 14px; }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Create empty container for the clock
    clock_container = st.empty()
    
    # JavaScript to update the clock
    st.components.v1.html(
        """
        <div id="clock" class="clock"></div>
        <script>
            function updateClock() {
                var now = new Date();
                var options = {
                    timeZone: 'America/New_York',
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false
                };
                var timeStr = now.toLocaleString('en-US', options)
                    .replace(',', '')
                    .replace(/(\d+)\/(\d+)\/(\d+)/, '$3-$1-$2');
                document.getElementById('clock').innerHTML = timeStr;
                setTimeout(updateClock, 1000);
            }
            updateClock();
        </script>
        """,
        height=50,
    )
    
    st.markdown("**Current User's Login:** Midoelafreet")
    st.markdown("---")
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

# Contract Specifications section
with st.sidebar:
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
            zb_risk_ticks = st.number_input(
                "ZB Risk (Ticks)", 
                value=3, 
                min_value=1
            )
            zb_profit_ticks = st.number_input(
                "ZB Profit (Ticks)", 
                value=6, 
                min_value=1
            )
            zb_commission = st.number_input(
                "ZB Commission ($ per RT)", 
                value=3.25, 
                step=0.01
            )
            
            # Calculate ZB amounts
            zb_win = (zb_profit_ticks * zb_tick_value * zb_contracts) - (zb_commission * zb_contracts)
            zb_loss = (-zb_risk_ticks * zb_tick_value * zb_contracts) - (zb_commission * zb_contracts)
            
            st.markdown(f"**ZB Win Amount: ${zb_win:.2f}**")
            st.markdown(f"**ZB Loss Amount: ${zb_loss:.2f}**")
        
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
            yen_risk_ticks = st.number_input(
                "Yen Risk (Ticks)", 
                value=3, 
                min_value=1
            )
            yen_profit_ticks = st.number_input(
                "Yen Profit (Ticks)", 
                value=6, 
                min_value=1
            )
            yen_commission = st.number_input(
                "Yen Commission ($ per RT)", 
                value=3.25, 
                step=0.01
            )
            
            # Calculate Yen amounts
            yen_win = (yen_profit_ticks * yen_tick_value * yen_contracts) - (yen_commission * yen_contracts)
            yen_loss = (-yen_risk_ticks * yen_tick_value * yen_contracts) - (yen_commission * yen_contracts)
            
            st.markdown(f"**Yen Win Amount: ${yen_win:.2f}**")
            st.markdown(f"**Yen Loss Amount: ${yen_loss:.2f}**")

# Trading Parameters section
with st.sidebar:
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
    # Initialize session state for tracking daily P&L if it doesn't exist
    if 'current_daily_pnl' not in st.session_state:
        st.session_state.current_daily_pnl = 0.0
    if 'zb_trades_taken' not in st.session_state:
        st.session_state.zb_trades_taken = 0
    if 'yen_trades_taken' not in st.session_state:
        st.session_state.yen_trades_taken = 0

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
        trade_result = st.radio("Trade Result", ["Win", "Loss"])
        
    with trade_col2:
        # Calculate potential P&L
        if market == "ZB":
            potential_pnl = zb_win * num_contracts if trade_result == "Win" else zb_loss * num_contracts
        else:
            potential_pnl = yen_win * num_contracts if trade_result == "Win" else yen_loss * num_contracts
            
        st.metric("Potential P&L", f"${potential_pnl:.2f}")
        
        # Warning messages
        if market == "ZB" and st.session_state.zb_trades_taken >= trades_per_day_zb:
            st.warning("⚠️ Maximum ZB trades for the day reached!")
        elif market == "Yen" and st.session_state.yen_trades_taken >= trades_per_day_yen:
            st.warning("⚠️ Maximum Yen trades for the day reached!")
        
        if st.session_state.current_daily_pnl + potential_pnl < -daily_risk_limit:
            st.warning("⚠️ This trade would exceed your daily risk limit!")

    # Add Trade Button
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
                
            # Add trade to history with current timestamp
            trade_data = {
                'market': market,
                'contracts': num_contracts,
                'result': trade_result,
                'pnl': potential_pnl,
                'timestamp': get_current_timestamp()
            }
            if 'trade_history' not in st.session_state:
                st.session_state.trade_history = []
            st.session_state.trade_history.append(trade_data)
            
            st.success(f"Trade added successfully! New daily P&L: ${st.session_state.current_daily_pnl:.2f}")
            st.rerun()

    # Reset Daily Stats Button
    if st.button("Reset Daily Stats"):
        st.session_state.current_daily_pnl = 0.0
        st.session_state.zb_trades_taken = 0
        st.session_state.yen_trades_taken = 0
        st.session_state.trade_history = []  # Clear trade history
        st.success("Daily stats reset successfully!")
        st.rerun()

    # Risk Analysis
    st.subheader("Risk Analysis")
    remaining_risk = daily_risk_limit + st.session_state.current_daily_pnl
    risk_percentage = (remaining_risk / daily_risk_limit) * 100
    
    # Create a progress bar for remaining risk
    st.progress(min(max(risk_percentage, 0), 100) / 100)
    
    # Additional risk metrics
    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        st.metric("Remaining Risk Amount", f"${max(remaining_risk, 0):.2f}")
        max_zb_trades = int(remaining_risk / abs(zb_loss)) if remaining_risk > 0 else 0
        st.metric("Max Additional ZB Trades Possible", 
                 min(max_zb_trades, trades_per_day_zb - st.session_state.zb_trades_taken))
    
    with risk_col2:
        st.metric("Risk Utilized", f"${min(daily_risk_limit - remaining_risk, daily_risk_limit):.2f}")
        max_yen_trades = int(remaining_risk / abs(yen_loss)) if remaining_risk > 0 else 0
        st.metric("Max Additional Yen Trades Possible", 
                 min(max_yen_trades, trades_per_day_yen - st.session_state.yen_trades_taken))

# PnL Monitoring and Export Section
st.header("Trading Performance Analytics")
with st.expander("Daily Performance & Export", expanded=True):
    # Initialize trade history if it doesn't exist
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
            for market_type in ['ZB', 'Yen']:
                market_trades = [t for t in st.session_state.trade_history if t['market'] == market_type]
                if market_trades:
                    wins = len([t for t in market_trades if t['pnl'] > 0])
                    total = len(market_trades)
                    win_rates[market_type] = (wins / total) * 100
            
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

    # Export Section
    st.subheader("Export Trading Data")
    if st.session_state.trade_history:
        df_export = pd.DataFrame(st.session_state.trade_history)
        current_time = get_current_timestamp()
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
