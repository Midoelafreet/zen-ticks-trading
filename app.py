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

# -------------------------------------------------------------
# 1. --- Session State Initialization ---
# -------------------------------------------------------------
# Make sure these session state variables exist before the app code runs
if 'current_daily_pnl' not in st.session_state:
    st.session_state.current_daily_pnl = 0.0
if 'zb_trades_taken' not in st.session_state:
    st.session_state.zb_trades_taken = 0
if 'yen_trades_taken' not in st.session_state:
    st.session_state.yen_trades_taken = 0
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

# -------------------------------------------------------------
# 2. --- Page and Title Setup ---
# -------------------------------------------------------------
st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")
st.title("Trading Strategy Monte Carlo Simulation Dashboard")
st.markdown("Multi-Instrument Trading Strategy Analysis")

# -------------------------------------------------------------
# 3. --- Helper Functions ---
# -------------------------------------------------------------
def get_current_timestamp():
    """
    Returns a formatted timestamp for New York time.
    """
    ny_tz = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_tz)
    return ny_time.strftime('%Y-%m-%d %H:%M:%S')

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
    """
    Simulates a trading session over multiple days using random outcomes
    based on the specified win rate and trade configurations.
    """
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
            
            # Randomly decide if the trade is a win or loss
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
        
        # Check if we've hit profit or loss limits
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
    """
    Runs multiple simulations (Monte Carlo) using the simulate_trading_session function
    and aggregates the results for further analysis.
    """
    
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
    """
    Calculates the risk of ruin given final capitals from simulations.
    """
    ruin_level = starting_capital - max_loss
    ruin_count = sum(1 for cap in final_capitals if cap <= ruin_level)
    return ruin_count / len(final_capitals)

def calculate_streak_stats(trades: List[bool]) -> dict:
    """
    Calculates winning and losing streaks from a list of Boolean trade results.
    """
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
    
    # Handle the last streak
    if current_type is not None:
        if current_type:
            streaks['winning'].append(current_streak)
        else:
            streaks['losing'].append(current_streak)
    
    return streaks

# -------------------------------------------------------------
# 4. --- Sidebar Configurations ---
# -------------------------------------------------------------
with st.sidebar:
    # Style for the clock
    st.markdown(
        """
        <style>
            .clock { 
                font-family: monospace; 
                font-size: 14px;
                color: white !important;
            }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Display Current Date and Time
    st.components.v1.html(
        """
        <div style="color: white;">
            <strong>Current Date and Time (New York Time):</strong>
            <div id="clock" style="color: white; margin-top: 5px;"></div>
        </div>
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
        height=70,
    )
    
    st.markdown('<p style="color: white;"><strong>Current User\'s Login:</strong> Midoelafreet</p>', unsafe_allow_html=True)
    st.markdown('<hr style="margin: 15px 0;">', unsafe_allow_html=True)
    
    st.header("Trading Parameters")
    
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

    with st.expander("Contract Specifications", expanded=True):
        col1, col2 = st.columns(2)
        
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

# -------------------------------------------------------------
# 5. --- Upper Metrics: Current Daily PnL, Trades, and Risk ---
# -------------------------------------------------------------
st.header("Daily Overview")

col_upper_1, col_upper_2, col_upper_3 = st.columns(3)

with col_upper_1:
    st.metric("Current Daily P&L (Upper)", f"${st.session_state.current_daily_pnl:.2f}")
with col_upper_2:
    st.metric("ZB Trades Taken (Upper)", f"{st.session_state.zb_trades_taken}/{trades_per_day_zb}")
with col_upper_3:
    st.metric("Yen Trades Taken (Upper)", f"{st.session_state.yen_trades_taken}/{trades_per_day_yen}")

st.markdown("---")

# -------------------------------------------------------------
# 6. --- Daily Risk Manager Calculator ---
# -------------------------------------------------------------
st.header("Daily Risk Manager Calculator")

with st.expander("Risk Management Calculator", expanded=True):
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

    # Section to enter a trade result
    st.subheader("Enter Trade Result")

    trade_col1, trade_col2 = st.columns(2)
    with trade_col1:
        market = st.selectbox("Select Market", ["ZB", "Yen"])
        num_contracts = st.number_input("Number of Contracts", min_value=1, value=1)
        trade_result = st.radio("Trade Result", ["Win", "Loss"])
        risk_ticks = st.number_input("Ticks Risked", min_value=1, value=3)
        profit_ticks = st.number_input("Ticks Profited", min_value=1, value=6)

    with trade_col2:
        # Based on market selection, pick the appropriate tick/commission
        if market == "ZB":
            tick_value = zb_tick_value
            commission = zb_commission
        else:
            tick_value = yen_tick_value
            commission = yen_commission
        
        if trade_result == "Win":
            potential_pnl = (profit_ticks * tick_value * num_contracts) - (commission * num_contracts)
        else:
            potential_pnl = - (risk_ticks * tick_value * num_contracts) - (commission * num_contracts)

        st.metric("Potential P&L", f"${potential_pnl:.2f}")

        # Warn if maximum trades are reached or daily risk exceeded
        if market == "ZB" and st.session_state.zb_trades_taken >= trades_per_day_zb:
            st.warning("⚠️ Maximum ZB trades for the day reached!")
        elif market == "Yen" and st.session_state.yen_trades_taken >= trades_per_day_yen:
            st.warning("⚠️ Maximum Yen trades for the day reached!")
        if st.session_state.current_daily_pnl + potential_pnl < -daily_risk_limit:
            st.warning("⚠️ This trade would exceed your daily risk limit!")

    if st.button("Add Trade"):
        # Validate if the trade is allowed
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

    if st.button("Reset Daily Stats"):
        st.session_state.current_daily_pnl = 0.0
        st.session_state.zb_trades_taken = 0
        st.session_state.yen_trades_taken = 0
        st.session_state.trade_history = []
        st.success("Daily stats reset successfully!")

    st.subheader("Risk Analysis")
    remaining_risk = daily_risk_limit + st.session_state.current_daily_pnl
    risk_percentage = (remaining_risk / daily_risk_limit) * 100
    st.progress(min(max(risk_percentage, 0), 100) / 100)

    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        st.metric("Remaining Risk Amount", f"${max(remaining_risk, 0):.2f}")
        if remaining_risk > 0:
            max_zb_trades = int(
                remaining_risk / abs(- (3 * zb_tick_value * 1) - (zb_commission * 1))
            )
        else:
            max_zb_trades = 0
        st.metric("Max Additional ZB Trades Possible", 
                  min(max_zb_trades, trades_per_day_zb - st.session_state.zb_trades_taken))
    with risk_col2:
        st.metric("Risk Utilized", f"${min(daily_risk_limit - remaining_risk, daily_risk_limit):.2f}")
        if remaining_risk > 0:
            max_yen_trades = int(
                remaining_risk / abs(- (3 * yen_tick_value * 1) - (yen_commission * 1))
            )
        else:
            max_yen_trades = 0
        st.metric("Max Additional Yen Trades Possible", 
                  min(max_yen_trades, trades_per_day_yen - st.session_state.yen_trades_taken))

# -------------------------------------------------------------
# 7. --- Trading Performance Analytics ---
# -------------------------------------------------------------
st.header("Trading Performance Analytics")
with st.expander("Daily Performance & Export", expanded=True):
    if st.session_state.trade_history:
        df_trades = pd.DataFrame(st.session_state.trade_history)
        
        # Cumulative P&L Chart
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
            fig_pnl = px.line(
                df_trades, 
                y='cumulative_pnl',
                title='Cumulative PnL Throughout the Day',
                labels={'cumulative_pnl': 'Cumulative PnL ($)', 'index': 'Trade #'}
            )
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        # Win Rate by Market
        with chart_col2:
            win_rates = {}
            for market_type in ['ZB', 'Yen']:
                market_trades = df_trades[df_trades['market'] == market_type]
                if not market_trades.empty:
                    wins = len(market_trades[market_trades['pnl'] > 0])
                    total = len(market_trades)
                    win_rates[market_type] = (wins / total) * 100
            
            if win_rates:
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
        
        st.subheader("Export Trading Data")
        current_time = get_current_timestamp()
        csv_data = df_trades.to_csv(index=False)
        st.download_button(
            label="📥 Download Trading Data (CSV)",
            data=csv_data,
            file_name=f"trading_data_{current_time.split()[0]}.csv",
            mime="text/csv",
            key="download_csv"
        )
        
        st.dataframe(df_trades, use_container_width=True)
    else:
        st.info("No trades to export or analyze yet")

# -------------------------------------------------------------
# 8. --- Monte Carlo Simulation and Visualization ---
# -------------------------------------------------------------
if st.button("Run Monte Carlo Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        # For demonstration, define static win/loss for each contract:
        zb_risk_ticks = 3
        zb_profit_ticks = 6
        yen_risk_ticks = 3
        yen_profit_ticks = 6
        
        zb_win = (zb_profit_ticks * zb_tick_value * zb_contracts) - (zb_commission * zb_contracts)
        zb_loss = - (zb_risk_ticks * zb_tick_value * zb_contracts) - (zb_commission * zb_contracts)
        yen_win = (yen_profit_ticks * yen_tick_value * yen_contracts) - (yen_commission * yen_contracts)
        yen_loss = - (yen_risk_ticks * yen_tick_value * yen_contracts) - (yen_commission * yen_contracts)

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
        
        profitable_sims = sum(1 for cap in final_capitals if cap > starting_capital)
        profit_probability = profitable_sims / num_simulations * 100
        
        avg_days = sum(days_to_complete) / len(days_to_complete)
        max_days = max(days_to_complete)
        min_days = min(days_to_complete)
        
        risk_of_ruin = calculate_risk_of_ruin(final_capitals, starting_capital, loss_limit)
        
        st.header("Simulation Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Profit Probability", f"{profit_probability:.1f}%")
            st.metric("Risk of Ruin", f"{risk_of_ruin*100:.1f}%")
        with col2:
            st.metric("Average Days", f"{avg_days:.1f}")
            st.metric("Maximum Days", f"{max_days}")
        with col3:
            st.metric("Minimum Days", f"{min_days}")
            st.metric("Total Simulations", f"{num_simulations:,}")

        st.session_state.final_capitals = final_capitals
        st.session_state.daily_returns = daily_returns
        st.session_state.days_to_complete = days_to_complete
        st.session_state.max_losing_streaks = max_losing_streaks
        st.session_state.max_winning_streaks = max_winning_streaks
        st.session_state.all_trades_for_streak = all_trades

# -------------------------------------------------------------
# 9. --- Visualization of Simulation Results ---
# -------------------------------------------------------------
if 'final_capitals' in st.session_state:
    final_capitals = st.session_state.final_capitals
    daily_returns = st.session_state.daily_returns
    days_to_complete = st.session_state.days_to_complete
    max_losing_streaks = st.session_state.max_losing_streaks
    max_winning_streaks = st.session_state.max_winning_streaks
    all_trades_for_streak = st.session_state.all_trades_for_streak

    tab1, tab2, tab3, tab4 = st.tabs([
        "Capital Distribution", 
        "Daily Returns", 
        "Trading Streaks", 
        "Detailed Statistics"
    ])

    with tab1:
        st.subheader("Final Capital Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=final_capitals,
            nbinsx=50,
            name="Capital Distribution",
            opacity=0.75
        ))
        
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
        streak_stats = calculate_streak_stats(all_trades_for_streak)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=streak_stats['winning'],
            name="Winning Streaks",
            opacity=0.75,
            marker_color='green'
        ))
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Winning Streak", f"{max(max_winning_streaks)}")
            if streak_stats['winning']:
                avg_win_streak = sum(streak_stats['winning']) / len(streak_stats['winning'])
                st.metric("Avg Winning Streak", f"{avg_win_streak:.2f}")
            else:
                st.metric("Avg Winning Streak", "0")
        with col2:
            st.metric("Max Losing Streak", f"{max(max_losing_streaks)}")
            if streak_stats['losing']:
                avg_lose_streak = sum(streak_stats['losing']) / len(streak_stats['losing'])
                st.metric("Avg Losing Streak", f"{avg_lose_streak:.2f}")
            else:
                st.metric("Avg Losing Streak", "0")

    with tab4:
        st.subheader("Detailed Statistics")
        
        final_capitals_array = np.array(final_capitals)
        daily_returns_array = np.array(daily_returns)
        
        def safe_div(x, y):
            return x / y if y != 0 else 0

        mean_capital = np.mean(final_capitals_array)
        median_capital = np.median(final_capitals_array)
        std_capital = np.std(final_capitals_array)
        skew_capital = stats.skew(final_capitals_array)
        kurt_capital = stats.kurtosis(final_capitals_array)
        
        mean_daily_return = np.mean(daily_returns_array) if len(daily_returns_array) > 0 else 0
        median_daily_return = np.median(daily_returns_array) if len(daily_returns_array) > 0 else 0
        std_daily_return = np.std(daily_returns_array) if len(daily_returns_array) > 0 else 1
        
        # Sharpe Ratio (assuming 0% risk-free)
        sharpe_ratio = safe_div(mean_daily_return, std_daily_return)

        # Profit Factor = sum of positive returns /| sum of negative returns|
        total_positive = sum(r for r in daily_returns_array if r > 0)
        total_negative = sum(r for r in daily_returns_array if r < 0)
        profit_factor = safe_div(total_positive, abs(total_negative)) if total_negative != 0 else 0

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
                "Sharpe Ratio (0% RF)",
                "Profit Factor"
            ],
            "Value": [
                f"${mean_capital:.2f}",
                f"${median_capital:.2f}",
                f"${std_capital:.2f}",
                f"{skew_capital:.3f}",
                f"{kurt_capital:.3f}",
                f"${mean_daily_return:.2f}",
                f"${median_daily_return:.2f}",
                f"${std_daily_return:.2f}",
                f"{sharpe_ratio:.3f}",
                f"{profit_factor:.3f}",
            ]
        }

        stats_df = pd.DataFrame(stats_data)
        st.table(stats_df.set_index('Metric'))
