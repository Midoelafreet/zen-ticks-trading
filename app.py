import streamlit as st
from typing import List

# Function to calculate monetary values based on ticks
def calculate_trade_values(ticks, tick_value, contracts):
    return ticks * tick_value * contracts

# Function to calculate risk of ruin
def calculate_risk_of_ruin(final_capitals: List[float], starting_capital: float, max_loss: float) -> float:
    ruin_level = starting_capital - max_loss
    ruin_count = sum(1 for cap in final_capitals if cap <= ruin_level)
    return ruin_count / len(final_capitals) * 100

# Dummy function to simulate running Monte Carlo simulation
# Replace this with your actual implementation
def run_monte_carlo(starting_capital, profit_target, loss_limit, daily_risk_limit, win_rate, 
                    zb_trades_per_day, yen_trades_per_day, zb_win, zb_loss, yen_win, yen_loss, 
                    zb_contracts, yen_contracts, num_simulations):
    # Placeholder for actual simulation logic
    final_capitals = [starting_capital + (i % 2) * profit_target - (i % 3) * loss_limit for i in range(num_simulations)]
    daily_returns = [0.01] * num_simulations
    days_to_complete = [10] * num_simulations
    max_losing_streaks = [5] * num_simulations
    max_winning_streaks = [7] * num_simulations
    all_trades = [10] * num_simulations
    return final_capitals, daily_returns, days_to_complete, max_losing_streaks, max_winning_streaks, all_trades

# Sidebar configurations
with st.sidebar:
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

    # Risk Management Inputs
    st.header("Risk Management")
    
    # For ZB Futures
    st.subheader("ZB Futures")
    zb_win_ticks = st.number_input("ZB Win Ticks", min_value=1, help="Number of ticks for winning ZB trades")
    zb_loss_ticks = st.number_input("ZB Loss Ticks", min_value=1, help="Number of ticks for losing ZB trades")
    
    # For Yen Futures
    st.subheader("Yen Futures")
    yen_win_ticks = st.number_input("Yen Win Ticks", min_value=1, help="Number of ticks for winning Yen trades")
    yen_loss_ticks = st.number_input("Yen Loss Ticks", min_value=1, help="Number of ticks for losing Yen trades")

    # Calculate monetary values based on ticks
    zb_tick_value = 31.25  # Dollar value per tick for ZB
    yen_tick_value = 12.50  # Dollar value per tick for Yen
    zb_contracts = st.number_input("ZB Contracts", min_value=1, value=1, help="Number of ZB contracts")
    yen_contracts = st.number_input("Yen Contracts", min_value=1, value=1, help="Number of Yen contracts")
    
    # Calculate win/loss amounts
    zb_win = calculate_trade_values(zb_win_ticks, zb_tick_value, zb_contracts)
    zb_loss = calculate_trade_values(zb_loss_ticks, zb_tick_value, zb_contracts)
    yen_win = calculate_trade_values(yen_win_ticks, yen_tick_value, yen_contracts)
    yen_loss = calculate_trade_values(yen_loss_ticks, yen_tick_value, yen_contracts)

    # Display calculated values
    st.write(f"ZB Win Amount: ${zb_win:.2f}")
    st.write(f"ZB Loss Amount: ${zb_loss:.2f}")
    st.write(f"Yen Win Amount: ${yen_win:.2f}")
    st.write(f"Yen Loss Amount: ${yen_loss:.2f}")

# Main content area - Simulation execution
starting_capital = 50000.0  # Example starting capital
profit_target = 10000.0     # Example profit target
loss_limit = 2000.0         # Example loss limit per trade
daily_risk_limit = 5000.0   # Example daily risk limit
win_rate = 0.6              # Example win rate
trades_per_day_zb = 5       # Example ZB trades per day
trades_per_day_yen = 5      # Example Yen trades per day
num_simulations = 1000      # Example number of simulations

if st.button("Run Monte Carlo Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        # Run the simulation with dynamic tick values
        final_capitals, daily_returns, days_to_complete, \
        max_losing_streaks, max_winning_streaks, all_trades = run_monte_carlo(
            starting_capital=starting_capital,
            profit_target=profit_target,
            loss_limit=loss_limit,
            daily_risk_limit=daily_risk_limit,
            win_rate=win_rate,
            zb_trades_per_day=trades_per_day_zb,
            yen_trades_per_day=trades_per_day_yen,
            zb_win=zb_win,        # Now using calculated values based on ticks
            zb_loss=zb_loss,      # Now using calculated values based on ticks
            yen_win=yen_win,      # Now using calculated values based on ticks
            yen_loss=yen_loss,    # Now using calculated values based on ticks
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
