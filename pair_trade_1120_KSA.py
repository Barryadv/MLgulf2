"""
Pair Trading Strategy: 1120.SR vs KSA
=====================================
A minimal, working backtester for a pairs trading strategy using:
- OLS hedge ratio (train-only)
- Z-score mean reversion signals
- Logistic regression probability filter
- Transaction costs (25 bps per leg)

Author: Auto-generated
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TICKERS = ["1120.SR", "KSA"]
DATA_START = "2016-01-01"       # Raw data fetch start
DATA_END = "2025-11-30"         # Raw data fetch end

TRAIN_START = "2016-07-01"      # After 6-month lookback warmup
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2025-11-30"

COST_BPS_PER_LEG = 25           # 25 bps per direction (50 bps round-trip)
ROLLING_WINDOW = 60             # ~3 months for z-score, vol, corr
LOOKBACK_WARMUP = 126           # ~6 months for feature warmup (NOTE: currently unused)

# Derived constants
TRADING_DAYS_PER_YEAR = 252
BPS_TO_DECIMAL = 10_000
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ZSCORE_ENTRY = 1.0              # Enter when |zscore| > this
ZSCORE_EXIT = 0.2               # Exit when |zscore| < this
PROB_THRESHOLD = 0.55           # ML probability threshold
MAX_HOLD_DAYS = 20              # Max holding period before forced exit

RANDOM_SEED = 42

# Cache settings
CACHE_FILE = "price_cache.csv"
FORCE_REFRESH = False           # Set True to force re-download

# ============================================================================
# FUNCTIONS
# ============================================================================

def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_file: str = CACHE_FILE,
    force_refresh: bool = FORCE_REFRESH,
) -> pd.DataFrame:
    """
    Download adjusted close prices from yfinance with caching.

    Args:
        tickers: List of ticker symbols to download.
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).
        cache_file: Path to CSV cache file.
        force_refresh: If True, bypass cache and re-download.

    Returns:
        DataFrame with tickers as columns and dates as index.

    Side Effects:
        Prints download/cache status messages.
        Writes to cache_file on fresh download.
    """
    cache_path = Path(cache_file)
    
    # Try to load from cache
    if not force_refresh and cache_path.exists():
        print(f"Loading from cache: {cache_file}")
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        
        # Check if cache covers our date range
        cache_end = cached.index.max()
        requested_end = pd.Timestamp(end)
        
        if cache_end >= requested_end - pd.Timedelta(days=5):  # 5-day tolerance
            print(f"  Cache valid: covers up to {cache_end.date()}")
            # Filter to requested range
            mask = (cached.index >= start) & (cached.index <= end)
            return cached[mask].copy()
        else:
            print(f"  Cache stale: ends {cache_end.date()}, need {end}")
    
    # Download fresh data
    print(f"Fetching data for {tickers} from {start} to {end}...")
    
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    
    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data['Adj Close'].copy()
    else:
        # Single ticker case
        adj_close = data[['Adj Close']].copy()
        adj_close.columns = tickers
    
    # Ensure column names match tickers
    if isinstance(adj_close.columns, pd.MultiIndex):
        adj_close.columns = adj_close.columns.get_level_values(0)
    
    # Save to cache
    adj_close.to_csv(cache_path)
    print(f"  Saved to cache: {cache_file}")
    
    print(f"  Raw data shape: {adj_close.shape}")
    print(f"  Date range: {adj_close.index.min().date()} to {adj_close.index.max().date()}")
    
    return adj_close


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill gaps in price data and report fill percentages.

    Args:
        df: DataFrame with price columns (already aligned by yfinance inner join).

    Returns:
        DataFrame with gaps forward-filled and leading NaNs dropped.

    Raises:
        ValueError: If input DataFrame is empty.

    Side Effects:
        Prints fill percentages and final shape.
    """
    if df.empty:
        raise ValueError("prepare_data received empty DataFrame")

    df = df.copy()

    # Count NaNs before fill
    nan_before = df.isna().sum()
    total_rows = len(df)
    
    # Forward-fill remaining gaps
    df = df.ffill()
    
    # Drop any leading NaNs that couldn't be filled
    df = df.dropna()
    
    # Report fill percentages
    print("\nData preparation:")
    for col in df.columns:
        filled_count = nan_before[col]
        fill_pct = (filled_count / total_rows) * 100 if total_rows > 0 else 0
        print(f"  {col}: {fill_pct:.2f}% values forward-filled ({filled_count}/{total_rows})")
    
    print(f"  Final shape after cleaning: {df.shape}")
    print(f"  Final date range: {df.index.min().date()} to {df.index.max().date()}")
    
    return df


def fit_hedge_ratio(
    train_df: pd.DataFrame,
    ticker_y: str = TICKERS[0],
    ticker_x: str = TICKERS[1],
) -> float:
    """
    Fit OLS hedge ratio on TRAIN data only.

    Model: log(ticker_y) = alpha + beta * log(ticker_x)

    Args:
        train_df: Training DataFrame with price columns.
        ticker_y: Dependent variable ticker (default: first in TICKERS).
        ticker_x: Independent variable ticker (default: second in TICKERS).

    Returns:
        Beta (hedge ratio) coefficient.

    Raises:
        ValueError: If train_df is empty.

    Side Effects:
        Prints fitted alpha, beta, and R-squared.
    """
    if len(train_df) == 0:
        raise ValueError("fit_hedge_ratio received empty training data")

    y = np.log(train_df[ticker_y])
    x = np.log(train_df[ticker_x])
    x_with_intercept = sm.add_constant(x)

    model = sm.OLS(y, x_with_intercept).fit()
    
    alpha = model.params.iloc[0]
    beta = model.params.iloc[1]
    
    print(f"\nHedge ratio (OLS on train):")
    print(f"  alpha = {alpha:.6f}")
    print(f"  beta  = {beta:.6f}")
    print(f"  R-squared = {model.rsquared:.4f}")
    
    return beta


def build_features_and_labels(
    df: pd.DataFrame,
    beta: float,
    ticker_y: str = TICKERS[0],
    ticker_x: str = TICKERS[1],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build features and labels for ML model.

    All rolling features use only data <= t (no look-ahead).
    Label is shifted forward; last row is dropped.

    Args:
        df: DataFrame with price columns.
        beta: Hedge ratio from fit_hedge_ratio().
        ticker_y: Dependent variable ticker.
        ticker_x: Independent variable ticker.

    Returns:
        Tuple of (DataFrame with features/labels, list of feature column names).

    Side Effects:
        Prints feature columns and row count.
    """
    df = df.copy()
    
    # Log prices
    log_y = np.log(df[ticker_y])
    log_x = np.log(df[ticker_x])
    
    # Spread using hedge ratio (beta fit on train, applied to all)
    df['spread'] = log_y - beta * log_x
    
    # Returns for correlation calculation
    df['ret_y'] = df[ticker_y].pct_change()
    df['ret_x'] = df[ticker_x].pct_change()
    
    # --- FEATURES (rolling, no future data) ---
    
    # Z-score of spread
    spread_mean = df['spread'].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
    spread_std = df['spread'].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
    df['zscore'] = (df['spread'] - spread_mean) / spread_std
    
    # Spread changes
    df['spread_chg_1d'] = df['spread'].diff(1)
    df['spread_chg_5d'] = df['spread'].diff(5)
    
    # Rolling correlation of returns
    df['roll_corr'] = df['ret_y'].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).corr(df['ret_x'])
    
    # Rolling volatility of spread
    df['roll_vol'] = df['spread'].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std()
    
    # --- LABEL (predict next-day spread reversion) ---
    # y_t = 1 if spread_{t+1} < spread_t (spread reverts down), else 0
    # This means: if label=1, spread is expected to decrease
    df['label'] = (df['spread'].shift(-1) < df['spread']).astype(float)
    
    # Drop last row (no label available)
    df = df.iloc[:-1]
    
    # Feature columns
    feature_cols = ['zscore', 'spread_chg_1d', 'spread_chg_5d', 'roll_corr', 'roll_vol']
    
    # Drop rows with NaN in features or label
    df = df.dropna(subset=feature_cols + ['label'])
    
    print(f"\nFeatures built:")
    print(f"  Feature columns: {feature_cols}")
    print(f"  Rows after dropping NaN: {len(df)}")
    
    return df, feature_cols


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[StandardScaler, LogisticRegression]:
    """
    Fit StandardScaler and LogisticRegression on TRAIN data only.

    Args:
        X_train: Feature DataFrame (train split only).
        y_train: Label Series (train split only).

    Returns:
        Tuple of (fitted StandardScaler, fitted LogisticRegression).

    Raises:
        ValueError: If X_train is empty.

    Side Effects:
        Prints train accuracy and model coefficients.
    """
    if len(X_train) == 0:
        raise ValueError("fit_model received empty training data")

    np.random.seed(RANDOM_SEED)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    model.fit(X_scaled, y_train)
    
    train_acc = model.score(X_scaled, y_train)
    print(f"\nModel fitted (LogisticRegression):")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Coefficients: {dict(zip(X_train.columns, model.coef_[0].round(4)))}")
    
    return scaler, model


def backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    model: LogisticRegression,
    beta: float,  # NOTE: kept for API consistency; not used in current logic
    cost_bps: int = COST_BPS_PER_LEG,
    max_hold: int = MAX_HOLD_DAYS,
) -> pd.DataFrame:
    """
    Backtest the trading strategy.

    Trading rules:
        - If prob > PROB_THRESHOLD and zscore > ZSCORE_ENTRY: SHORT spread
        - If prob > PROB_THRESHOLD and zscore < -ZSCORE_ENTRY: LONG spread
        - Exit when |zscore| < ZSCORE_EXIT or held > max_hold days

    Position encoding:
        +1 = long spread (long Y, short X*beta)
        -1 = short spread (short Y, long X*beta)
         0 = flat

    Args:
        df: DataFrame with features and spread data.
        feature_cols: List of feature column names.
        scaler: Fitted StandardScaler.
        model: Fitted LogisticRegression.
        beta: Hedge ratio (kept for API; not used in current logic).
        cost_bps: Transaction cost in basis points per leg.
        max_hold: Maximum holding period in days.

    Returns:
        DataFrame with positions, returns, costs, and equity columns.
    """
    df = df.copy()
    
    # Get predictions
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    df['prob'] = model.predict_proba(X_scaled)[:, 1]  # P(label=1) = P(spread decreases)
    
    # Spread returns (for position P&L)
    # If long spread: profit when spread increases
    # spread_return = d(spread) ≈ ret_y - beta * ret_x (approx for small moves)
    df['spread_ret'] = df['spread'].diff()
    
    # Initialize
    positions = np.zeros(len(df))
    hold_days = np.zeros(len(df))
    
    current_pos = 0
    current_hold = 0
    
    for i in range(len(df)):
        zscore = df['zscore'].iloc[i]
        prob = df['prob'].iloc[i]
        
        # Exit conditions
        if current_pos != 0:
            current_hold += 1
            exit_signal = (abs(zscore) < ZSCORE_EXIT) or (current_hold >= max_hold)
            if exit_signal:
                current_pos = 0
                current_hold = 0
        
        # Entry conditions (only if flat)
        if current_pos == 0:
            if prob > PROB_THRESHOLD and zscore > ZSCORE_ENTRY:
                # Expect spread to decrease (label=1 more likely), go SHORT spread
                current_pos = -1
                current_hold = 0
            elif prob > PROB_THRESHOLD and zscore < -ZSCORE_ENTRY:
                # Expect spread to increase (label=0 more likely), go LONG spread
                current_pos = 1
                current_hold = 0
        
        positions[i] = current_pos
        hold_days[i] = current_hold
    
    df['position'] = positions
    df['hold_days'] = hold_days
    
    # Position changes (for transaction costs)
    df['pos_change'] = df['position'].diff().fillna(0)
    
    # Transaction costs: 25 bps per leg
    # Entry from flat: 1 leg cost (we apply cost_bps)
    # Exit to flat: 1 leg cost
    # Flip: 2 leg costs
    cost_rate = cost_bps / BPS_TO_DECIMAL
    
    df['turnover'] = df['pos_change'].abs()
    # Flip from +1 to -1 or vice versa = change of 2, costs = 2 legs
    # Entry/exit = change of 1, costs = 1 leg
    df['cost'] = df['turnover'] * cost_rate
    
    # Strategy returns: position * spread_return - costs
    # Shift position by 1 because position at t earns return from t to t+1
    df['strat_ret'] = df['position'].shift(1).fillna(0) * df['spread_ret'] - df['cost']
    
    # Cumulative returns
    df['cum_ret'] = df['strat_ret'].cumsum()
    df['equity'] = 1 + df['cum_ret']
    
    # Track trade-level info
    df['trade_id'] = (df['position'] != df['position'].shift()).cumsum()
    df.loc[df['position'] == 0, 'trade_id'] = 0  # Flat periods get trade_id=0
    
    return df


def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract individual trades with entry/exit dates and P&L.

    Args:
        df: Backtest DataFrame with trade_id, position, spread_ret, cost columns.

    Returns:
        DataFrame with one row per trade containing entry/exit dates,
        direction, hold_days, gross_pnl, costs, and net_pnl.
    """
    trades: list[dict[str, Any]] = []

    # Get unique trade IDs (excluding 0 which is flat)
    trade_ids = df[df['trade_id'] > 0]['trade_id'].unique()

    for trade_id_val in trade_ids:
        trade_data = df[df['trade_id'] == trade_id_val]
        
        entry_date = trade_data.index[0]
        exit_date = trade_data.index[-1]
        direction = trade_data['position'].iloc[0]
        hold_days = len(trade_data)
        
        # Gross P&L (before costs)
        gross_pnl = (trade_data['position'].shift(1).fillna(0) * trade_data['spread_ret']).sum()
        
        # Costs for this trade
        costs = trade_data['cost'].sum()
        
        # Net P&L
        net_pnl = gross_pnl - costs
        
        trades.append({
            'trade_id': trade_id_val,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'hold_days': hold_days,
            'gross_pnl': gross_pnl,
            'costs': costs,
            'net_pnl': net_pnl
        })
    
    return pd.DataFrame(trades)


def show_current_position_and_recent_trades(
    df: pd.DataFrame,
    n_recent: int = 10,
) -> pd.DataFrame:
    """
    Display current position and N most recent trades with P&L.

    Args:
        df: Concatenated backtest DataFrame (train + test).
        n_recent: Number of recent trades to display.

    Returns:
        DataFrame of all trades (for further analysis).

    Side Effects:
        Prints current position and recent trade summary.
    """
    df = df.copy()
    
    # Recalculate trade_id on the full concatenated data
    df['trade_id'] = (df['position'] != df['position'].shift()).cumsum()
    df.loc[df['position'] == 0, 'trade_id'] = 0
    
    print("\n" + "="*60)
    print("CURRENT POSITION & RECENT TRADES")
    print("="*60)
    
    # Current position (last row)
    last_row = df.iloc[-1]
    current_date = df.index[-1]
    
    pos_map = {1: "LONG SPREAD (long 1120.SR, short KSA)", 
               -1: "SHORT SPREAD (short 1120.SR, long KSA)", 
               0: "FLAT (no position)"}
    
    print(f"\nAs of {current_date.date()}:")
    print(f"  Position: {pos_map[int(last_row['position'])]}")
    if last_row['position'] != 0:
        print(f"  Days held: {int(last_row['hold_days'])}")
        print(f"  Current z-score: {last_row['zscore']:.3f}")
    
    # Extract and show recent trades
    trades_df = extract_trades(df)
    
    if len(trades_df) == 0:
        print("\n  No trades executed.")
        return trades_df
    
    print(f"\n{n_recent} Most Recent Trades:")
    print("-"*80)
    
    recent = trades_df.tail(n_recent).iloc[::-1]  # Most recent first
    
    for _, trade in recent.iterrows():
        print(f"  {trade['entry_date'].date()} -> {trade['exit_date'].date()} | "
              f"{trade['direction']:5s} | {trade['hold_days']:2d}d | "
              f"Gross: {trade['gross_pnl']*100:+6.2f}% | "
              f"Costs: {trade['costs']*100:5.2f}% | "
              f"Net: {trade['net_pnl']*100:+6.2f}%")
    
    print("-"*80)
    print(f"  Total trades: {len(trades_df)}")
    print(f"  Total gross P&L: {trades_df['gross_pnl'].sum()*100:+.2f}%")
    print(f"  Total costs: {trades_df['costs'].sum()*100:.2f}%")
    print(f"  Total net P&L: {trades_df['net_pnl'].sum()*100:+.2f}%")
    
    return trades_df


def performance_report(df: pd.DataFrame, label: str = "Test") -> dict[str, Any]:
    """
    Calculate and print performance metrics.

    Args:
        df: Backtest DataFrame with strat_ret, position, cost columns.
        label: Label for the report (e.g., "TRAIN", "TEST").

    Returns:
        Dict with keys: period, start_date, end_date, total_return, ann_return,
        ann_vol, sharpe, max_drawdown, n_trades, win_rate, avg_hold_days, total_costs.

    Side Effects:
        Prints formatted performance report.
    """
    returns = df['strat_ret'].dropna()

    # Basic stats
    total_ret = returns.sum()
    ann_ret = returns.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # Max drawdown
    cum_ret = returns.cumsum()
    running_max = cum_ret.cummax()
    drawdown = cum_ret - running_max
    max_dd = drawdown.min()
    
    # Trade analysis
    positions = df['position']
    pos_changes = positions.diff().fillna(0)
    
    # Count trades (entries)
    entries = (pos_changes != 0) & (positions != 0)
    n_trades = entries.sum()
    
    # Win rate: count periods where position * spread_ret > 0
    # Group by trade (each contiguous non-zero position block)
    trade_id = (positions != positions.shift()).cumsum()
    trade_id[positions == 0] = 0
    
    if n_trades > 0:
        # For each trade, sum the returns
        trade_returns = df.groupby(trade_id)['strat_ret'].sum()
        trade_returns = trade_returns[trade_returns.index != 0]  # Exclude flat periods
        winning_trades = (trade_returns > 0).sum()
        win_rate = winning_trades / len(trade_returns) if len(trade_returns) > 0 else 0
        
        # Average holding period
        trade_lengths = df[positions != 0].groupby(trade_id[positions != 0]).size()
        avg_hold = trade_lengths.mean() if len(trade_lengths) > 0 else 0
    else:
        win_rate = 0
        avg_hold = 0
    
    # Total costs incurred
    total_costs = df['cost'].sum()
    
    metrics = {
        'period': label,
        'start_date': df.index.min().date(),
        'end_date': df.index.max().date(),
        'total_return': total_ret,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': int(n_trades),
        'win_rate': win_rate,
        'avg_hold_days': avg_hold,
        'total_costs': total_costs
    }
    
    print(f"\n{'='*50}")
    print(f"PERFORMANCE REPORT: {label}")
    print(f"{'='*50}")
    print(f"  Date range:      {metrics['start_date']} to {metrics['end_date']}")
    print(f"  Total return:    {metrics['total_return']*100:.2f}%")
    print(f"  Ann. return:     {metrics['ann_return']*100:.2f}%")
    print(f"  Ann. volatility: {metrics['ann_vol']*100:.2f}%")
    print(f"  Sharpe ratio:    {metrics['sharpe']:.3f}")
    print(f"  Max drawdown:    {metrics['max_drawdown']*100:.2f}%")
    print(f"  # Trades:        {metrics['n_trades']}")
    print(f"  Win rate:        {metrics['win_rate']*100:.1f}%")
    print(f"  Avg hold (days): {metrics['avg_hold_days']:.1f}")
    print(f"  Total costs:     {metrics['total_costs']*100:.2f}%")
    
    return metrics


def plot_monthly_heatmap(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str = "Monthly Returns",
) -> pd.DataFrame:
    """
    Plot monthly returns as a heatmap (year x month).

    Args:
        df: Backtest DataFrame with strat_ret column.
        ax: Matplotlib axes to plot on.
        title: Plot title.

    Returns:
        Pivot table of monthly returns (year x month).
    """
    # Resample to monthly returns
    monthly_ret = df['strat_ret'].resample('M').sum()

    # Create pivot table: year x month
    monthly_df = pd.DataFrame({
        'year': monthly_ret.index.year,
        'month': monthly_ret.index.month,
        'return': monthly_ret.values
    })

    pivot = monthly_df.pivot(index='year', columns='month', values='return')
    pivot.columns = MONTH_NAMES
    
    # Create heatmap
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.15)

    ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect='auto')
    
    # Labels
    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                color = 'white' if abs(val) > 0.05 else 'black'
                ax.text(j, i, f'{val*100:.1f}', ha='center', va='center', 
                       fontsize=7, color=color)
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    return pivot


def plot_annual_returns(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str = "Annual Returns",
) -> None:
    """
    Plot annual returns as a bar chart.

    Args:
        df: Backtest DataFrame with strat_ret column.
        ax: Matplotlib axes to plot on.
        title: Plot title.
    """
    annual_ret = df['strat_ret'].resample('Y').sum()
    years = annual_ret.index.year
    
    colors = ['green' if r > 0 else 'red' for r in annual_ret.values]
    
    bars = ax.bar(years, annual_ret.values * 100, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Return (%)')
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, annual_ret.values):
        ypos = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val*100:.1f}%', 
               ha='center', va='bottom' if ypos >= 0 else 'top', fontsize=8)
    
    ax.grid(axis='y', alpha=0.3)


def plot_return_histogram(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str = "Return Distribution",
) -> None:
    """
    Plot histogram of daily returns.

    Args:
        df: Backtest DataFrame with strat_ret column.
        ax: Matplotlib axes to plot on.
        title: Plot title.
    """
    returns = df['strat_ret'].dropna() * 100  # Convert to percentage
    
    ax.hist(returns, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=returns.mean(), color='red', linewidth=2, linestyle='--', 
               label=f'Mean: {returns.mean():.2f}%')
    
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_drawdown(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str = "Drawdown",
) -> None:
    """
    Plot drawdown over time.

    Args:
        df: Backtest DataFrame with strat_ret column.
        ax: Matplotlib axes to plot on.
        title: Plot title.
    """
    cum_ret = df['strat_ret'].cumsum()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) * 100
    
    ax.fill_between(df.index, drawdown, 0, color='red', alpha=0.3)
    ax.plot(df.index, drawdown, color='darkred', linewidth=0.8)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Annotate max drawdown
    min_dd_idx = drawdown.idxmin()
    min_dd = drawdown.min()
    ax.annotate(f'Max DD: {min_dd:.1f}%', xy=(min_dd_idx, min_dd),
                xytext=(10, -20), textcoords='offset points', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.5))


def plot_equity_curve(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics_test: dict[str, Any],
) -> None:
    """
    Plot equity curves for train and test periods.

    Args:
        train_df: Backtest DataFrame for train period.
        test_df: Backtest DataFrame for test period.
        metrics_test: Performance metrics dict for test period.

    Side Effects:
        Saves plot to equity_curve.png and displays it.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Train period
    ax1 = axes[0]
    ax1.plot(train_df.index, train_df['equity'], 'b-', linewidth=1)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Equity Curve - TRAIN Period', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Equity')
    ax1.grid(True, alpha=0.3)
    
    # Test period
    ax2 = axes[1]
    ax2.plot(test_df.index, test_df['equity'], 'g-', linewidth=1)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title(f"Equity Curve - TEST Period (Sharpe: {metrics_test['sharpe']:.2f}, MaxDD: {metrics_test['max_drawdown']*100:.1f}%)", 
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('Equity')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('equity_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nEquity curve saved to equity_curve.png")


def plot_detailed_analytics(df: pd.DataFrame, period_label: str = "Full") -> None:
    """
    Plot comprehensive analytics: heatmap, annual returns, histogram, drawdown.

    Args:
        df: Backtest DataFrame with strat_ret and equity columns.
        period_label: Label for the period (used in titles and filename).

    Side Effects:
        Saves plot to analytics_{period_label}.png and displays it.
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.25)
    
    # Monthly heatmap (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_monthly_heatmap(df, ax1, title=f"Monthly Returns Heatmap ({period_label})")
    
    # Annual returns (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_annual_returns(df, ax2, title=f"Annual Returns ({period_label})")
    
    # Equity curve (middle, spanning both columns)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(df.index, df['equity'], 'steelblue', linewidth=1)
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax3.fill_between(df.index, 1, df['equity'], alpha=0.2, 
                     color='green' if df['equity'].iloc[-1] > 1 else 'red')
    ax3.set_title(f'Equity Curve ({period_label})', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Equity')
    ax3.grid(alpha=0.3)
    
    # Return histogram (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    plot_return_histogram(df, ax4, title=f"Daily Return Distribution ({period_label})")
    
    # Drawdown (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    plot_drawdown(df, ax5, title=f"Drawdown ({period_label})")
    
    plt.savefig(f'analytics_{period_label.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nAnalytics saved to analytics_{period_label.lower().replace(' ', '_')}.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("PAIR TRADING BACKTEST: 1120.SR vs KSA")
    print("=" * 60)
    
    # --- Step 1: Fetch data (with caching) ---
    prices = fetch_prices(TICKERS, DATA_START, DATA_END)
    
    # --- Step 2: Prepare data ---
    prices = prepare_data(prices)
    
    # --- Step 3: Split into train/test by DATE ---
    train_mask = (prices.index >= TRAIN_START) & (prices.index <= TRAIN_END)
    test_mask = (prices.index >= TEST_START) & (prices.index <= TEST_END)
    
    prices_train = prices[train_mask].copy()
    prices_test = prices[test_mask].copy()
    
    print(f"\nTrain/Test split:")
    print(f"  Train: {prices_train.index.min().date()} to {prices_train.index.max().date()} ({len(prices_train)} rows)")
    print(f"  Test:  {prices_test.index.min().date()} to {prices_test.index.max().date()} ({len(prices_test)} rows)")
    
    # --- Step 4: Fit hedge ratio on TRAIN ONLY ---
    beta = fit_hedge_ratio(prices_train)
    
    # --- Step 5: Build features on FULL data (but use train beta) ---
    # We need to build features on full data first, then split
    full_data, feature_cols = build_features_and_labels(prices, beta)
    
    # Re-split after feature building
    train_data = full_data[(full_data.index >= TRAIN_START) & (full_data.index <= TRAIN_END)].copy()
    test_data = full_data[(full_data.index >= TEST_START) & (full_data.index <= TEST_END)].copy()
    
    print(f"\nAfter feature building:")
    print(f"  Train rows: {len(train_data)}")
    print(f"  Test rows:  {len(test_data)}")
    
    # --- Step 6: Fit model on TRAIN ONLY ---
    X_train = train_data[feature_cols]
    y_train = train_data['label']
    
    scaler, model = fit_model(X_train, y_train)
    
    # --- Step 7: Backtest on TRAIN (for reference) ---
    print("\n" + "-"*50)
    print("BACKTESTING...")
    print("-"*50)
    
    train_bt = backtest(train_data, feature_cols, scaler, model, beta)
    metrics_train = performance_report(train_bt, label="TRAIN (in-sample)")
    
    # --- Step 8: Backtest on TEST (out-of-sample) ---
    test_bt = backtest(test_data, feature_cols, scaler, model, beta)
    metrics_test = performance_report(test_bt, label="TEST (out-of-sample)")
    
    # --- Step 9: Current position and recent trades ---
    all_bt = pd.concat([train_bt, test_bt])
    trades_df = show_current_position_and_recent_trades(all_bt, n_recent=10)
    
    # --- Step 10: Plots ---
    plot_equity_curve(train_bt, test_bt, metrics_test)
    
    # Detailed analytics for test period
    plot_detailed_analytics(test_bt, period_label="TEST OOS")
    
    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Hedge ratio (beta): {beta:.4f}")
    print(f"Features: {feature_cols}")
    print(f"Model: LogisticRegression with StandardScaler")
    print(f"Costs: {COST_BPS_PER_LEG} bps per leg")
    print(f"\nOut-of-sample Sharpe: {metrics_test['sharpe']:.3f}")


# ============================================================================
# RISK NOTES
# ============================================================================
"""
RISK NOTES - Double-check these items:

1. TIMING LEAKAGE:
   - Label uses shift(-1) which correctly looks at FUTURE spread value
   - Verify: df['label'] = (df['spread'].shift(-1) < df['spread']) means we're 
     predicting if tomorrow's spread < today's spread
   - Rolling features use default pandas rolling which only uses past data

2. FORWARD-FILL DISTORTION:
   - If fill % is high (>5%), the correlation structure may be distorted
   - Inner join on dates should minimize this for most liquid instruments
   - Check printed fill percentages in output

3. MISSING DATA:
   - 1120.SR is a Saudi stock (different trading calendar than US)
   - KSA is a US-listed ETF
   - Significant calendar mismatch may cause data loss after inner join
   - Consider: if final dataset is <500 rows, results may not be reliable

4. HEDGE RATIO STABILITY:
   - Beta is estimated once on train data and applied unchanged to test
   - In production, consider rolling recalibration (e.g., expanding window)
   - If beta drifts significantly, strategy may underperform

5. TRANSACTION COST ASSUMPTIONS:
   - 25 bps per leg is reasonable for institutional trading
   - Retail costs may be higher; verify with actual broker fees
   - Does not include slippage or market impact

6. SMALL SAMPLE CONCERNS:
   - Test period is ~2 years; may have few trades
   - Sharpe ratio confidence intervals are wide with few observations
   - Win rate with <30 trades is not statistically meaningful
"""

# ============================================================================
# SELF-AUDIT CHECKLIST
# ============================================================================
"""
SELF-AUDIT CHECKLIST - Confirming no look-ahead bias:

[✓] Hedge ratio fit on TRAIN ONLY
    - Line: beta = fit_hedge_ratio(prices_train)
    - prices_train is filtered to TRAIN_START:TRAIN_END before fitting

[✓] StandardScaler fit on TRAIN ONLY  
    - Line: scaler.fit_transform(X_train) in fit_model()
    - X_train is from train_data only

[✓] LogisticRegression fit on TRAIN ONLY
    - Line: model.fit(X_scaled, y_train) in fit_model()
    - Only train data used

[✓] Rolling features use only data <= t
    - pandas .rolling() by default uses past window only
    - No closed='right' or center=True that would include future

[✓] Label shift is FORWARD (shift(-1)), not backward
    - Line: df['label'] = (df['spread'].shift(-1) < df['spread'])
    - shift(-1) means we look at t+1 to create label for t
    - This is correct: we predict future movement

[✓] Last row dropped before training
    - Line: df = df.iloc[:-1] in build_features_and_labels()
    - Last row has NaN label (no t+1 available)

[✓] Train/test split is by TIME, not random
    - Lines: train_mask = (prices.index >= TRAIN_START) & (prices.index <= TRAIN_END)
    - No shuffle; strictly temporal split

[✓] Transaction costs applied on position changes
    - Line: df['cost'] = df['turnover'] * cost_rate
    - Turnover = abs(position change)
    - Applied same day as position change

[✓] Position at t earns return from t to t+1
    - Line: df['strat_ret'] = df['position'].shift(1) * df['spread_ret'] - df['cost']
    - Position is shifted by 1 so today's position earns tomorrow's return
    - This is correct: signal at t, hold overnight, realize return at t+1

[✓] Cache does not cause data leakage
    - Cache stores raw prices only, no derived features
    - Features are always recomputed from raw data
"""
