# main.py
import os
import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timezone

# --- 配置项 ---
# 策略起始日期，所有在此之前的交易将被忽略
START_DATE_STR = '2025-10-16'
# 数据存储文件
DATA_FILE_PATH = 'data/daily_pnl.csv'
# 假设的初始资金，用于计算收益率和绘制资金曲线
INITIAL_CAPITAL = 2000

def get_exchange_client():
    """从环境变量安全地获取 API 密钥并初始化 ccxt 客户端"""
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        raise ValueError("API Key 或 Secret 未在环境变量中设置！")

    # 我们假设是U本位合约账户。如果是其他类型，请修改这里
    # 例如：'binance' (现货), 'binancecoinm' (币本位)
    exchange = ccxt.binanceusdm({
        'apiKey': api_key,
        'secret': api_secret,
        'options': {
            'defaultType': 'future',
        },
    })
    return exchange

def fetch_new_trades(exchange, since_timestamp):
    """获取从 'since_timestamp' 以来的所有新交易"""
    all_trades = []
    limit = 200 # API 单次请求限制
    
    # 注意：fetchMyTrades 返回的是平仓成交记录，其中包含盈亏
    trades = exchange.fetch_my_trades(since=since_timestamp, limit=limit)
    
    while trades:
        all_trades.extend(trades)
        # 获取最后一条交易的时间戳，用于下一次分页请求
        since_timestamp = trades[-1]['timestamp'] + 1
        trades = exchange.fetch_my_trades(since=since_timestamp, limit=limit)
        
    print(f"成功获取 {len(all_trades)} 条新交易记录。")
    return all_trades

def process_trades_to_daily_pnl(trades):
    """将交易列表处理成每日盈亏 DataFrame"""
    if not trades:
        return pd.DataFrame()

    pnl_data = []
    for trade in trades:
        # ccxt 统一了数据结构，'info' 字段包含交易所原始数据
        # 对于币安U本位合约，已实现盈亏在 'realizedPnl' 字段
        if 'realizedPnl' in trade['info'] and float(trade['info']['realizedPnl']) != 0:
            pnl_data.append({
                'timestamp': trade['timestamp'],
                'pnl': float(trade['info']['realizedPnl']),
            })
    
    if not pnl_data:
        return pd.DataFrame()

    df = pd.DataFrame(pnl_data)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # 按天聚合，计算每日总盈亏
    daily_pnl = df.groupby(df['datetime'].dt.date)['pnl'].sum().reset_index()
    daily_pnl.rename(columns={'datetime': 'date'}, inplace=True)
    
    return daily_pnl

def update_data_file(new_daily_pnl):
    """读取旧数据，合并新数据，去重并保存"""
    if os.path.exists(DATA_FILE_PATH):
        old_df = pd.read_csv(DATA_FILE_PATH, parse_dates=['date'])
        old_df['date'] = old_df['date'].dt.date
        combined_df = pd.concat([old_df, new_daily_pnl], ignore_index=True)
    else:
        # 如果文件不存在，直接使用新数据
        combined_df = new_daily_pnl
        os.makedirs('data', exist_ok=True) # 确保data目录存在

    # 按天聚合，防止同一天有多条记录
    final_df = combined_df.groupby('date')['pnl'].sum().reset_index()
    final_df = final_df.sort_values(by='date').reset_index(drop=True)
    
    final_df.to_csv(DATA_FILE_PATH, index=False)
    print(f"数据文件已更新: {DATA_FILE_PATH}")
    return final_df

def calculate_statistics(df):
    """基于每日盈亏数据计算各项关键指标"""
    if df.empty:
        return {}
    
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['equity_curve'] = INITIAL_CAPITAL + df['cumulative_pnl']
    
    total_days = (df['date'].max() - df['date'].min()).days + 1
    
    # 1. 总收益率
    total_return = (df['equity_curve'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    
    # 2. 年化收益率
    annual_return = ((1 + total_return / 100) ** (365.0 / total_days) - 1) * 100 if total_days > 0 else 0
    
    # 3. 夏普比率 (假设无风险利率为0)
    daily_returns = df['pnl'] / df['equity_curve'].shift(1).fillna(INITIAL_CAPITAL)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (365 ** 0.5) if daily_returns.std() != 0 else 0
    
    # 4. 最大回撤
    peak = df['equity_curve'].cummax()
    drawdown = (df['equity_curve'] - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    # 5. 胜率 (按天计算)
    win_days = (df['pnl'] > 0).sum()
    loss_days = (df['pnl'] < 0).sum()
    trade_days = win_days + loss_days
    win_rate = (win_days / trade_days) * 100 if trade_days > 0 else 0

    # 6. 盈亏比
    avg_win = df[df['pnl'] > 0]['pnl'].mean()
    avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean())
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    stats = {
        "总收益率 (%)": f"{total_return:.2f}",
        "年化收益率 (%)": f"{annual_return:.2f}",
        "夏普比率": f"{sharpe_ratio:.2f}",
        "最大回撤 (%)": f"{max_drawdown:.2f}",
        "日胜率 (%)": f"{win_rate:.2f}",
        "盈亏比": f"{profit_loss_ratio:.2f}",
        "总交易天数": trade_days,
        "盈利天数": win_days,
        "亏损天数": loss_days,
        "最后更新时间 (UTC)": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
        
    print("统计指标已生成: stats.json")
    return df # 返回带有计算列的df

def create_equity_curve_plot(df):
    """生成资金曲线图并保存"""
    if df.empty:
        print("数据为空，无法生成图表。")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(pd.to_datetime(df['date']), df['equity_curve'], label='Equity Curve', color='royalblue')
    ax.fill_between(pd.to_datetime(df['date']), df['equity_curve'], INITIAL_CAPITAL, 
                    where=(df['equity_curve'] > INITIAL_CAPITAL), 
                    facecolor='green', alpha=0.3, interpolate=True)
    ax.fill_between(pd.to_datetime(df['date']), df['equity_curve'], INITIAL_CAPITAL, 
                    where=(df['equity_curve'] <= INITIAL_CAPITAL), 
                    facecolor='red', alpha=0.3, interpolate=True)

    ax.set_title('Strategy Equity Curve', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity (USD)', fontsize=12)
    ax.legend()
    fig.autofmt_xdate() # 自动格式化日期标签
    
    plt.tight_layout()
    plt.savefig('equity_curve.png', dpi=150)
    print("资金曲线图已生成: equity_curve.png")

def main():
    """主执行函数"""
    start_ts = int(datetime.strptime(START_DATE_STR, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    # 确定从何时开始拉取数据，实现增量更新
    since_ts = start_ts
    if os.path.exists(DATA_FILE_PATH):
        try:
            df = pd.read_csv(DATA_FILE_PATH, parse_dates=['date'])
            if not df.empty:
                last_date = df['date'].max()
                since_ts = int(last_date.timestamp() * 1000) + 1 # 从最后一天的下一毫秒开始
        except Exception as e:
            print(f"读取旧数据文件失败: {e}，将从头开始获取。")

    exchange = get_exchange_client()
    new_trades = fetch_new_trades(exchange, since_ts)
    new_daily_pnl = process_trades_to_daily_pnl(new_trades)
    
    full_df = update_data_file(new_daily_pnl)
    
    # 只有当数据有更新时才重新生成报告
    if not new_daily_pnl.empty or not os.path.exists('equity_curve.png'):
        stats_df = calculate_statistics(full_df)
        create_equity_curve_plot(stats_df)
    else:
        print("没有新的交易数据，跳过报告生成。")

if __name__ == '__main__':
    main()