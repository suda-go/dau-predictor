import numpy as np
from scipy.optimize import curve_fit


def power_func(x, a, b):
    return a * np.power(x, b)


def fit_retention(days, retention_rates):
    """拟合留存曲线为幂函数 R(n) = a * n^b"""
    days = np.array(days, dtype=float)
    retention_rates = np.array(retention_rates, dtype=float)
    popt, _ = curve_fit(power_func, days, retention_rates, p0=[1.0, -0.5], maxfev=10000)
    return popt[0], popt[1]


def get_retention_curve(a, b, max_day):
    """根据拟合参数生成留存曲线"""
    days = np.arange(1, max_day + 1)
    return power_func(days, a, b)


LT_DIMENSIONS = [7, 30, 90, 180, 365]


def calc_lt_fitting(a, b, max_day):
    """通过幂函数积分计算 LT
    LT = 1(注册当天) + ∫₁^max_day a * x^b dx
    """
    if b == -1:
        integral = a * np.log(max_day)
    else:
        integral = a / (b + 1) * (max_day ** (b + 1) - 1)
    return 1.0 + integral


def calc_lt_sum(retention_rates):
    """直接对留存率累加计算 LT
    LT = 1(注册当天) + Σ retention_rates
    """
    return 1.0 + np.sum(retention_rates)


def calc_lt_multi(retention_rates, method="fitting"):
    """计算多维度 LT：LT7, LT30, LT90, LT180, LT365"""
    results = {}
    if method == "fitting":
        days = np.arange(1, len(retention_rates) + 1)
        a, b = fit_retention(days, np.array(retention_rates))
        for d in LT_DIMENSIONS:
            results[f"LT{d}"] = calc_lt_fitting(a, b, d)
    else:
        extended = extend_retention_curve(retention_rates, 365)
        for d in LT_DIMENSIONS:
            results[f"LT{d}"] = 1.0 + np.sum(extended[:d])
    return results


def parse_imported_retention(df):
    """从导入的 DataFrame 解析留存数据
    支持格式：
    - 列: day, retention (或 天数, 留存率)
    - 或者单列留存率数据
    """
    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {
        '留存率': 'retention', '留存': 'retention', 'retention': 'retention',
        'retention_rate': 'retention', 'rate': 'retention',
        '天数': 'day', 'day': 'day', 'days': 'day', 'n': 'day',
    }
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})

    if 'retention' in df.columns:
        retention = df['retention'].values.astype(float)
    elif len(df.columns) == 1:
        retention = df.iloc[:, 0].values.astype(float)
    elif len(df.columns) == 2:
        retention = df.iloc[:, 1].values.astype(float)
    else:
        retention = df.iloc[:, 1].values.astype(float)

    if np.any(retention > 1.0):
        retention = retention / 100.0

    return retention


def extend_retention_curve(retention_rates, target_days):
    """将留存曲线扩展到目标天数（用最后的衰减趋势外推）"""
    rates = np.array(retention_rates, dtype=float)
    if len(rates) >= target_days:
        return rates[:target_days]

    if len(rates) >= 2:
        last_ratio = rates[-1] / rates[-2] if rates[-2] > 0 else 0.95
        last_ratio = max(0.8, min(last_ratio, 0.999))
    else:
        last_ratio = 0.95

    extended = list(rates)
    for _ in range(target_days - len(rates)):
        next_val = max(extended[-1] * last_ratio, 0.001)
        extended.append(next_val)
    return np.array(extended)


def calc_daily_active_retention(retention_rates):
    """从留存曲线尾部计算日活跃留存率（老用户的日留存比例）
    取留存曲线后半段的日环比均值，代表成熟用户每天的留存概率
    """
    rates = np.array(retention_rates, dtype=float)
    if len(rates) < 4:
        return rates[-1] / rates[-2] if len(rates) >= 2 and rates[-2] > 0 else 0.95

    # 用后半段数据的日环比
    half = max(len(rates) // 2, 2)
    tail = rates[half:]
    ratios = tail[1:] / tail[:-1]
    ratios = ratios[(ratios > 0.5) & (ratios < 1.0)]
    if len(ratios) == 0:
        return 0.95
    return float(np.mean(ratios))


def predict_dau(new_users_array, retention_curve, predict_days, base_dau=0, base_daily_retention=None):
    """预测 DAU
    DAU(T) = 存量衰减 + 新增累积
    存量衰减：base_dau * daily_retention^T （老用户按日活跃留存率平缓衰减）
    新增累积：Σ new_users[i] * retention[T-i], i=0..T
    retention[0] = 1.0（注册当天）
    """
    full_retention = np.concatenate([[1.0], retention_curve])

    if len(full_retention) < predict_days:
        full_retention = np.concatenate([
            [1.0],
            extend_retention_curve(retention_curve, predict_days)
        ])

    if base_daily_retention is None:
        base_daily_retention = calc_daily_active_retention(retention_curve)

    # 基数DAU已包含当天新增，所以第0天直接用基数
    # 从第1天起：基数中的存量老用户按日活跃留存率衰减 + 每天新增用户按留存曲线累积
    # 存量老用户 = base_dau - 当天新增（第0天的新增已包含在基数中）
    base_old_users = max(base_dau - new_users_array[0], 0) if base_dau > 0 else 0.0

    dau = np.zeros(predict_days)
    for t in range(predict_days):
        if t == 0 and base_dau > 0:
            # 第0天直接用基数（已包含当天新增）
            dau[t] = base_dau
            continue

        # 存量老用户衰减
        base_remaining = base_old_users * (base_daily_retention ** t) if base_dau > 0 else 0.0

        # 新增用户累积（从第0天的新增开始算留存贡献）
        new_contribution = 0.0
        for i in range(t + 1):
            if i < len(new_users_array):
                nu = new_users_array[i]
            else:
                nu = new_users_array[-1]
            day_diff = t - i
            if day_diff < len(full_retention):
                new_contribution += nu * full_retention[day_diff]

        dau[t] = base_remaining + new_contribution
    return dau


def generate_new_users_array(mode, predict_days, daily_new=1000, growth_rate=0.0, custom_array=None):
    """生成新增用户数组"""
    if mode == "fixed":
        return np.full(predict_days, daily_new, dtype=float)
    elif mode == "growth":
        return np.array([daily_new * (1 + growth_rate) ** i for i in range(predict_days)])
    elif mode == "custom" and custom_array is not None:
        arr = np.array(custom_array, dtype=float)
        if len(arr) < predict_days:
            arr = np.concatenate([arr, np.full(predict_days - len(arr), arr[-1])])
        return arr[:predict_days]
    return np.full(predict_days, daily_new, dtype=float)


def calc_required_retention(target_dau, daily_new, current_retention, method="fitting"):
    """根据目标 DAU 和日新增，反推需要的留存水平
    稳态 DAU ≈ daily_new × LT，所以 LT_needed = target_dau / daily_new
    然后根据当前留存曲线的形状（衰减速率 b），反推需要的 a 值，
    再输出建议的关键天留存率。

    返回: dict with required_lt, suggested_retention (关键天), scale_factor
    """
    lt_needed = target_dau / daily_new

    days = np.arange(1, len(current_retention) + 1)
    current_arr = np.array(current_retention, dtype=float)

    if method == "fitting":
        a, b = fit_retention(days, current_arr)
        current_lt = calc_lt_fitting(a, b, 365)
        scale_factor = lt_needed / current_lt if current_lt > 0 else 1.0
        a_new = a * scale_factor
        suggested = {}
        for d in [1, 3, 7, 14, 30, 60, 90]:
            suggested[f"第{d}天"] = min(power_func(d, a_new, b), 1.0)
    else:
        current_lt = calc_lt_sum(current_arr)
        scale_factor = lt_needed / current_lt if current_lt > 0 else 1.0
        suggested = {}
        for d in [1, 3, 7, 14, 30, 60, 90]:
            if d - 1 < len(current_arr):
                suggested[f"第{d}天"] = min(current_arr[d - 1] * scale_factor, 1.0)
            else:
                ext = extend_retention_curve(current_arr, d)
                suggested[f"第{d}天"] = min(ext[d - 1] * scale_factor, 1.0)

    return {
        "required_lt": lt_needed,
        "current_lt": current_lt if method == "fitting" else calc_lt_sum(current_arr),
        "scale_factor": scale_factor,
        "suggested_retention": suggested,
    }
