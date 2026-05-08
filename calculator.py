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


def predict_dau(new_users_array, retention_curve, predict_days):
    """预测 DAU
    DAU(T) = Σ new_users[i] * retention[T-i], i=0..T
    retention[0] = 1.0（注册当天）
    """
    full_retention = np.concatenate([[1.0], retention_curve])

    if len(full_retention) < predict_days:
        full_retention = np.concatenate([
            [1.0],
            extend_retention_curve(retention_curve, predict_days)
        ])

    dau = np.zeros(predict_days)
    for t in range(predict_days):
        daily_dau = 0.0
        for i in range(t + 1):
            if i < len(new_users_array):
                nu = new_users_array[i]
            else:
                nu = new_users_array[-1]
            day_diff = t - i
            if day_diff < len(full_retention):
                daily_dau += nu * full_retention[day_diff]
        dau[t] = daily_dau
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
