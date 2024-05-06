import datetime


def get_monthly_periods(from_date:str, to_date:str) -> list:
    """
    Generate a list of tubple of begining and end of each month
    
    Example:
    get_monthly_periods(from_date = "2022-01-03", to_date = "2022-02-10")
    return:
    [('2022-01-01', '2022-01-31'), ('2022-02-01', '2022-02-28')]
    """
    dt_from = datetime.date.fromisoformat(from_date)
    dt_to = datetime.date.fromisoformat(to_date)
    months = (dt_to.year - dt_from.year) * 12 + dt_to.month - dt_from.month + 1
    periods_list = []
    this_month = dt_from.replace(day=1)
    
    for _ in range(months):
        next_month = (this_month.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        this_month_lastday = next_month - datetime.timedelta(days=1)
        periods_list.append((this_month.isoformat(), this_month_lastday.isoformat()))

        this_month = next_month


    return periods_list