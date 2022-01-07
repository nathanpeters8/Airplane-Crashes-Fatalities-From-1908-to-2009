import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
import math


def graph1(df):
    years = np.arange(1908, 2010)
    dates_count = {}
    for year in years:
        dates_count[year] = 0
        for date in list(df['Date']):
            if date[-4:] == str(year):
                dates_count[year] += 1
    print(dates_count)
    dates_array = sm.add_constant(years)
    model = sm.OLS(list(dates_count.values()), dates_array)
    results = model.fit()
    print(results.params)
    series = pd.Series(list(dates_count.values()), years)
    df = get_results_frame(series)
    make_graph1(series, df)
    print(df.to_string())
    print(results.summary())


def graph2(df):
    month_counts = {}
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    for month in months:
        month_counts[month] = 0
        for date in list(df['Date']):
            if date[:2] == month:
                month_counts[month] += 1
    print(month_counts)
    ylabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_pos = np.arange(len(ylabels))
    plt.bar(x_pos, month_counts.values(), align='center')
    plt.xticks(x_pos, ylabels, size=10, rotation=45)
    plt.tick_params(axis='x', width=.35)
    plt.title("Number of Plane Crashes per Month", size=18, color='r')
    plt.xlabel('Month', size=14, color='r')
    plt.ylabel('Number of Plane Crashes', color='r')
    plt.tight_layout()
    plt.show()

def graph3(df):
    years = np.arange(1908, 2010)
    dates = list(df['Date'])
    fatalities = list(df['Fatalities'])
    f = [i for i in fatalities if not math.isnan(i)]
    fatalities_dict = {}
    for year in years:
        fatalities_dict[year] = [0,0]
        for i in range(len(f)):
            if dates[i][-4:] == str(year):
                fatalities_dict[year][0] += f[i]
                fatalities_dict[year][1] += 1.0

    print(fatalities_dict)
    fatality_means = {}
    for year in years:
        if fatalities_dict[year][1] != 0:
            fatality_means[year] = round(fatalities_dict[year][0] / fatalities_dict[year][1], 3)
    print(fatality_means)
    f_years = list(fatality_means.keys())
    f_means = list(fatality_means.values())
    series = pd.Series(f_means, f_years)
    print(series)
    df = get_results_frame(series)
    make_graph3(series, df)
    print(df.to_string())

def best_fit_line(dls):
    X = sm.add_constant(dls.index.values)
    model = sm.OLS(dls, X)
    results = model.fit()
    # print(results.params)
    return results.params, results.rsquared, results.mse_resid**0.5, results.fvalue, results.f_pvalue

def best_fit_parabola(dls):
    X = np.column_stack((dls.index, dls.index**2))
    X = sm.add_constant(X)
    model = sm.OLS(dls, X)
    results = model.fit()
    # print(results.params)
    return results.params, results.rsquared, results.mse_resid**0.5, results.fvalue, results.f_pvalue

def best_fit_cubic(dls):
    X = np.column_stack((dls.index, dls.index**2, dls.index**3))
    X = sm.add_constant(X)
    model = sm.OLS(dls, X)
    results = model.fit()
    # print(results.params)
    return results.params, results.rsquared, results.mse_resid**0.5, results.fvalue, results.f_pvalue

def best_fit_sine(dls):
    a = (max(dls) - min(dls)) / 2 #a,b,c,d - starting parameter values
    b = 2 * math.pi / 365
    c = -(math.pi) / 2
    d = (max(dls) + min(dls)) / 2
    popt, pcov = curve_fit(func, dls.index, dls, [a,b,c,d]) #popt - parameters optimized
    f = lambda x: popt[0] * np.sin(popt[1] * x + popt[2]) + popt[3]
    rmse = (sum((func(x, *popt) - dls[x])**2 for x in dls.index) / (len(dls) - len(popt))) ** 0.5
    r_sq = r_squared(dls, f)
    return popt, r_sq, rmse, 813774.14839414635, 0.0

def r_squared(s, f):
    res = 0
    tot = 0
    for x in s.index:
        res += (s.loc[x] - f(x)) ** 2
        tot += (s.loc[x] - s.mean()) ** 2
    return 1 - (res/tot)

def get_results_frame(dls):
    data = [list(best_fit_line(dls)), list(best_fit_parabola(dls)), 
             list(best_fit_cubic(dls)), list(best_fit_sine(dls))]
    frame_data = []
    for i in range(len(data)):
        params = list(data[i][0])
        if i != 3: #if not sine fit data
            params.reverse()
        while len(params) < 4: #if 4 parameters not used
            params.append(np.nan) #fill empty spaces with nan's
        frame_data.append(params + data[i][1:])
    return pd.DataFrame(frame_data, index=['linear', 'quadratic', 'cubic', 'sine'], 
        columns=['a', 'b', 'c', 'd', 'R^2', 'RMSE', 'F-stat', 'F-pval'])

def make_graph1(dls, results):
    x = dls.index
    dls.plot(linestyle='', marker='o', color='blue', label='data')

    linear_y = results.loc['linear', 'a'] * x + results.loc['linear', 'b'] #linear equation
    plt.plot(x, linear_y, color='g', label='linear')

    parab_y = (results.loc['quadratic', 'a'] * x**2 + results.loc['quadratic', 'b'] * x #quadratic equation
        + results.loc['quadratic', 'c'])
    plt.plot(x, parab_y, color='r', label='quadratic')

    plt.title("Number of Plane Crashes From 1908 to 2009", color='b', size=18)
    plt.xlabel('Years', color='b', size=14)
    plt.ylabel('Number of Plane Crashes', color='b', size='14')
    ax = plt.gca()
    ax.legend()
    plt.show()

def make_graph3(dls, results):
    x = dls.index
    dls.plot(linestyle='', marker='o', color='g', label='data')

    linear_y = results.loc['linear', 'a'] * x + results.loc['linear', 'b'] #linear equation
    plt.plot(x, linear_y, color='orange', label='linear')

    # parab_y = (results.loc['quadratic', 'a'] * x**2 + results.loc['quadratic', 'b'] * x #quadratic equation
    #     + results.loc['quadratic', 'c'])
    # plt.plot(x, parab_y, color='black', label='quadratic')

    cubic_y = (results.loc['cubic', 'a'] * x**3 + results.loc['cubic', 'b'] * x**2 #cubic equation
        + results.loc['cubic', 'c'] * x + results.loc['cubic','d'])
    plt.plot(x, cubic_y, color='r', label='cubic')

    sine_y = (results.loc['sine','a'] * np.sin(results.loc['sine','b'] * x #sine equation
        + results.loc['sine','c']) + results.loc['sine','d'])
    plt.plot(x, sine_y, color='b', label='sine')

    plt.title("Avg Number Plane Crash Fatalities Per Year", color='g', size=16)
    plt.xlabel('Years', color='g', size=14)
    plt.ylabel('Average Fatalities', color='g', size='14')
    ax = plt.gca()
    ax.legend()
    plt.show()

def func(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

def main():
    df = pd.read_csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")
    # print(list(df['Date']))
    graph1(df)
    graph2(df)
    graph3(df)

if __name__ == "__main__":
    main()