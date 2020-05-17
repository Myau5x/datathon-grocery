times = pd.DataFrame({'times^1':np.linspace(0,1, 187)})
times = sm.add_constant(times)
n = 25
for x in range(2, n+1):
    col_n = 'times^'+str(x)
    col_s = 'times^'+str(x-1)
    times[col_n] = times['times^1']*times[col_s]

results = []
for x in range(1,n+1):
    model = sm.OLS(pears_prof_week.values, times.iloc[:,:x+1])
    #models.append(model)
    results.append(model.fit())

### Whats AIC BIC???
aic=[]
bic =[]

for res  in results:
    aic.append(res.aic)
    bic.append(res.bic)
AICBIC = pd.DataFrame({'aic': aic, 'bic':bic}, index=range(1,n+1))
AICBIC['a_o_b']= AICBIC['aic']/AICBIC['bic']
AICBIC.T

fig, axs = plt.subplots(24, 1 , figsize = (20,50))
for model, ax in zip(results, axs.flatten()):
    ax.plot(pears_prof_week, label = 'data', color = 'black')
    degree = len(model.params)-1
    ax.plot(pears_prof_week.index, model.fittedvalues, label = 'degree ' + str(degree))

    ax.legend()

def create_monthly_dummies(series):
    month = series.index.month
    # Only take 11 of the 12 dummies to avoid strict colinearity.
    return pd.get_dummies(month).ix[:, :11]

def fit_seasonal_trend(series):
    dummies = create_monthly_dummies(series)
    X = sm.add_constant(dummies.values, prepend=False)
    seasonal_model = sm.OLS(series.values, X).fit()
    return seasonal_model.predict(X)


def plot_seasonal_trend(ax, name, series):
    seasons_average_trend = fit_seasonal_trend(series)
    ax.plot(series.index, series)
    ax.set_title(name)

    ax.plot(series.index, seasons_average_trend)

model_comb = sm.OLS(pears_prof_week.values, times_month_p.iloc[:,:-2])  ### degree 25-k
res_comb= model_comb.fit()

fig, ax = plt.subplots( figsize=(14, 5))
ax.plot(pears_prof_week.index, res_comb.fittedvalues, label = 'model prediction')
ax.plot(pears_prof_week, label = 'data')
ax.legend()
