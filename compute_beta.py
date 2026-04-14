import statsmodels.api as sm

def compute_beta(y, x):
    """
    Menghitung Beta menggunakan Ordinary Least Squares (OLS).
    y: Return portofolio
    x: Return indeks pembanding
    """
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model.params.iloc[1]
