from __future__ import print_function
from scipy import linspace, polyval, polyfit, sqrt, stats, randn, optimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

#Generate random data (orient to linear regresstion model)
#Y=W*X + B
#Sample data
n = int(5e6) #milions data
t = np.linspace(-10,10,n) #Random from -10 to 10, may be there is milions number
#parameters
a=3.25; b= -6.5
x=polyval([a,b],t) #retun array a*t**1 + b*t**0
#add some noise
xn=x+3*randn(n)

#Draw random sample points
# xvar = np.random.choice(t, size = 20)
# yvar = polyval([a,b], xvar) + 3 * randn(20)
# plt.scatter(xvar, yvar, c='green', edgecolors='k')
# plt.grid(True)
# plt.show()

#Method: Scipy.Polyfit
print('**********Method: Scipy.Polyfit***********')
#Linear regression - polyfit
t1=time.time()
#Polyfit:
#Fit a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to points (x, y).
#Returns a vector of coefficients p that minimises the squared error.
(ar,br) = polyfit(t,xn,1) #Calculate W
xr=polyval([ar,br],t) #Test results
#Compute the mean suare error
err=sqrt(sum((xr-xn)**2)/n) #Lost function
t2=time.time()
t_polyfit=float(t2-t1)
print('Linear regression using polyfit')
print('parameters: a=%.2f b= %.2f, ms error= %.3f' % (ar,br,err))
print('Time taken: {} seconds'.format(t_polyfit))


#Method: Stats.linregress
print('**********Method: Stats.linregress***********')
#Liner regression using stats.linregress
#linregress: Calculate a linear least-squares regression
#Return slope, intercept, coefficient, standard error,...
#Y = intercept + slope*x
t1=time.time()
(a_s, b_s, r, tt, stderr) = stats.linregress(t,xn)
t2=time.time()
t_linregress = float(t2-t1)
print('Linear regression using stats.linregress')
print('parameters: a=%.2f b = %.2f, std error=%.3f, r^2 coefficient= %.3f' % (a_s, b_s, stderr, r))
print('Time taken: {} seconds'.format(t_linregress))

#Method: Optimize.curve_fit
print('**********Method: Optimize.curve_fit***********')
# Use non-linear least squares to fit a function, f, to data.
# Assumes ydata = f(xdata, *params) + eps
# Optimal values for the parameters so that the sum of the squared residuals of
# f(xdata, *popt) - ydata is minimized
def flinear(t,a,b):
    result=a*t+b
    return (result)
t1=time.time()
popt,pcov =optimize.curve_fit(flinear, xdata=t, ydata=xn, method='lm')
t2=time.time()
t_optimize_curve_fit = float(t2-t1)
print('Linear regression using optimize.curve_fit')
print('parameters: a=%.2f, b=%.2f' % (popt[0],popt[1]))
print('Time taken: {} seconds'.format(t_optimize_curve_fit))


#Method: numpy.linalg.lstsq
print('**********Method: numpy.linalg.lstsq***********')
# Return the least-squares solution to a linear matrix equation.
# Solves the equation a x = b by computing a vector x
# that minimizes the Euclidean 2-norm || b - a x ||^2.
t1=time.time()
A=np.vstack([t, np.ones(len(t))]).T
result = np.linalg.lstsq(A, xn, rcond=None)
ar,br = result[0]
err = np.sqrt(result[1]/len(xn))
t2=time.time()
t_linalg_lstsq = float(t2-t1)
print('Linear regression using numpy.linalg.lstsq')
print('parameters: a=%.2f b=%.2f, ms error=%.3f' % (ar, br, err))
print('Time taken: {} seconds'.format(t_linalg_lstsq))

#Method: Statsmodels.OLS
print('**********Method: Statsmodels.OLS***********')
t1=time.time()
t=sm.add_constant(t)
model = sm.OLS(x,t)
results=model.fit()
ar = results.params[1]
br=results.params[0]
t2=time.time()
t_OLS = float(t2-t1)
print('Linear regression using Statsmodels.OLS')
print('parameters: a=%.2f b=%.2f' % (ar, br))
print('Time taken: {} seconds'.format(t_OLS))

# print(results.summary())

#Analytic solution using Moore-Penrose pseudoinverse
print('**********Method: Moore-Penrose pseudoinverse***********')
# # The pseudo-inverse of a matrix A, denoted A^+, is defined as:
# # “the matrix that ‘solves’ [the least-squares problem] Ax = b,” i.e.,
# if \bar{x} is said solution, then A^+ is that matrix such that \bar{x} = A^+b.

t1=time.time()
mpinv=np.linalg.pinv(t)
result=mpinv.dot(x)
ar=result[1]
br=result[0]
t2=time.time()
t_inv_matrix = float(t2-t1)
print('Linear regression using Moore-Penrose inverse')
print('parameters: a=%.2f b=%.2f'% (ar,br))
print("Time taken: {} seconds".format(t_inv_matrix))

#Analytic solution using simple multiplicative matrix inverse
print('**********Method: simple multiplicative matrix inverse***********')
t1=time.time()
m=np.dot(np.dot(np.linalg.inv(np.dot(t.T,t)),t.T),x)
ar = m[1]
br = m[0]
t2 = time.time()
t_simple_inv = float(t2-t1)
print('Linear regression using simple inverse')
print('parameters: a=%.2f b=%.2f'% (ar,br))
print("Time taken: {} seconds".format(t_simple_inv))

#Method: sklearn.linear_model.LinearRegression
print('**********Method: sklearn.linear_model.LinearRegression***********')
t1=time.time()
lm=LinearRegression()
lm.fit(t,x)
ar=lm.coef_[1]
br=lm.intercept_
t2=time.time()
t_sklearn_linear=float(t2-t1)
print('Linear regression using sklearn.linear_model.LinearRegression')
print('parameters: a=%.2f b=%.2f'% (ar,br))
print("Time taken: {} seconds".format(t_sklearn_linear))


#Bucket all the execution times in a list and plot

times =  [t_polyfit, t_linregress, t_optimize_curve_fit, t_linalg_lstsq
            , t_OLS, t_inv_matrix, t_simple_inv, t_sklearn_linear]
plt.figure(figsize=(20,5))
plt.grid(True)
plt.bar(left=[l*0.8 for l in range(8)],height=times, width=0.4,
        tick_label=['Polyfit','Stats.linregress','Optimize.curve_fit',
                    'numpy.linalg.lstsq','statsmodels.OLS','Moore-Penrose matrix inverse',
                    'Simple matrix inverse','sklearn.linear_model'])
plt.show()