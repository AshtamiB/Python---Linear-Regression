# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 00:39:47 2018

@author: Ashtami
"""

import scipy.stats as sps
import numpy.polynomial.polynomial as nppp
from scipy import linspace, sqrt, randn
from pylab import plot, title, show, legend
#Sample data creation
#number of points
n = 50
t = linspace(-5, 5, n)
#parameters
a = 0.8
b = -4
x = nppp.polyval(t,[a, b])
xn= x+randn(n) #add some noise
(ar,br) = nppp.polyfit(t,xn,1)
xr = nppp.polyval(t,[ar,br])
#compute the mean square error
err = sqrt(sum((xr-xn)**2)/n)
print('Linear regression using polyfit')
print('parameters: a=%.2f b=%.2f \nregression: a=%.2f b=%.2f,
ms error= %.3f' % (a,b,ar,br,err))
print('-----------------------------------------------------')
#Linear regression using stats.linregress
(a_s,b_s,r,tt,stderr) = sps.linregress(t,xn)
print('Linear regression using stats.linregress')
print('parameters: a=%.2f b=%.2f \nregression: a=%.2f b=%.2f,
std error= %.3f' % (a,b,a_s,b_s,stderr))
#matplotlib ploting
title('Linear Regression Example')
plot(t,x,'g.--')
plot(t,xn,'k.')
plot(t,xr,'r.-')
legend(['original','plus noise', 'regression'])
show()