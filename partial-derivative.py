from sympy import *
from math import sqrt
import numpy as np

# define the number of variables
vlen = int(input('Please enter how many variables you have: '))
# define the number of differential variables
vdiff = int(input('Please enter how many variables you want to differentiate wrt: '))
# creating symbols from the numbers defined above
vnames = ['x' + str(i) for i in range(vlen)]
variables = [symbols(n.replace('x', 'x')) for n in vnames]
vname = ['x' + str(i) for i in range(vdiff)]
variablesdiff = [symbols(n.replace('x', 'x')) for n in vname]

# updating the global dictionary
globals().update(dict(zip(vnames, variables)))

# showing the variables to be used
print(f'Please use the following variables in your equation: {variables}, it will be differentiated wrt to: {variablesdiff}')
# taking the input of the equation
f = input('Please enter your equation: ').lower()
# turning input into an equation
func = eval(f)

# initialising the lists
difflist =[]
inputlist=[]
sumlist=[]
errorlist =[]
soln =[]

# taking input for substitutions
inputlist = list(float(num) for num in input("Enter the variable substitutions separated by space: ").strip().split())[:vdiff]
errorlist = list(float(num) for num in input("Enter the error values separated by space: ").strip().split())[:vdiff]

# for loop to differentiate each part of the equation and substituting
for i in range(vdiff):
    d = diff(func, variablesdiff[i])
    difflist.append(d)
    e = d.subs(variablesdiff[i], inputlist[i])
    sumlist.append(e)
    final = (sumlist[i]*errorlist[i])**2
    soln.append(final)

difflt = sum(difflist)
sumlt = sum(sumlist)
sol = sqrt(sum(soln))

# printing everything
print(f'The partial derivative is: {difflt}')
print(f'The numerical solution of the derivative is: {sumlt:.2E}')
print(f'The solution is: {sol:.2E}')
