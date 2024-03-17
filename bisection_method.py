import math
import numpy as np
from colors import bcolors
import sympy as sp
from sympy import *
from sympy.utilities.lambdify import lambdify


"""
Receives 3 parameters:
    1.  a - start value.
    2.  b - end  value. 
    3.  err - value of tolerable error

Returns variables:
    1.  S - The minimum number of iterations required to reach the desired accuracy
"""
def max_steps(a, b, err):
    s = int(np.floor(- np.log2(err / (b - a)) / np.log2(2) - 1))
    return s

"""
Performs Iterative methods for Nonlinear Systems of Equations to determine the roots of the given function f
Receives 4 parameters:
    1. f - continuous function on the interval [a, b], where f (a) and f (b) have opposite signs.
    2. a - start value.
    3. b - end  value. 
    4. tol - the tolerable error , the default value will set as 1e-16

Returns variables:
    1.  c - The approximate root of the function f
"""
def bisection_method(f, a, b, tol=1e-6):
    h = sp.diff(f)
    f = lambdify(x,f)
    z = f
    flag = False
    K = []
    A = []
    B = []
    C = []
    F_a = []
    F_b = []
    F_c = []

    if np.sign(f(a)) == np.sign(f(b)):
        h = lambdify(x, h)
        if np.sign(h(a)) == np.sign(h(b)):
            raise Exception("The scalars a and b do not bound a root")
        else:
            f = h
            flag = True

    c, k = 0, 0
    steps = max_steps(a, b, tol)  # calculate the max steps possible

    #print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Iteration", "a", "b", "f(a)", "f(b)", "c", "f(c)"))

    # while the diff af a&b is not smaller than tol, and k is not greater than the max possible steps
    while abs(b - a) > tol and k < steps:
        c = a + (b - a) / 2  # Calculation of the middle value

        if f(c) == 0 :
            return c  # Procedure completed successfully

        if f(c) * f(a) < 0:  # if sign changed between steps
            b = c  # move forward
        else:
            a = c  # move backward

        #print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(k, a, b, f(a), f(b), c, f(c)))
        #saving the data
        K.append(k)
        A.append(a)
        B.append(b)
        C.append(c)
        F_a.append(f(a))
        F_b.append(f(b))
        F_c.append(f(c))

        k += 1

    if flag:  #check if that a derivative function
        if abs(z(c)) > 0.0001:  #check if it real a root in the source function if not it will raise exception
            raise Exception("The scalars a and b do not bound a root")

    #print the data
    print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Iteration", "a", "b", "f(a)", "f(b)", "c", "f(c)"))
    for i in range(k):
        print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(K[i], A[i], B[i], F_a[i], F_b[i], C[i], F_c[i]))

    return c  # return the current root



if __name__ == '__main__':
    x = sp.symbols('x')
    f = x**2 - 4 * sin(x)
    g = (x-2)**2
    a = -1
    b = 3
    print(f"the input function is {f} and the limts is {a} and {b}")
    print("https://github.com/Babilabong/analiza_nomarit_tester_2\ngroup:Almog Babila 209477678, Hai karmi 207265678, Yagel Batito 318271863, Meril Hasid 318271863\nstudent:Almog Babila 209477678")
    jump = (b-a)/10
    i=a+jump
    while i<=b:
        try:
            roots = bisection_method(f, a, i)
            print(bcolors.OKBLUE, f"\nThe equation f(x) has an approximate root at x = {roots}",bcolors.ENDC,)
            print()
        except Exception:
            print(f"none roots between ({a})-({i})\n")
        a = i
        i = i + jump