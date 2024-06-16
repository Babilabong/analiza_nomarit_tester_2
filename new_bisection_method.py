import math
import numpy as np
from colors import bcolors
import sympy as sp
from sympy import *
from sympy.utilities.lambdify import lambdify


# פונקציות שנוספו:

def is_root_bound(function, start_value, end_value):
    """Check if there is a root between start_value and end_value."""
    return np.sign(function(start_value)) != np.sign(function(end_value))


def update_interval(function, start_value, end_value):
    """Update the interval for the bisection method based on the midpoint."""
    mid_value = start_value + (end_value - start_value) / 2
    if function(mid_value) * function(start_value) < 0:
        return start_value, mid_value
    else:
        return mid_value, end_value


def print_bisection_table(iterations, a_values, b_values, fa_values, fb_values, c_values, fc_values):
    """Print the bisection table with iteration results."""
    print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Iteration", "a", "b", "f(a)", "f(b)", "c", "f(c)"))
    for i in range(len(iterations)):
        print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            iterations[i], a_values[i], b_values[i], fa_values[i], fb_values[i], c_values[i], fc_values[i]
        ))


# פונקציות ששונו:

def calculate_max_steps(start_value, end_value, tolerable_error):
    """Calculate the maximum number of iterations to reach the desired accuracy."""
    max_iterations = int(np.floor(- np.log2(tolerable_error / (end_value - start_value)) / np.log2(2) - 1))
    return max_iterations


def bisection_method(function, start_value, end_value, tolerable_error=1e-6):
    """Perform the bisection method to find the root of the function."""
    derivative = sp.diff(function)
    function = lambdify(x, function)
    original_function = function
    is_derivative = False

    iterations, a_values, b_values, c_values, fa_values, fb_values, fc_values = [], [], [], [], [], [], []

    if not is_root_bound(function, start_value, end_value):
        derivative = lambdify(x, derivative)
        if not is_root_bound(derivative, start_value, end_value):
            raise Exception("The scalars start_value and end_value do not bound a root")
        else:
            function = derivative
            is_derivative = True

    mid_value, iteration = 0, 0
    max_iterations = calculate_max_steps(start_value, end_value, tolerable_error)

    while abs(end_value - start_value) > tolerable_error and iteration < max_iterations:
        mid_value = start_value + (end_value - start_value) / 2

        if function(mid_value) == 0:
            return mid_value

        start_value, end_value = update_interval(function, start_value, end_value)

        iterations.append(iteration)
        a_values.append(start_value)
        b_values.append(end_value)
        c_values.append(mid_value)
        fa_values.append(function(start_value))
        fb_values.append(function(end_value))
        fc_values.append(function(mid_value))

        iteration += 1

    if is_derivative:
        if abs(original_function(mid_value)) > 0.0001:
            raise Exception("The scalars start_value and end_value do not bound a root")

    print_bisection_table(iterations, a_values, b_values, fa_values, fb_values, c_values, fc_values)

    return mid_value


if __name__ == '__main__':
    x = sp.symbols('x')
    function = 6 * (x ** 6) - 3 * (x ** 5) - 2 * (x ** 2) - 7
    a = -2
    b = 2
    print(f"The input function is {function} and the limits are {a} and {b}")
    print(
        "https://github.com/Babilabong/analiza_nomarit_tester_2\ngroup:Almog Babila 209477678, Hai karmi 207265678, Yagel Batito 318271863, Meril Hasid 324569714\nstudent:Almog Babila 209477678")

    interval_step = (b - a) / 10
    current_start = a + interval_step

    while current_start <= b:
        try:
            root = bisection_method(function, a, current_start)
            print(bcolors.OKBLUE, f"\nThe equation f(x) has an approximate root at x = {root}", bcolors.ENDC)
        except Exception:
            print(f"No roots between ({a}) - ({current_start})\n")
        a = current_start
        current_start += interval_step