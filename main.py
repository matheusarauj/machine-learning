
from numpy import *
from math import sqrt

def compute_erros_for_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m*x + b))**2

    return totalError / float(len(points))

def step_gradient(current_b, current_m, points, learning_rate):
    # gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((current_m*x) + current_b))
        m_gradient += -(2/N) * x * (y - ((current_m*x) + current_b))

    new_b = current_b - (learning_rate * b_gradient)
    new_m = current_m - (learning_rate * m_gradient)

    return[new_b, new_m, b_gradient, m_gradient]

def current_gradient_rate(b_gradient, m_gradient):
    return sqrt((b_gradient**2) + (m_gradient**2))
    
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    b_gradient = 1
    m_gradient = 1
    iterations = 0

    while(current_gradient_rate(b_gradient, m_gradient) > learning_rate):
        b, m, b_gradient, m_gradient = step_gradient(b, m, array(points), learning_rate)
        iterations += 1

    return [b, m, iterations]

def normal_equations(points, initial_b, initial_m, learning_rate, num_iterations):
    tempx = 0
    tempy = 0
    points_xy = array(points)

    for i in range(len(points_xy)):
        tempx += points_xy[i, 0]
        tempy += points_xy[i, 1]

    avg_x = tempx/float(len(points_xy))
    avg_y = tempy/float(len(points_xy))

    a = 0
    b = 0

    for i in range(len(points_xy)):
        a += (points_xy[i, 0] - avg_x)*(points_xy[i, 1] - avg_y)
        b += (points_xy[i, 0] - avg_x)**2

    w1 = a/b
    w0 = avg_y - w1*avg_x

    return[w0, w1, 0]

def run():
    points = genfromtxt("income.csv", delimiter=",")
    learning_rate = 0.0001

    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 0

    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_erros_for_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m, num_iterations] = normal_equations(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_erros_for_given_points(b, m, points)))
    print(b)
    print(m)

if __name__ == '__main__':
    run()