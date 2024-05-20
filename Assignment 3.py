import numpy as np

# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_MATRICNUMBER(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """
    # your code goes here

    # Part a
    a_out = np.zeros(num_iters)
    f1_out = np.zeros(num_iters)
    a = 2.5 # Initialisation
    a_out[0] = a - (learning_rate * 4 * a**3)
    f1_out[0] = a_out[0]**4

    # Part b
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)
    b = 0.6 # Initialisation - Radians
    b_out[0] = b - (learning_rate * 2*np.sin(b)*np.cos(b))
    f2_out[0] = np.sin(b_out[0]) ** 2

    # Part c
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)
    c, d = 2, 3
    c_out[0] = c - (learning_rate * (5 * c**4))
    d_out[0] = d - (learning_rate * (2 * d * np.sin(d) + d**2 * np.cos(d)))
    f3_out[0] = (c_out[0]**5) + (d_out[0]**2 * np.sin(d_out[0]))

    for i in range(1, num_iters):
        aVal = a_out[i-1]
        a_out[i] = aVal - (learning_rate * 4 * aVal**3)
        f1_out[i] = a_out[i]**4

        bVal = b_out[i-1]
        b_out[i] = bVal - (learning_rate * 2*np.sin(bVal)*np.cos(bVal))
        f2_out[i] = np.sin(b_out[i]) ** 2

        cVal = c_out[i-1]
        dVal = d_out[i-1]
        c_out[i] = cVal - (learning_rate * (5 * cVal**4))
        d_out[i] = dVal - (learning_rate * (2 * dVal * np.sin(dVal) + dVal**2 * np.cos(dVal)))
        f3_out[i] = (c_out[i]**5) + (d_out[i]**2 * np.sin(d_out[i]))

    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 

# TESTING

# learning_rate = 0.1
# num_iters = 5
# a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out = A3_MATRICNUMBER(learning_rate, num_iters)
# print(f"a_out: {a_out}, \nf1_out: {f1_out}, \nb_out: {b_out}, \nf2_out: {f2_out}, \nc_out: {c_out}, \nd_out: {d_out}, \nf3_out: {f3_out}")