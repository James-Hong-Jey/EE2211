import numpy as np

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_MATRICNUMBER(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :InvXTX type: numpy.ndarray
    :w type: numpy.ndarray
   
    """

    # your code goes here
    # (Xt * X)
    InvXTX = np.linalg.inv(X.T @ X) 

    w = InvXTX @ X.T @ y

    # return in this order
    return InvXTX, w

if __name__ == "__main__":
    # Example usage
    X = np.array([[1, 1, 2],
                [1, 0, 6],
                [1, 1, 0],
                [1, 0, 5],
                [1, 1, 7]])

    y = np.array([[1],
                [2],
                [3],
                [4],
                [5]])

    X_pseudo_inv, w = A1_MATRICNUMBER(X, y)

    print("Pseudo-inverse of X:")
    print(X_pseudo_inv)
    print("\nLeast squares solution w:")
    print(w)

    x = np.array([[1, 1, 3]])
    print("Now the output is:")
    print(x @ w)
    