import numpy as np
from numpy.linalg import inv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_MATRICNUMBER(N):

    iris = load_iris()
    X = iris['data']
    y = iris['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state = N)

    one_hot_encoder = OneHotEncoder(sparse_output=False) # False means array
    reshaped_ytr = y_train.reshape(len(y_train), 1) # turn it into a column
    Ytr = one_hot_encoder.fit_transform(reshaped_ytr)  

    # Do the same for the y_test data
    reshaped_yts = y_test.reshape(len(y_test), 1)
    Yts = one_hot_encoder.fit_transform(reshaped_yts)

    reg_factor = 0.0001
    Ptrain_list, Ptest_list, w_list, error_train, error_test= [], [], [], [], []

    for order in range (8):
        poly = PolynomialFeatures(degree= (order + 1) )
        P = poly.fit_transform(X_train)
        Pt = poly.fit_transform(X_test)

        Ptrain_list.append(P)
        Ptest_list.append(Pt)

        #Check for over/ underdetermined
        #Use the appropriate psuedo inverse (left/right)
        samples = P.shape[0]
        unknowns = P.shape[1]
        if samples > unknowns: # Left inverse - overdetermined
            reg_L = reg_factor * np.identity(unknowns)
            wp = inv(P.T @ P + reg_L) @ P.T @ Ytr # Primal Ridge
        else: # Right inverse - underdetermined
            reg_L2 = reg_factor * np.identity(samples)
            wp = P.T @ inv(P @ P.T + reg_L2) @ Ytr # Dual Ridge

        w_list.append(wp)

    # Get errors
    for order in range (8):
        wp = w_list[order]

        # Training
        ytr_est_p = Ptrain_list[order] @ wp
        ytr_cls_p = [ [1 if y == max(x) else 0 for y in x] for x in ytr_est_p]

        ytr1 = np.matrix(Ytr)
        ytr2 = np.matrix(ytr_cls_p)
        difference_train = np.abs(ytr1 - ytr2)

        correct_p_train = np.where(~difference_train.any(axis=1))[0]
        incorrect_p_train = len(difference_train) - len(correct_p_train)
        error_train.append(incorrect_p_train)

        # Testing
        yt_est_p = Ptest_list[order] @ wp
        yt_cls_p = [ [1 if y == max(x) else 0 for y in x] for x in yt_est_p]

        yts1 = np.matrix(Yts)
        yts2 = np.matrix(yt_cls_p)
        difference_test = np.abs(yts1 - yts2)

        correct_p_test = np.where(~difference_test.any(axis=1))[0]
        incorrect_p_test = len(difference_test) - len(correct_p_test)
        error_test.append(incorrect_p_test)
    

    error_train_array = np.array(error_train)
    error_test_array = np.array(error_test)
    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array
