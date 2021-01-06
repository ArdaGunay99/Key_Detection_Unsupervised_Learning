# Importing packages
import csv
import matplotlib.pyplot as plt
import numpy as num
import sklearn as sk
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
import warnings

#warnings.filterwarnings("ignore")


# function to element-vise divide two matrices that may contain zeros
def divide_matrices(A, B):
    C = num.copy(A)
    for a, b, c in num.nditer([A, B, C], op_flags=['readwrite']):
        if (b != 0):
            c = a / b

    return C


def simlin_coef(age_list, exp_list):
    ageAvg = num.average(age_list)
    expAvg = num.average(exp_list)

    b1Nominator = 0
    for i in range(len(age_list)):
        b1Nominator += (age_list[i] - ageAvg) * (exp_list[i] - expAvg)
    b1Denominator = 0
    for i in range(len(age_list)):
        b1Denominator += num.square(age_list[i] - ageAvg)
    b1 = b1Nominator / b1Denominator
    b0 = expAvg - (b1 * ageAvg)
    coefs = [b1, b0]

    return coefs


def calculate_adjRsquare(y, y_pred, d):
    RSS = 0
    TSS = 0
    n = len(y)
    yAvg = num.average(y)
    for i in range(len(y)):
        RSS += num.square(y[i] - y_pred[i])
    for i in range(len(y)):
        TSS += num.square(y[i] - yAvg)
        nominator = RSS / (n - d - 1)
        denominator = TSS / (n - 1)
    adjRsquare = 1 - (nominator / denominator)
    return adjRsquare


def calculate_Rsquare(actual_exp, prediction):
    RSS = 0
    TSS = 0
    expAvg = num.mean(actual_exp)
    for i in range(len(actual_exp)):
        RSS += num.square(actual_exp[i] - prediction[i])
    for i in range(len(actual_exp)):
        TSS += num.square(actual_exp[i] - expAvg)

    Rsquare = 1 - (RSS / TSS)
    return Rsquare


def simlin_plot(age_list_1, exp_list_1, age_list_2, exp_list_2, coefs1, coefs2):
    exp_pred_1 = []
    for i in range(len(age_list_1)):
        exp_pred_1.append((coefs2[0] * age_list_1[i]) + coefs2[1])
    exp_pred_2 = []
    for i in range(len(age_list_2)):
        exp_pred_2.append((coefs1[0] * age_list_2[i]) + coefs1[1])

    print("R^2 score for exp list 1: ")
    print(calculate_Rsquare(exp_list_1, exp_pred_1))
    print("R^2 score for exp list 2:")
    print(calculate_Rsquare(exp_list_2, exp_pred_2))

    plt.figure()
    plt.plot(age_list_1, exp_pred_1, c='r')
    plt.scatter(age_list_1, exp_list_1, c="b")
    plt.ylabel("Experience")
    plt.xlabel("Player age")
    plt.figure()
    plt.plot(age_list_2, exp_pred_2, c='r')
    plt.scatter(age_list_2, exp_list_2, c="b")
    plt.ylabel("Experience")
    plt.xlabel("Player age")

    plt.show()

def coefficients(x,y):
    tdot = num.dot(x.T,x)
    a_equation = num.linalg.inv(tdot)
    a_dot = num.dot(a_equation,x.T)
    result = num.dot(a_dot,y)
    return result
def k_fold_cv(X,y):
    cv_hat = num.array([])
    temp = 0
    for i in range(0, num.size(X)/10, 1):

            if num.size(X) - (i*10) < 10:
                temp = i
                break
            else:
                X_training = num.delete(X, range(i * 10, (i * 10) + 10), 0)
                X_testing = X[range(i * 10, (i * 10) + 10)]
                test_y = num.delete(y, range(i * 10, (i * 10) + 10), 0)
                coefficients_from_multi_lin = coefficients(X_training,test_y)
                cv_hat = num.append(cv_hat, num.dot(X_testing,coefficients_from_multi_lin))
    # 10th iteration is outside the loop
    X_training = num.delete(X, range(temp * 10, num.size(X), 0))
    X_testing = X[range(temp * 10, num.size(X))]
    test_y = num.delete(y, range(temp * 10, num.size(X), 0))
    coefficients_from_multi_lin = coefficients(X_training, test_y)
    cv_hat = num.append(cv_hat, num.dot(X_testing, coefficients_from_multi_lin))
    return cv_hat

def multiple_lin_coefs(X_transpose, X, y):
    coefs = num.array([])
    XTdotX = num.dot(X_transpose, X)
    inverted_XTdotX = num.linalg.inv(XTdotX)
    inverted_dot_transpose = num.dot(inverted_XTdotX, X_transpose)
    coefs = num.append(coefs, num.dot(inverted_dot_transpose, y))
    return coefs


def cross_validation(X, salary_list):
    scv_hat = num.array([])
    for i in range(0, 10, 1):
        X_train = num.delete(X, range(i * 4, (i * 4) + 4), 0)

        X_test = X[range(i * 4, (i * 4) + 4)]

        test_salary = num.delete(salary_list, range(i * 4, (i * 4) + 4), 0)

        coefs = multiple_lin_coefs(X_train.transpose(), X_train, test_salary)
        scv_hat = num.append(scv_hat, num.dot(X_test, coefs))
    return scv_hat


def calculate_MSE(y_hat, y):
    MSE = 0
    for i in range(len(y)):
        MSE += (y_hat[i] - y[i]) ** 2
    MSE = MSE / len(y)
    return MSE


def cubic_spline(x, y, knots):
    ones = num.array([])
    for i in range(len(x)):
        ones = num.append(ones, int(1))
    xSquare = num.square(x)
    xCube = num.power(x, 3)
    X = num.vstack((ones, x, xSquare, xCube))
    for i in range(len(knots)):
        knotColumn = num.array([])
        for j in range(len(x)):
            value = 0
            if ((x[j] - knots[i]) > 0):
                value = (x[j] - knots[i]) ** 3

            knotColumn = num.append(knotColumn, int(value))

        X = num.vstack((X, knotColumn))
    X = X.transpose()

    sortedX = X[num.argsort(X[:, 1])]
    y = y[X[:, 1].argsort()]

    coefs = multiple_lin_coefs(sortedX.transpose(), sortedX, y)
    line = num.dot(sortedX, coefs)
    sortedx = sortedX[:, 1]
    return line, sortedx


def lasso_reg(X, y):
    lambdas = num.arange(0.1, 10.1, 0.1)
    MSE_values = num.array([])
    for i in range(len(lambdas)):
        lasso = linear_model.Lasso(alpha=lambdas[i])
        sum_of_errors = 0
        for j in range(len(X)):
            X_test = X[j]
            y_test = y[j]
            X_train = num.delete(X, j, axis=0)
            y_train = num.delete(y, j, axis=0)
            lasso.fit(X_train, y_train)

            X_test = X_test.reshape(1, -1)

            pred = lasso.predict(X_test)
            sum_of_errors += (y_test - pred) ** 2
        MSE = sum_of_errors / len(X)
        MSE_values = num.append(MSE_values, float(MSE))
    MSE_minimum = num.amin(MSE_values)
    MSE_minimum_pos = num.argmin(MSE_values)
    lambda_min = lambdas[MSE_minimum_pos]
    lasso = linear_model.Lasso(alpha=lambda_min)
    lasso.fit(X, y)
    predictions = lasso.predict(X)
    return predictions


def random_forest(X, y):
    X_train = X[0:900]
    y_train = y[0:900]
    X_test = X[900::]
    y_test = y[900::]
    autoR = num.array([])
    sqrtR = num.array([])
    fourR = num.array([])

    # for i in range(1, 151):
    #     reg = RandomForestRegressor(max_depth=7, n_estimators=i, max_features="auto")
    #     reg.fit(X_train, y_train)
    #     pred_y = reg.predict(X_test)
    #
    #     autoR = num.append(autoR, calculate_Rsquare(y_test, pred_y))
    #
    #     reg = RandomForestRegressor(max_depth=7, n_estimators=i, max_features="sqrt")
    #     reg.fit(X_train, y_train)
    #     pred_y = reg.predict(X_test)
    #     sqrtR = num.append(sqrtR, calculate_Rsquare(y_test, pred_y))
    #
    #     reg = RandomForestRegressor(max_depth=7, n_estimators=i, max_features=4)
    #     reg.fit(X_train, y_train)
    #     pred_y = reg.predict(X_test)
    #     fourR = num.append(fourR, calculate_Rsquare(y_test, pred_y))

    reg = RandomForestRegressor(max_depth=7, n_estimators=150, max_features="sqrt")
    reg.fit(X_train, y_train)
    y_pred_1 = reg.predict(X)

    # reg = RandomForestRegressor(max_depth=1, n_estimators=150, max_features=4)
    # reg.fit(X_train, y_train)
    # y_pred_2 = reg.predict(X_test)

    return y_pred_1


def SVM(X, y):
    X_train = X[0:900]
    y_train = y[0:900]
    X_test = X[900::]
    y_test = y[900::]
    # linearZeroCorrect = num.array([])
    # linearZeroIncorrect = num.array([])
    # linearOneIncorrect = num.array([])
    # linearOneCorrect = num.array([])
    # linearZeroCorrectResult = num.array([])
    # linearZeroIncorrectResult = num.array([])
    # linearOneIncorrectResult = num.array([])
    # linearOneCorrectResult = num.array([])
    # polyZeroCorrect = num.array([])
    # polyZeroIncorrect = num.array([])
    # polyOneIncorrect = num.array([])
    # polyOneCorrect = num.array([])
    # polyZeroCorrectResult = num.array([])
    # polyZeroIncorrectResult = num.array([])
    # polyOneIncorrectResult = num.array([])
    # polyOneCorrectResult = num.array([])
    # rbfZeroCorrect = num.array([])
    # rbfZeroIncorrect = num.array([])
    # rbfOneIncorrect = num.array([])
    # rbfOneCorrect = num.array([])
    # rbfZeroCorrectResult = num.array([])
    # rbfZeroIncorrectResult = num.array([])
    # rbfOneIncorrectResult = num.array([])
    # rbfOneCorrectResult = num.array([])
    #
    # svm = SVC(kernel='linear')
    # svm.fit(X_train, y_train)
    # pred_y = svm.predict(X_test)
    # for i in range(len(pred_y)):
    #     if (pred_y[i] == 0):
    #         if (y_test[i] == 0):
    #             linearZeroCorrect = num.append(linearZeroCorrect, i)
    #         if (y_test[i] == 1):
    #             linearZeroIncorrect = num.append(linearZeroIncorrect, i)
    #     if (pred_y[i] == 1):
    #         if (y_test[i] == 0):
    #             linearOneIncorrect = num.append(linearOneIncorrect, i)
    #         if (y_test[i] == 1):
    #             linearOneCorrect = num.append(linearOneCorrect, i)
    #
    # row = X_test[int(linearZeroCorrect[0])]
    # linearZeroCorrectResult = row
    # for i in range(1, len(linearZeroCorrect), 1):
    #     row = X_test[int(linearZeroCorrect[i])]
    #     print("row is :", row)
    #     linearZeroCorrectResult = num.vstack((linearZeroCorrectResult, row))
    # print(linearZeroCorrectResult)
    #
    # row = X_test[int(linearZeroIncorrect[0])]
    # linearZeroIncorrectResult = row
    # for i in range(1, len(linearZeroIncorrect), 1):
    #     row = X_test[int(linearZeroIncorrect[i])]
    #     print("row is :", row)
    #     linearZeroIncorrectResult = num.vstack((linearZeroIncorrectResult, row))
    # print(linearZeroIncorrectResult)
    #
    # row = X_test[int(linearOneCorrect[0])]
    # linearOneCorrectResult = row
    # for i in range(1, len(linearOneCorrect), 1):
    #     row = X_test[int(linearOneCorrect[i])]
    #     print("row is :", row)
    #     linearOneCorrectResult = num.vstack((linearOneCorrectResult, row))
    # print(linearOneCorrectResult)
    #
    # row = X_test[int(linearOneIncorrect[0])]
    # linearOneIncorrectResult = row
    # for i in range(1, len(linearOneIncorrect), 1):
    #     row = X_test[int(linearOneIncorrect[i])]
    #     print("row is :", row)
    #     linearOneIncorrectResult = num.vstack((linearOneIncorrectResult, row))
    # print(linearOneIncorrectResult)
    #
    # svm = SVC(kernel='poly')
    # svm.fit(X_train, y_train)
    # pred_y = svm.predict(X_test)
    # for i in range(len(pred_y)):
    #     if (pred_y[i] == 0):
    #         if (y_test[i] == 0):
    #             polyZeroCorrect = num.append(polyZeroCorrect, i)
    #         if (y_test[i] == 1):
    #             polyZeroIncorrect = num.append(polyZeroIncorrect, i)
    #     if (pred_y[i] == 1):
    #         if (y_test[i] == 0):
    #             polyOneIncorrect = num.append(polyOneIncorrect, i)
    #         if (y_test[i] == 1):
    #             polyOneCorrect = num.append(polyOneCorrect, i)
    # print("poly one correct is: ", polyOneCorrect)
    # row = X_test[int(polyZeroCorrect[0])]
    # polyZeroCorrectResult = row
    # for i in range(1, len(polyZeroCorrect), 1):
    #     row = X_test[int(polyZeroCorrect[i])]
    #     print("row is :", row)
    #     polyZeroCorrectResult = num.vstack((polyZeroCorrectResult, row))
    # print(polyZeroCorrectResult)
    #
    # row = X_test[int(polyZeroIncorrect[0])]
    # polyZeroIncorrectResult = row
    # for i in range(1, len(polyZeroIncorrect), 1):
    #     row = X_test[int(polyZeroIncorrect[i])]
    #     print("row is :", row)
    #     polyZeroIncorrectResult = num.vstack((polyZeroIncorrectResult, row))
    # print(polyZeroIncorrectResult)
    #
    # # row = X_test[int(polyOneCorrect[0])]
    # # polyOneCorrectResult = row
    # # for i in range(1, len(polyOneCorrect), 1):
    # #     row = X_test[int(polyOneCorrect[i])]
    # #     print("row is :", row)
    # #     polyOneCorrectResult = num.vstack((polyOneCorrectResult, row))
    # # print(polyOneCorrectResult)
    #
    # row = X_test[int(polyOneIncorrect[0])]
    # polyOneIncorrectResult = row
    # for i in range(1, len(polyOneIncorrect), 1):
    #     row = X_test[int(polyOneIncorrect[i])]
    #     print("row is :", row)
    #     polyOneIncorrectResult = num.vstack((polyOneIncorrectResult, row))
    # print(polyOneIncorrectResult)

    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    pred_y_rad = svm.predict(X)
    # for i in range(len(pred_y)):
    #     if (pred_y[i] == 0):
    #         if (y_test[i] == 0):
    #             rbfZeroCorrect = num.append(rbfZeroCorrect, i)
    #         if (y_test[i] == 1):
    #             rbfZeroIncorrect = num.append(rbfZeroIncorrect, i)
    #     if (pred_y[i] == 1):
    #         if (y_test[i] == 0):
    #             rbfOneIncorrect = num.append(rbfOneIncorrect, i)
    #         if (y_test[i] == 1):
    #             rbfOneCorrect = num.append(rbfOneCorrect, i)
    # print("rbf one correct is: ", rbfOneCorrect)
    # row = X_test[int(rbfZeroCorrect[0])]
    # rbfZeroCorrectResult = row
    # for i in range(1, len(rbfZeroCorrect), 1):
    #     row = X_test[int(rbfZeroCorrect[i])]
    #     print("row is :", row)
    #     rbfZeroCorrectResult = num.vstack((rbfZeroCorrectResult, row))
    # print(rbfZeroCorrectResult)
    #
    # row = X_test[int(rbfZeroIncorrect[0])]
    # rbfZeroIncorrectResult = row
    # for i in range(1, len(rbfZeroIncorrect), 1):
    #     row = X_test[int(rbfZeroIncorrect[i])]
    #     print("row is :", row)
    #     rbfZeroIncorrectResult = num.vstack((rbfZeroIncorrectResult, row))
    # print(rbfZeroIncorrectResult)
    #
    # row = X_test[int(rbfOneCorrect[0])]
    # rbfOneCorrectResult = row
    # for i in range(1, len(rbfOneCorrect), 1):
    #     row = X_test[int(rbfOneCorrect[i])]
    #     print("row is :", row)
    #     rbfOneCorrectResult = num.vstack((rbfOneCorrectResult, row))
    # print(rbfOneCorrectResult)
    #
    # row = X_test[int(rbfOneIncorrect[0])]
    # rbfOneIncorrectResult = row
    # for i in range(1, len(rbfOneIncorrect), 1):
    #     row = X_test[int(rbfOneIncorrect[i])]
    #     print("row is :", row)
    #     rbfOneIncorrectResult = num.vstack((rbfOneIncorrectResult, row))
    # print(rbfOneIncorrectResult)
    return pred_y_rad


ones = num.array([])

c = num.array([])
cSharp = num.array([])
d = num.array([])
dSharp = num.array([])
e = num.array([])
f = num.array([])
fSharp = num.array([])
g = num.array([])
gSharp = num.array([])
a = num.array([])
aSharp = num.array([])
b = num.array([])

y = num.array([])

with open("C:/Users/Arda/PycharmProjects/CE475/key_exists.csv", encoding="Latin-1") as z:
    csv_list1 = list(csv.reader(z))

with open("C:/Users/Arda/PycharmProjects/CE475/occurrence.csv", encoding="Latin-1") as z:
    csv_list2 = list(csv.reader(z))

with open("C:/Users/Arda/PycharmProjects/CE475/duration.csv", encoding="Latin-1") as z:
    csv_list3 = list(csv.reader(z))

for row in csv_list1:
    if row != csv_list1[0]:
        c = num.append(c, float(row[1]))
        cSharp = num.append(cSharp, float(row[2]))
        d = num.append(d, float(row[3]))
        dSharp = num.append(dSharp, float(row[4]))
        e = num.append(e, float(row[5]))
        f = num.append(f, float(row[6]))
        fSharp = num.append(fSharp, float(row[7]))
        g = num.append(g, float(row[8]))
        gSharp = num.append(gSharp, float(row[9]))
        a = num.append(a, float(row[10]))
        aSharp = num.append(aSharp, float(row[11]))
        b = num.append(b, float(row[12]))
        y = num.append(y, int(row[13]))

for i in range(len(a)):
    ones = num.append(ones, int(1))

key_exists_lin = num.vstack((ones, c, cSharp, d, dSharp, e, f, fSharp, g, gSharp, a, aSharp, b))
key_exists_lin = key_exists_lin.transpose()
key_exists = num.vstack((c, cSharp, d, dSharp, e, f, fSharp, g, gSharp, a, aSharp, b))
key_exists = key_exists.transpose()

c1 = num.array([])
cSharp1 = num.array([])
d1 = num.array([])
dSharp1 = num.array([])
e1 = num.array([])
f1 = num.array([])
fSharp1 = num.array([])
g1 = num.array([])
gSharp1 = num.array([])
a1 = num.array([])
aSharp1 = num.array([])
b1 = num.array([])

for row in csv_list2:
    if row != csv_list2[0]:
        c1 = num.append(c1, float(row[1]))
        c1Sharp = num.append(cSharp1, float(row[2]))
        d1 = num.append(d1, float(row[3]))
        dSharp1 = num.append(dSharp1, float(row[4]))
        e1 = num.append(e1, float(row[5]))
        f1 = num.append(f1, float(row[6]))
        fSharp1 = num.append(fSharp1, float(row[7]))
        g1 = num.append(g1, float(row[8]))
        gSharp1 = num.append(gSharp1, float(row[9]))
        a1 = num.append(a1, float(row[10]))
        aSharp1 = num.append(aSharp1, float(row[11]))
        b1 = num.append(b1, float(row[12]))




occurrence_lin = num.vstack((ones, c, cSharp, d, dSharp, e, f, fSharp, g, gSharp, a, aSharp, b))
occurrence_lin = occurrence_lin.transpose()
occurrence = num.vstack((c, cSharp, d, dSharp, e, f, fSharp, g, gSharp, a, aSharp, b))
occurrence = occurrence.transpose()

c2 = num.array([])
cSharp2 = num.array([])
d2 = num.array([])
dSharp2 = num.array([])
e2 = num.array([])
f2 = num.array([])
fSharp2 = num.array([])
g2 = num.array([])
gSharp2 = num.array([])
a2 = num.array([])
aSharp2 = num.array([])
b2 = num.array([])

for row in csv_list3:
    if row != csv_list3[0]:
        c2 = num.append(c2, float(row[1]))
        cSharp2 = num.append(cSharp2, float(row[2]))
        d2 = num.append(d2, float(row[3]))
        dSharp2 = num.append(dSharp2, float(row[4]))
        e2 = num.append(e2, float(row[5]))
        f2 = num.append(f2, float(row[6]))
        fSharp2 = num.append(fSharp2, float(row[7]))
        g2 = num.append(g2, float(row[8]))
        gSharp2 = num.append(gSharp2, float(row[9]))
        a2 = num.append(a2, float(row[10]))
        aSharp2 = num.append(aSharp2, float(row[11]))
        b2 = num.append(b2, float(row[12]))




duration_lin = num.vstack((ones, c, cSharp, d, dSharp, e, f, fSharp, g, gSharp, a, aSharp, b))
duration_lin = duration_lin.transpose()
duration = num.vstack((c, cSharp, d, dSharp, e, f, fSharp, g, gSharp, a, aSharp, b))
duration = duration.transpose()

errors = num.array([])

occurrence_over_duration = divide_matrices(occurrence, duration)
duration_over_occurrence = divide_matrices(duration, occurrence)


key_exists_mult_lin_coefs = multiple_lin_coefs(key_exists_lin.transpose(), key_exists_lin, y)
key_exists_mult_lin_predictions = num.dot(key_exists_lin,key_exists_mult_lin_coefs)
key_exists_mult_lin_err = calculate_MSE(key_exists_mult_lin_predictions, y)
errors = num.append(errors, key_exists_mult_lin_err)

occurrence_mult_lin_coefs = multiple_lin_coefs(occurrence_lin.transpose(), occurrence_lin, y)
occurrence_mult_lin_predictions = num.dot(occurrence_lin, occurrence_mult_lin_coefs)
occurrence_mult_lin_err = calculate_MSE(occurrence_mult_lin_predictions, y)
errors = num.append(errors, occurrence_mult_lin_err)


duration_mult_lin_coefs = multiple_lin_coefs(duration_lin.transpose(), duration_lin, y)
duration_mult_lin_predictions = num.dot(duration_lin, duration_mult_lin_coefs)
duration_mult_lin_err = calculate_MSE(duration_mult_lin_predictions, y)
errors = num.append(errors, duration_mult_lin_err)


key_exists_lasso_predictions = lasso_reg(key_exists_lin, y)
key_exists_lasso_err = calculate_MSE(key_exists_lasso_predictions, y)
errors = num.append(errors, key_exists_lasso_err)


occurrence_lasso_predictions = lasso_reg(occurrence_lin, y)
occurrence_lasso_err = calculate_MSE(occurrence_lasso_predictions, y)
errors = num.append(errors, occurrence_lasso_err)


duration_lasso_predictions = lasso_reg(duration_lin, y)
duration_lasso_err = calculate_MSE(duration_lasso_predictions, y)
errors = num.append(errors, duration_lasso_err)


key_exists_RF_predictions = random_forest(key_exists, y)
key_exists_RF_err = calculate_MSE(key_exists_RF_predictions, y)
errors = num.append(errors, key_exists_RF_err)


occurrence_RF_predictions = random_forest(occurrence, y)
occurrence_RF_err = calculate_MSE(occurrence_RF_predictions, y)
errors = num.append(errors, occurrence_RF_err)


duration_RF_predictions = random_forest(duration, y)
duration_RF_err = calculate_MSE(duration_RF_predictions, y)
errors = num.append(errors, duration_RF_err)


dur_over_occ_RF_predictions = random_forest(duration_over_occurrence, y)
dur_over_occ_RF_err = calculate_MSE(dur_over_occ_RF_predictions, y)
errors = num.append(errors, dur_over_occ_RF_err)


occ_over_dur_RF_predictions = random_forest(occurrence_over_duration, y)
occ_over_dur_RF_err = calculate_MSE(occ_over_dur_RF_predictions, y)
errors = num.append(errors, occ_over_dur_RF_err)


key_exists_SVM_predictions = SVM(key_exists, y)
key_exists_SVM_err = calculate_MSE(key_exists_SVM_predictions, y)
errors = num.append(errors, key_exists_SVM_err)


occurrence_SVM_predictions = SVM(occurrence, y)
occurrence_SVM_err = calculate_MSE(occurrence_SVM_predictions, y)
errors = num.append(errors, occurrence_SVM_err)


duration_SVM_predictions = SVM(duration, y)
duration_SVM_err = calculate_MSE(duration_SVM_predictions, y)
errors = num.append(errors, duration_SVM_err)


dur_over_occ_SVM_predictions = SVM(duration_over_occurrence, y)
dur_over_occ_SVM_err = calculate_MSE(dur_over_occ_SVM_predictions, y)
errors = num.append(errors, dur_over_occ_SVM_err)


occ_over_dur_SVM_predictions = SVM(occurrence_over_duration, y)
occ_over_dur_SVM_err = calculate_MSE(occ_over_dur_SVM_predictions, y)
errors = num.append(errors, occ_over_dur_SVM_err)

min_error = num.amin(errors)
min_error_position = num.argmin(errors)

print("Errors: ", errors)
print("minimum: ", min_error)
print("minimum position: ", min_error_position)
print("minimum errorred predictions: ", occ_over_dur_RF_predictions)

num.savetxt("best_results.csv", occ_over_dur_RF_predictions, delimiter=",")






# X = num.vstack((ACE1,ACE2))
# X = X.transpose()
# linearZeroCorrectResult,linearZeroIncorrectResult,linearOneCorrectResult,linearOneIncorrectResult,rbfOneCorrectResult,rbfOneIncorrectResult,rbfZeroCorrectResult,rbfZeroIncorrectResult,polyOneCorrectResult,polyOneIncorrectResult,polyZeroCorrectResult,polyZeroIncorrectResult = SVM(X, y)

# plt.figure()
# plt.title("SVC with linear kernel")
# plt.scatter(linearZeroCorrectResult[:,0], linearZeroCorrectResult[:,1], label="first player correct ", c="r")
# plt.scatter(linearZeroIncorrectResult[:,0], linearZeroIncorrectResult[:,1],label="first player incorrect ", c="b")
# plt.scatter(linearOneIncorrectResult[:,0], linearOneIncorrectResult[:,1],label="second player incorrect ", c="y")
# plt.scatter(linearOneCorrectResult[:,0], linearOneCorrectResult[:,1],label="second player correct ", c="k")
# plt.legend()

# plt.figure()
# plt.title("SVC with polinomial kernel")
# plt.scatter(polyZeroCorrectResult[:,0], polyZeroCorrectResult[:,1],label="first player correct ", c="r")
# plt.scatter(polyZeroIncorrectResult[:,0], polyZeroIncorrectResult[:,1],label="first player incorrect ", c="b")
# plt.scatter(polyOneIncorrectResult[0], polyOneIncorrectResult[1],label="second player incorrect ", c="y")
# plt.legend()

# plt.figure()
# plt.title("SVC with radial kernel")
# plt.scatter(rbfZeroCorrectResult[:,0], rbfZeroCorrectResult[:,1],label="first player correct ", c="r")
# plt.scatter(rbfZeroIncorrectResult[:,0], rbfZeroIncorrectResult[:,1],label="first player incorrect ", c="b")
# plt.scatter(rbfOneIncorrectResult[0], rbfOneIncorrectResult[1],label="second player incorrect ", c="y")
# plt.scatter(rbfOneCorrectResult[0], rbfOneCorrectResult[1],label="second player correct ", c="k")
# plt.legend()

# plt.show()

# autoR, sqrtR, fourR, y_pred_1, y_pred_2, y_test = random_forest(X, y)
# simple = num.arange(1, 151, 1)

# plt.figure()
# plt.plot(simple, autoR, c="r")
# plt.plot(simple, sqrtR, c="b")
# plt.plot(simple, fourR, c="y")


# error1 = y_test - y_pred_1
# for i in range(len(y_test)):
# errors1 = num.append(error1, num.square(y_test[i] - y_pred_1[i]))

# error2 = y_test - y_pred_2
# for i in range(len(y_test)):
# errors2 = num.append(error2, num.square(y_test[i] - y_pred_2[i]))
# zy = num.array([])
# for i in range (len(y_test)):
#   zy = num.append(zy, 0)
# zx = num.array([])
# for i in range (len(y_test)):
#   zx = num.append(zx, i)


# plt.figure()
# plt.scatter(y_pred_1, error1, c="r")
# plt.scatter(y_pred_2, error2, c="b")
# plt.plot(zx, zy, c="k")

# plt.show()


# MSE_values = lasso_reg(X,TPW1,lambdas)
# MSE_minimum = num.amin(MSE_values)
# MSE_minimum_pos = num.argmin(MSE_values)
# lambda_min = lambdas[MSE_minimum_pos]
# print("the lambda value ")
# print(lambda_min)
# print("yields the minimum error of ")
# print(MSE_minimum)

# lasso = linear_model.Lasso(alpha=lambda_min)
# lasso.fit(X, TPW1)
# print("The lasso coefficients with the optimal lambda value: ")
# print(lasso.coef_)

# lasso = linear_model.Lasso(alpha=0)
# lasso.fit(X, TPW1)
# print("Regular least squares coefficients: ")
# print(lasso.coef_)


# plt.figure()
# plt.plot(lambdas,MSE_values)
# plt.ylabel("MSE VALUES")
# plt.xlabel("LAMBDA VALUES")
# plt.show()
# knots1 = num.array([55, 65, 70])
# knots2 = num.array([60, 75])
# knots3 = num.array([62])

# line1,sortedx = cubic_spline(x, y, knots1)
# line2,sortedx = cubic_spline(x, y, knots2)
# line3,sortedx = cubic_spline(x, y, knots3)

# plt.figure()
# plt.scatter(x, y, c="b")
# plt.plot(sortedx, line1, c="k")
# plt.plot(sortedx, line2, c="y")
# plt.plot(sortedx, line3, c="g")
# plt.ylabel("y")
# plt.xlabel("x")
# plt.show()

# age_list = num.append(age_list, int(row[4]))
# exp_list = num.append(exp_list, int(row[6]))
# pow_list = num.append(pow_list, float(row[7]))
# salary_list = num.append(salary_list, int(row[8]))

# for i in range(len(age_list)):
#   ones = num.append(ones, int(1))

# X = num.vstack((ones, age_list, exp_list, pow_list))
# X = X.transpose()
# column0 = X[:, 0]
# yMean = num.mean(salary_list)
# M0 = column0*yMean
# adjRsquare_of_M0 = calculate_adjRsquare(salary_list, M0, 0)
# print("adjusted R^2 of M0 is: ")
# print(adjRsquare_of_M0)

# Rsquares_of_variables = num.array([])

# for i in range(1, 4):
#   column_i = X[:, i]
#  tempMatrix = num.vstack((column0, column_i))
# tempMatrix = tempMatrix.transpose()
# coefs = multiple_lin_coefs(tempMatrix.transpose(), tempMatrix, salary_list)
# predicted_y = num.dot(tempMatrix, coefs)
# Rsquare = calculate_Rsquare(salary_list, predicted_y)
# Rsquares_of_variables = num.append(Rsquares_of_variables, Rsquare)


# best_Rsquare = num.argmax(Rsquares_of_variables) + 1
# best_Rsquare_value = num.max(Rsquares_of_variables)
# print("best R^2 is: ")
# print(best_Rsquare)
# print("value of this R^2 is: ")
# print(best_Rsquare_value)
# column_best = X[:, best_Rsquare]
# newMatrix = num.vstack((column0, column_best))
# newMatrix = newMatrix.transpose()
# coefs = multiple_lin_coefs(newMatrix.transpose(), newMatrix, salary_list)
# predicted_y = num.dot(newMatrix, coefs)

# new_adjRsquare = calculate_adjRsquare(salary_list, predicted_y, 1)
# print("new adjusted R^2 value is: ")
# print(new_adjRsquare)
# cv_hat = cross_validation(X, salary_list)
# y_hat = num.dot(X, multiple_lin_coefs(X.T, X, salary_list))


# MSE_cross = calculate_MSE(cv_hat, salary_list)
# MSE_noCross = calculate_MSE(y_hat, salary_list)

# errors_for_cross = num.array([])
# for i in range(len(salary_list)):
#   errors_for_cross = num.append(errors_for_cross, num.absolute(salary_list[i] - cv_hat[i]))
# errors_for_noCross = num.array([])
# for i in range(len(salary_list)):
#   errors_for_noCross = num.append(errors_for_noCross, num.absolute(salary_list[i] - y_hat[i]))

# plt.figure()
# plt.scatter(cv_hat, errors_for_cross, c="b")
# plt.scatter(y_hat, errors_for_noCross, c="r")
# plt.ylabel("error margins")
# plt.xlabel("predicted y values")


# plt.figure()
# plt.scatter(y_hat, cv_hat, c="b")
# for x=y plot
# plt.plot(y_hat, y_hat, c="r")
# plt.ylabel("cv_hat")
# plt.xlabel("y_hat")
# plt.show()


# print("MSE with cross:")
# print(MSE_cross)
# print("Mse without cross:")
# print(MSE_noCross)

# errors = num.array([])
# for i in range(len(salary_list)):
#   errors = num.append(errors, num.absolute(salary_list[i]-predicted_y[i]))

# X_transpose = X.transpose()
# coefs = multiple_lin_coefs(X_transpose, X, salary_list)

# predicted_y = num.dot(X, coefs)
# errors = num.array([])
# for i in range(len(salary_list)):
#   errors = num.append(errors, num.absolute(salary_list[i]-predicted_y[i]))

# plt.figure()
# plt.scatter(predicted_y, errors, c="b")
# plt.ylabel("error margins")
# plt.xlabel("predicted y values")
# plt.show()

# for R^2:
# random_list = num.array([])
# for i in range(len(age_list)):
#    random_list = num.append(random_list, int(num.random.randint(-500, 500)))

# X_with_rand = num.transpose(X)
# X_with_rand = num.vstack((ones, age_list, exp_list, pow_list, random_list))
# X_with_rand = num.transpose(X_with_rand)


# print("Showing original result: ")
# print(calculate_Rsquare(salary_list, predicted_y))

# calculating coefs with new random column:
# coefs_with_rand = multiple_lin_coefs(X_with_rand.transpose(), X_with_rand, salary_list)
# predicted_y_with_rand = num.dot(X_with_rand, coefs_with_rand)

# print("Showing results with an added random column: ")
# print(calculate_Rsquare(salary_list, predicted_y_with_rand))


# team1Coefs = simlin_coef(age_list_1,exp_list_1)
# team2Coefs = simlin_coef(age_list_2,exp_list_2)

# m1 = team1Coefs[0]
# b1 = team1Coefs[1]
# m2 = team2Coefs[0]
# b2 = team2Coefs[1]

# simlin_plot(age_list_1,exp_list_1,age_list_2,exp_list_2,team1Coefs,team2Coefs)
