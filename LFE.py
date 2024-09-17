# %%
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler

# %%
data = np.loadtxt('traindata.txt')
test_data = np.loadtxt('testinputs.txt')

# %%
X_train_d = data[:, :-1] 
Y_train_d = data[:, -1]   

# %%
X_train_d.shape

# %%
X_train_d, Y_train_d = shuffle(X_train_d, Y_train_d, random_state=10)

# %%
#Feature engineering

def feature_engineering(X_train_d):
    X_train_df = [None] * len(X_train_d) 

    for data in range (0, len(X_train_d)):
        logs = []
        poly = []
        sin = []
        cos = []
        recip = []
        sqrt = []
        for i in X_train_d[data]:
            if i ==0:
                logs.append(0)
                recip.append(0)
            else:
                logs.append(np.log(i)) 
                recip.append(1/i)
            poly.append((i**2))   #Quadratic
            poly.append((i**3))   #Cubic
            sin.append(np.sin(i))
            cos.append(np.cos(i))
            sqrt.append(np.sqrt(i))  # Square root
    

            

        logs = np.array(logs, dtype=float)
        sin = np.array(sin, dtype=float)
        cos = np.array(cos, dtype=float)
        recip = np.array(recip, dtype=float)
        sqrt = np.array(sqrt, dtype=float)


        X_train_df[data] = np.concatenate([X_train_d[data], poly, logs[[0,4,7,3]], sin[[7]], cos[[4]], recip,sqrt])
        
    return np.array(X_train_df)


# %%
X_train_df = feature_engineering(X_train_d)

# %%
def Least_Squares(X, Y, P):

    Z =X
    Z = Z.T
    
    Z = np.array(Z,dtype=float)
    Y = np.array(Y,dtype=float)


    M = np.matmul(Z, Z.T)
    S = np.matmul(Z, Y)
    try:
        w = np.linalg.solve(M,S)
    except:
        print("Perfect correlation leading to unviability")
        return 0,0

    R = 1/len(X)*(np.sum(np.square([Y[i] - np.matmul(w,np.array(Z.T[i])) for i in range(0, len(X))])))

    return w, R

# %%
def tester(X, Y, w, P):
    Z = X
    Z = Z.T
    Z = np.array(Z,dtype=float)
    Y = np.array(Y,dtype=float)
    R = 1/len(X)*(np.sum(np.square([Y[i] - np.matmul(w,np.array(Z.T[i])) for i in range(0, len(X))])))
    return R

# %%
def Splitter(trunk_size,set, X, Y):
    
    test_start = (1+set-1)*trunk_size
    test_end = (set+1)*trunk_size

    #test data
    Y_test_data = Y[test_start:test_end]
    X_test_data = X[test_start:test_end]
    
    #train data
    try:
        X_train_data = np.concatenate((X[0:test_start], X[test_end:]), axis=0)
        Y_train_data = np.concatenate((Y[0:test_start], Y[test_end:]), axis=0)
    except  Exception as e:
        print(e)


    return X_train_data, Y_train_data, X_test_data, Y_test_data

# %%
def CrossValidation(P, K, X, Y):
    avg_train_loss = []
    avg_test_loss = []
    all_w = []
    trunk_size = int(len(X)/K)
    for set in range(K):
        X_train, Y_train, X_test, Y_test = Splitter(trunk_size, set, X, Y)
        w, R = Least_Squares(X_train, Y_train, P)
        if R == 0:
            raise NameError('Exception in least squares')
        avg_train_loss.append(R)
        all_w.append(w)
        R_test = tester(X_test, Y_test, w, P)
        avg_test_loss.append(R_test)

        min = avg_test_loss.index(np.min(avg_test_loss))

    return np.mean(avg_train_loss), np.mean(avg_test_loss), all_w[min]

# %%
Train_loss, Test_loss, W = CrossValidation(100, 10, X_train_df, Y_train_d)

# %%
print("The train loss is: ", Train_loss)
print("The test loss is: ", Test_loss)

# %%
X_test_df = feature_engineering(test_data)

# %%
def predict(X, w):
    return [np.matmul(w,np.array(X[i])) for i in range(0, len(X))]

# %%
prediction = predict(X_test_df, W)

# %%
import json

with open('output.txt', 'w') as filehandle:
    json.dump(prediction, filehandle)


