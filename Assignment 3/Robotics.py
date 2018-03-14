import scipy.io as spio
from hmmlearn import hmm
import numpy as np
from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# Comment to print concactenated array
np.set_printoptions(threshold=np.nan)

# Loading data files
nom = spio.loadmat('Pre-processed_variables.mat')
ano = spio.loadmat('Pre-processed_variables_M_ano.mat')

X_all = nom['All_Variable_1_Period'] 
X_sel= nom['Selected_Variable_1_Period'] 

Y_all = ano['All_Variable_1_Period'] 
Y_sel= ano['Selected_Variable_1_Period'] 

print(X_all.shape) #(3500,27)
print(X_sel.shape) #(3500,12)




# Fitting model to data
# n_components represents hidden states, which varies from 4, 10, 100, and 3500 for this case
model = GaussianHMM(n_components=10, covariance_type="diag", n_iter=10, random_state = 1).fit(X_all)

# X,Z = model.sample(3500)
# print(X)
# print(Z)

# Prints out the transition matrix
print('Transition matrix size:',model.transmat_.shape )

# Predict hidden states using trained model
hidden_states_nom = model.predict(X_all)
hidden_states_ano = model.predict(Y_all)

# log_prob_states,decoded_states_nom = model.decode(X_all)
# print('Log Probability:',log_prob_states)
# print(decoded_states_nom.shape)

print('hidden_states shape:', hidden_states_nom.shape)
# print('decoded_states_nom:', decoded_states_nom.shape)

# Saving to csv file
np.savetxt("hidden_states_nom(10,10,27).csv",hidden_states_nom,delimiter=",")
np.savetxt("hidden_states_ano(10,10,27).csv",hidden_states_ano,delimiter=",")
# np.savetxt("decoded_states_nom.csv",decoded_states_nom,delimiter=",")
