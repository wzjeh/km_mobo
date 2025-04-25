import sys
import torch
sys.path.append('..')
import numpy as np
from core import doe,plot,mobo
import time
from core.kinetic import Experiment
Y_all_list = []

# Define the constrain of reaction conditions
def reaction(X):
    X = np.array(X)
    Conditions = {'Time(s)': X[0], 'Temperature(C)':X[1], 'NB(mol/l)': X[2],  'NO2+(mol/l)': X[3],
                  'H2SO4(mol/l)' : X[4]}
    Y= Experiment(**Conditions)
    return Y # e-factor and Conversion
objective_func = reaction
# Define the design space
X_name = ['Time(s)', 'Temperature(C)','NB(mol/l)', 'NO2+(mol/l)','H2SO4(mol/l)']
Y_name = ['E_factor','Conversion']
n = len(X_name)
X_ranges =  [[0, 200],
             [60,82], 
             [0, 1],
             [0, 2],
             [15,18.4]]



# Latin hypercube design with 15 initial points
samples = 5
# Get the Get the initial experimental conditions
train_X = doe.lhs(samples, n) 
# Get initial results, optimize for two goal( e-factor and conversion) 
train_Y = doe.Y_lhs(train_X, objective_func, X_ranges,Y_obj = 2) 

Iterate = 15 # Number of iterations 15
n_candidates = 1 # Number of candidate points
start_time = time.time()
model = mobo.initialize_model(train_X,train_Y) # Gaussian process
data = mobo.bo(train_X,train_Y,model,X_ranges)
for i in range(Iterate):
    new_x,candidates = data.qNEHVI(n_candidates,obj1 = 1) # obj1 = 1 represent minimize e-factor
    Y_real,X_real,X_all, Y_all = data.update_training_points(new_x, candidates, objective_func, Y_obj = 2)
    Y_all_list.append(Y_all.clone())  # save Y_all in every iteration
end_time = time.time()
print('Optimizing the experiment takes {:.2f} minutes.'.format((end_time-start_time)/60))

# plt_max_observed(Y_all[:,1],samples,Iterate,n_candidates)
# plot.plt_max_observed(Y_all,samples,Iterate,n_candidates,'progression plot.png')
# Plot the Pareto Front
plot.pareto_front(Y_all, samples, Iterate, n_candidates,'pareto_bo.png')

torch.save(Y_all_list, 'Y_all_iter.pt')  # export to hypervolume.py 