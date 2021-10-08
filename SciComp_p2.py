import numpy as np
import pandas as pd
from chladni_show import *
from least_squares import *


#get the a matrix representation K of the biharmonic operator
Kmat = np.load("Chladni-Kmat.npy")
#get a random vector x0
np.random.seed(42)
x0 = np.random.rand(len(Kmat),)


##########################################
#a)1.
##########################################
def gershgorin(A):
    """Takes Square Matrix, returns centers and
    radisu of a Gershgorin disc a Gershgorin disc"""
    #only real values
    A = A.real
    center_list = []
    radius_list = []
    for i in range(len(A)):
        #values in the diagonal are the centers of the rings
        center = A[i][i]
        center_list.append(center)

        radius_array = A[i]
        #sum all the absolute values of the rows and subtstract the value in the diagaonal
        row_sum = np.sum(np.absolute(radius_array))
        radius = row_sum - np.absolute(center)
        radius_list.append(radius)
    return center_list, radius_list

def eigen_values_range(center, radii):
    eigen_value_min = []
    eigen_value_max = []
    for i in range(len(center)):

      upper = center[i] + radii[i]
      lower = center[i] - radii[i]
      eigen_value_max.append(upper)
      eigen_value_min.append(lower)

    return eigen_value_min, eigen_value_max
##########################################
#a)2.
##########################################
centers, radii = gershgorin(Kmat)
eigen_value_min, eigen_value_max = eigen_values_range(centers, radii)
df = pd.DataFrame({'Center':centers,'Radii':radii,'Min_eigen':eigen_value_min,'Max_eigen':eigen_value_max})

# uncomment line bellow to save for report
#df.to_csv('gershgorin.csv',index=False)

#upper_eigen_bound , lower_eigen_bound = eigen_values_range(centers, radii)
print(f"""Acording with Gershgorin’s theorem the eigen values are
in the union of all the rings, in the complex plane.
Assuming all the eigen values are real and using Gershgorin theorem
we can find the eigen values are in the diameter crossing the real
number line as shown in the table bellow:
{df}""")

# ##########################################
# #b)1.
# ##########################################
#
def rayleight_qt(A,x):
    x_T_A = np.dot(A,x)
    x_T_x = np.dot(np.transpose(x),x)
    return np.dot(np.transpose(x),x_T_A)/x_T_x

# ##########################################
# #b)2.
# ##########################################
def power_iteration(A,x0):
    #initate counting
    count = 0
    #compate initial vector with power iteration
    comparison = x0 == (np.dot(A,x0)/np.linalg.norm((np.dot(A,x0))))
    #preform max 1000 iterations
    while comparison.all() == False:
        x0 = (np.dot(A,x0)/np.linalg.norm((np.dot(A,x0))))
        comparison = np.absolute(x0).round(4) == np.absolute((np.dot(A,x0)/np.linalg.norm((np.dot(A,x0))))).round(4)
        count += 1
        if count == 1000:
            break
    return x0, count

##########################################
#b)3.
##########################################

def rayleight_residual(A,e_vec,e_val):
    residual = (e_val*e_vec) - np.dot(A,e_vec)
    residual_norm = np.linalg.norm(residual)
    return residual_norm

#get eigen vector and number of iterations
max_eigen_vector, iter = power_iteration(Kmat, x0)
#get the eigen value using the previously obtained eigen vector
max_eigen_value =  rayleight_qt(Kmat,max_eigen_vector)
# get te rayleight residual
residual = rayleight_residual(Kmat,max_eigen_vector,max_eigen_value)

#compare eigen value
eigen_values_np = np.linalg.eig(Kmat)[0]

print(f"""The eigen value found is:
{max_eigen_value}
with a Rayleigh residual:
{residual}
and it took {iter} iterations""")

print(f"""The max eigenvalue using numpy library is
{max(eigen_values_np)}""")

##########################################
#b)4.
##########################################

print(f"""The largest eigen value is :
{max_eigen_value}""")
show_waves(max_eigen_vector,'Max_eigen_vector_waves',basis_set)
show_nodes(max_eigen_vector,'Max_eigen_vector_nodes',basis_set)

##########################################
#c)1.
##########################################

def rayleigh_iterate(A, x0, eigen_guess):
    n = len(A)
    # get Identity in right form
    I = np.identity(n)
    # initial condition

    mat = (A - eigen_guess * I)
    # solve least squares
    v = least_squares(mat, x0)
    v = v / np.linalg.norm(v)


    # get the eigenvalue from rayleight_qt
    eig = rayleight_qt(A, v)

    count = 0
    comparison = np.abs(x0).round(11) == np.abs(v).round(11)
    while comparison.all() == False:
        x0 = v
        mat = (A - eig * I)
        v = least_squares(mat, x0)
        v = v / np.linalg.norm(v)
        comparison = np.abs(x0).round(11) == np.abs(v).round(11)
        # v = v/max(v)
        eig = rayleight_qt(A, v)
        count += 1
        if comparison.all() == True:
            break

    return v, eig, count

#########################################
#c)1.
##########################################

#get a list of 15 differet eigen_value guesses ussing the centers of gershgorin

def shift_eigen_val(A,x0,guesses):

    guesses.sort()
    eigen_vect_matrix = np.zeros(A.shape)
    eigen_values_shift = []
    iteration_list = []
    iterations_sum = 0
    col = 0
    for  eigen_guess in guesses:
        eigen_vector, eigen_val, count= rayleigh_iterate(A,x0,eigen_guess)
        iterations_sum += count
        eigen_values_shift.append(eigen_val)
        eigen_vect_matrix[:, col] = eigen_vector
        iteration_list.append(count)
        col += 1


    return eigen_values_shift,eigen_vect_matrix, iteration_list, iterations_sum




# # ##########################################
# # #c)2.
# # ##########################################

eigen_values, eigen_vect_matrix, iteration_ind, iterations_tot = shift_eigen_val(Kmat,x0,centers)


#obtain the rayleigh residuals
def multiple_residuals(A,eigen_vectors,eigen_values):
    residuals_list = []
    for col, eigen_value in enumerate(eigen_values):
        if eigen_value == 0:
            residual = 0
            residuals_list.append(residual)
        else:
            residual = rayleight_residual(A,eigen_vectors[:,col],eigen_value)
            residuals_list.append(residual)
    return residuals_list

#preparing reults to remve duplicates and find the index were duplicates ocurr
eigen_values_round = np.array(eigen_values)
eigen_values_round = np.around(eigen_values_round, 10)
residuals = multiple_residuals(Kmat,eigen_vect_matrix,eigen_values)
df = pd.DataFrame({'Eigen_values':eigen_values_round,'Residuals':residuals,'Iterations':iteration_ind})

#remove duplicates from df
df.drop_duplicates(subset ="Eigen_values",
                     keep = 'first', inplace = True)

#find duplicate eigen values and indexes
from collections import defaultdict

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items()
                            if len(locs)>1)

duplicate_eigen_values = [dup for dup in sorted(list_duplicates(eigen_values_round))]
#remove duplicates
unique_eigen_val = np.unique(eigen_values_round)


print(f"""Using the duplicates index is possible to locate where are the missing eigen values
{duplicate_eigen_values} 
unfortunaly the duplicates have consecutive indices making finding the range of the missing eigen values harder""")

#uncomment line bellow to save dataframe
#df.to_csv('Rayleigh_qt.csv',index=False)
print(f"""Using the Rayleigh iteration function we obtained the following
eigenvalues and the iterations taken for each eigenvalue in the table bellow:
{df}""")
#print(eigen_values_np)
print(f"""In {iterations_tot} iterations , it was possible to obtain
{len(unique_eigen_val)} unique eigenvalues
using using the centers of gershgorin,
As seen in the table above and the centers tables is possible to see that some
of the centers are inside other rings therefore using the centers only we obtained repeated eigen values""")



##########################################
#d)2.
##########################################


#obtaining the remaing eigen values the Transform matrxi is created
#get missing eigen values
miss_e_vec1, miss_e_val1, count = rayleigh_iterate(Kmat, x0, 3.27790711e+04)
miss_e_vec2, miss_e_val2, count = rayleigh_iterate(Kmat, x0, 5.04300277e+04)
unique_eigen_val = np.append(unique_eigen_val,3.27790711e+04)
unique_eigen_val = np.append(unique_eigen_val,5.04300277e+04)
unique_eigen_val = np.append(unique_eigen_val,max_eigen_value)
#sort eigen values from smallest to largest
unique_eigen_val.sort()
lambdas = unique_eigen_val
#breate new zero matrix
transform_mat = np.zeros(Kmat.shape)
#populate matrx with the respective eigenvectors
for col,eigen_value in enumerate(lambdas):
    vector, value, count = rayleigh_iterate(Kmat, x0, eigen_value)
    transform_mat[:,col] = vector

show_nodes(transform_mat[:,0],'lowest_eigenvalue',basis_set)

#check for K = TDiagTinv
inv_transform_mat = np.linalg.inv(transform_mat)
diagonal = np.diag(lambdas)
t_diagonal = np.dot(transform_mat,diagonal)
K = np.dot(t_diagonal,inv_transform_mat)

#comparing the two matrices
comparison = np.around(Kmat,0) == np.around(K,0)
equal_arrays = comparison.all()
print(comparison)

#checking problematic value
print(f"""The point where the Kmat matrix and K differ are {Kmat[0][3]}, {K[0][3]} respectively""")
#plot
show_all_wavefunction_nodes(transform_mat,lambdas,'All_solutions', basis_set)







