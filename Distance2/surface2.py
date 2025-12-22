#%%
from surface2_functions import *
from Assignment_matrix import *
from Density_matrix import *
#%%
# state_prep_test(p_1q=0.003, 
#                 p_2q=0.02, 
#                 p_meas=0.02, 
#                 shots = 10000,
#                 T2 = 50,
# )

#%%
# Figure3(
#                 p_1q=0.002, 
#                 p_2q=0.02, 
#                 p_meas=0.01, 
#                 shots=10000,
#                 T2 = 15,
#                 squence_time = 2,)

#%%
# plot_figure5(   
#                 p_1q=0.002, 
#                 p_2q=0.02, 
#                 p_meas=0.01, 
#                 shots=1000000,
#                 T2 = 15,
#                 squence_time = 0,
#                 rounds=10,)
#%%
# ass = assignment_matrix(DATA_QUBITS,
#                 p_1q = 0,
#                 p_2q = 0,
#                 p_meas = 0,
#                 shots = 100000,
#                 squence_time = 0,)
# print(ass)
# %%
state = '-'
# density matrix 확인
den_mat = density_matrix(
    target_state = state,
                p_1q=0.00, 
                p_2q=0.00, 
                p_meas=0.00, 
                shots=10000000,
                T2 = 15,
                squence_time = 0,
                )
