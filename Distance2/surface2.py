#%%
from surface2_functions import *
from Assignment_matrix import *
from Density_matrix import *

save_directory = "Distance2/Figures/251223"
#%%
# state_prep_test(p_1q=0.005, 
#                 p_2q=0.02, 
#                 p_meas=0.01, 
#                 shots = 10000,
#                 T2 = 15,
#                 squence_time = 2,
# )

#%%
# Figure3(
#                 p_1q=0.005, 
#                 p_2q=0.02, 
#                 p_meas=0.01, 
#                 shots=10000,
#                 T2 = 15,
#                 squence_time = 2,                
#                  with_plot = True,
                # save_directory = save_directory)


#%%
plot_figure5(   
                p_1q=0.005, 
                p_2q=0.02, 
                p_meas=0.01, 
                shots=10000,
                rounds=10,
                with_plot = True,
                save_directory = save_directory,
                T2 = 15,
                squence_time = 2,
)
#%%
# ass = assignment_matrix(DATA_QUBITS,
#                 p_1q = 0,
#                 p_2q = 0,
#                 p_meas = 0,
#                 shots = 100000,
#                 squence_time = 0,)
# print(ass)
# %%
# state = '-'
# # density matrix 확인
# for state_rep in ['0','1','+','-']:
#     den_mat = density_matrix(
#     target_state = state_rep,
#                 p_1q=0.005, 
#                 p_2q=0.02, 
#                 p_meas=0.01, 
#                 shots=10000,
#                 T2 = 15,
#                 squence_time = 2,
#                 with_plot = True,
#                 save_directory = save_directory
#                 )