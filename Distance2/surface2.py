#%%
from surface2_functions import *
from Assignment_matrix import *
from Density_matrix import *

save_directory = "Distance2/Figures/251224"
#%%
# state_prep_test(p_1q=0.005, 
#                 p_2q=0.03, 
#                 p_meas=0.01, 
#                 shots = 10000,
#                 T2 = 15,
#                 sequence_time = 2,
# )

#%%
# Figure3(
#                 p_1q=0.005, 
#                 p_2q=0.03, 
#                 p_meas=0.01, 
#                 shots = 10000,
#                 T2 = 15,
#                 sequence_time = 2,
#                 with_plot = True,
#                 save_directory = save_directory)


#%%
plot_figure5(   
                p_1q=0.005, 
                p_2q=0.03, 
                p_meas=0.01, 
                shots=1000000,
                rounds=10,
                with_plot = True,
                save_directory = save_directory,
                T2 = 15,
                sequence_time = 2,
)
#%%
# ass = assignment_matrix(DATA_QUBITS,
#                 p_1q = 0.005,
#                 p_2q = 0.03,
#                 p_meas = 0.01,
#                 shots = 100000,
#                 T2 = 15,
#                 sequence_time = 2,)
# plot_assignment_matrix(ass, title="4-Qubit Assignment Matrix", show_values=True)
#%%
# state_rep  = '-'
# # density matrix 확인
# # for state_rep in ['0','1','+','-']:
#     den_mat = density_matrix(
#     target_state = 'state_rep',
#                 p_1q=0.005, 
#                 p_2q=0.03, 
#                 p_meas=0.01, 
#                 shots=10000,
#                 T2 = 15,
#                 sequence_time = 2,
#                 with_plot = True,
#                 save_directory = save_directory
#                 )

#%% Circuit 시각화, 저장
# builder = state_prep('0',
#     p_1q=0.00, 
#     p_2q=0.0, 
#     p_meas=0.0, 
#     T2 = 15,
#     sequence_time = 0,)
# # builder.measure_ancilla()
# circuit = builder.get_circuit()
# visualize_circuit_ticks(circuit, save_directory)
#%% qubit들 위치 시각화
# plot_qubit_layout()