#%%
from surface2_functions import state_prep_test, Figure3, plot_figure5
#%%
state_prep_test(p_1q=0.005,   
                p_2q=0.02,    
                p_meas=0.03,
                p_1q_z=0.0     
                )

#%%
Figure3(shots = 10000,
        p_1q=0.005,   
        p_2q=0.02, 
        p_meas=0.03, 
        p_1q_z=0.0   
        )

#%%
plot_figure5(
    p_1q=0.005, 
    p_2q=0.02, 
    p_meas=0.03, 
    p_1q_z=0, 
    shots=1000000, 
    rounds=10
)