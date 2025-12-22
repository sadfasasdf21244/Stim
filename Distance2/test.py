#%%    
import itertools
bases = ['X', 'Y', 'Z']
    
# 3^4 = 81가지 기저 조합 생성
basis_combinations = list(itertools.product(bases, repeat=4))
# %%
print(basis_combinations)
# %%
