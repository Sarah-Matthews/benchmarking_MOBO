from open_file_corr import yld_values_array_mobo, ton_values_array_mobo, yld_values_array_bofire, ton_values_array_bofire
import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


'''
This file is responsible for creating multi-objective plots
'''


mobo_yld = yld_values_array_mobo[:,-35:]
mobo_ton = ton_values_array_mobo[:,-35:]
bofire_yld = yld_values_array_bofire[:,-35:]
bofire_ton = ton_values_array_bofire[:,-35:]

points = np.array(list(zip(mobo_yld, mobo_ton)))



all_yields_mobo = mobo_yld.flatten() 
all_tons_mobo = mobo_ton.flatten()  
all_yields_bofire = bofire_yld.flatten() 
all_tons_bofire = bofire_ton.flatten()  



#array for the iteration values (1 to 30) for each column
iterations = np.tile(np.arange(1, 36), 15)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(all_yields_mobo, all_tons_mobo, c=iterations, cmap='viridis', s=50)


plt.colorbar(scatter, label='Iteration')


plt.xlabel('Yield (%)', fontsize = 13)
plt.ylabel('Turnover number', fontsize = 13)
#plt.title('Yield vs TON for BayBE MOBO optimisation')


plt.show()


#array for the iteration values (1 to 30) for each column
iterations = np.tile(np.arange(1, 36), 15)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(all_yields_bofire, all_tons_bofire, c=iterations, cmap='viridis', s=50)


plt.colorbar(scatter, label='Iteration')


plt.xlabel('Yield (%)', fontsize = 13)
plt.ylabel('Turnover number', fontsize = 13)
#plt.title('Yield vs TON for BoFire optimisation')


plt.show()




all_points_mobo = np.column_stack((all_yields_mobo, all_tons_mobo))
all_points_bofire = np.column_stack((all_yields_bofire, all_tons_bofire))
