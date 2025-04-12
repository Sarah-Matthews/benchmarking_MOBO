import pymoo
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.util.misc import stack
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import gaussian_kde
from mobo_plots import all_yields_mobo, all_tons_mobo, all_yields_bofire, all_tons_bofire, mobo_yld, mobo_ton, bofire_ton, bofire_yld
from times_analysis_corr import average_time_bofire, average_time_mobo, std_bofire, std_mobo


'''
This file is responsible for hypervolume calculations and plotting pareto fronts
'''


all_points_mobo = np.column_stack((all_yields_mobo, all_tons_mobo))
all_points_bofire = np.column_stack((all_yields_bofire, all_tons_bofire))


nds_mobo = NonDominatedSorting().do(-all_points_mobo, only_non_dominated_front=True)
pareto_front_mobo = all_points_mobo[nds_mobo]
pareto_front_mobo = pareto_front_mobo[np.argsort(pareto_front_mobo[:, 0])]

nds_bofire = NonDominatedSorting().do(-all_points_bofire, only_non_dominated_front=True)
pareto_front_bofire = all_points_bofire[nds_bofire]
pareto_front_bofire = pareto_front_bofire[np.argsort(pareto_front_bofire[:, 0])]



iterations = np.tile(np.arange(1, 36), 15)


plt.figure(figsize=(8, 6))
scatter_mobo = plt.scatter(all_points_mobo[:, 0], all_points_mobo[:, 1], c=iterations, cmap='viridis', s=50, label="All Solutions")


#plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', edgecolor='k', s=100, label="Pareto Front")
plt.plot(pareto_front_mobo[:, 0], pareto_front_mobo[:, 1], color='red', linewidth=2, label="Pareto Front")
#plt.scatter(pareto_front_mobo[:, 0], pareto_front_mobo[:, 1], color='red', edgecolor='k', s=30, )


plt.colorbar(scatter_mobo, label='Iteration')


plt.xlabel('Yield (%)', fontsize = 13)
plt.ylabel('Turnover Number', fontsize = 13)
#plt.title('Yield vs TON for BayBE MOBO Optimisation')
plt.legend()
plt.show()



plt.figure(figsize=(8, 6))
scatter_bofire = plt.scatter(all_points_bofire[:, 0], all_points_bofire[:, 1], c=iterations, cmap='viridis', s=50, label="All Solutions")


#plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', edgecolor='k', s=100, label="Pareto Front")
plt.plot(pareto_front_bofire[:, 0], pareto_front_bofire[:, 1], color='red', linewidth=2, label="Pareto Front")
#plt.scatter(pareto_front_bofire[:, 0], pareto_front_bofire[:, 1], color='red', edgecolor='k', s=30, marker='x')


plt.colorbar(scatter_bofire, label='Iteration')


plt.xlabel('Yield (%)', fontsize = 13)
plt.ylabel('Turnover Number', fontsize = 13)
#plt.title('Yield vs TON for BOFire MOBO Optimisation')
plt.legend()
plt.show()


from pymoo.indicators.hv import Hypervolume
import numpy as np
import matplotlib.pyplot as plt




'''
MOBO hypervolume calcs
'''







approx_ideal_mobo = pareto_front_mobo.min(axis=0)  # max values for each objective
approx_nadir_mobo = pareto_front_mobo.max(axis=0)  # min values for each objective


ref_point_mobo = np.array([approx_nadir_mobo[0] - 0.1, approx_nadir_mobo[1] - 0.1])

metric_mobo = Hypervolume(ref_point=ref_point_mobo,
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal_mobo,
                     nadir=approx_nadir_mobo)


#calculating hypervolume for each iteration, averaged over 15 repeats

hv_per_iteration_mobo = []
hv_std_per_iteration_mobo = []

for i in range(35):  
    hv_for_repeats_mobo = []  
    
    for j in range(15): 
        
        cumulative_points_mobo = np.column_stack((mobo_yld[j, :i+1], mobo_ton[j, :i+1]))
        
        
        hv_for_repeats_mobo.append(metric_mobo.do(cumulative_points_mobo))
    
    
    hv_per_iteration_mobo.append(np.mean(hv_for_repeats_mobo))

    
    hv_std_per_iteration_mobo.append(np.std(hv_for_repeats_mobo))


iterations = np.arange(1, 36)
hv_per_iteration_mobo = np.array(hv_per_iteration_mobo)
hv_std_per_iteration_mobo = np.array(hv_std_per_iteration_mobo)

'''
plt.figure(figsize=(8, 6))
scatter_mobo = plt.scatter(all_points_mobo[:, 0], all_points_mobo[:, 1], c=iterations, cmap='viridis', s=50, label="All Solutions")


#plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', edgecolor='k', s=100, label="Pareto Front")
plt.plot(pareto_front_mobo[:, 0], pareto_front_mobo[:, 1], color='red', linewidth=2, label="Pareto Front")
#plt.scatter(pareto_front_mobo[:, 0], pareto_front_mobo[:, 1], color='red', edgecolor='k', s=30, )

# Plot ideal, nadir, and reference points
plt.scatter(approx_ideal_mobo[0], approx_ideal_mobo[1], color='blue', marker='*', s=150, label="Ideal Point")  
plt.scatter(approx_nadir_mobo[0], approx_nadir_mobo[1], color='black', marker='X', s=150, label="Nadir Point")
plt.scatter(ref_point_mobo[0], ref_point_mobo[1], color='purple', marker='D', s=150, label="Reference Point")

plt.colorbar(scatter_mobo, label='Iteration')

plt.title('NADIR')
plt.xlabel('Yield (%)', fontsize = 13)
plt.ylabel('Turnover Number', fontsize = 13)
#plt.title('Yield vs TON for BayBE MOBO Optimisation')
plt.legend()
plt.show()
'''

#plotting the average hypervolume with standard deviation as shaded area
plt.figure(figsize=(8, 6))
plt.plot(iterations, hv_per_iteration_mobo, color='#ff7f0e', lw=1.5,) #label="Average Hypervolume")
plt.fill_between(
    iterations,
    hv_per_iteration_mobo - hv_std_per_iteration_mobo,
    hv_per_iteration_mobo + hv_std_per_iteration_mobo,
    color='#ff7f0e',
    alpha=0.2,
    #label="±1 standard deviation"
)
plt.scatter(iterations, hv_per_iteration_mobo, facecolor="none", edgecolor='#ff7f0e', marker="p")
#plt.title("Hypervolume Convergence with Standard Deviation")
plt.minorticks_on()
plt.grid(True)
plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   # Lighter gridlines for y-axis
plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.xlabel("Iterations", fontsize = 13)
plt.ylabel("Hypervolume", fontsize = 13)
#plt.legend()
plt.show()

terminal_hypervolume_mobo = hv_per_iteration_mobo[-1]
terminal_hypervolume_std_mobo = hv_std_per_iteration_mobo[-1]

print(f"Terminal Hypervolume BayBE: {terminal_hypervolume_mobo}")
print(f"Terminal Hypervolume Standard Deviation BayBE: {terminal_hypervolume_std_mobo}")




'''
BOFire hypervolume calcs
'''



approx_ideal_bofire = pareto_front_bofire.min(axis=0)  
approx_nadir_bofire = pareto_front_bofire.max(axis=0)  


ref_point_bofire = np.array([approx_nadir_bofire[0] + 0.1, approx_nadir_bofire[1] + 0.1])


metric_bofire = Hypervolume(ref_point=ref_point_bofire,
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal_bofire,
                     nadir=approx_nadir_bofire)


#calculating hypervolume for each iteration, averaged over 15 repeats

hv_per_iteration_bofire = []
hv_std_per_iteration_bofire = []

for i in range(35):  
    hv_for_repeats_bofire = []  
    
    for j in range(15): 
        
        cumulative_points_bofire = np.column_stack((bofire_yld[j, :i+1], bofire_ton[j, :i+1]))
        
        
        hv_for_repeats_bofire.append(metric_bofire.do(cumulative_points_bofire))
    
    
    hv_per_iteration_bofire.append(np.mean(hv_for_repeats_bofire))

    
    hv_std_per_iteration_bofire.append(np.std(hv_for_repeats_bofire))


iterations = np.arange(1, 36)
hv_per_iteration_bofire = np.array(hv_per_iteration_bofire)
hv_std_per_iteration_bofire = np.array(hv_std_per_iteration_bofire)

#plotting the average hypervolume with standard deviation as shaded area
plt.figure(figsize=(8, 6))
plt.plot(iterations, hv_per_iteration_bofire, color='#2ca02c', lw=1.5,) #label="Average Hypervolume")
plt.fill_between(
    iterations,
    hv_per_iteration_bofire - hv_std_per_iteration_bofire,
    hv_per_iteration_bofire + hv_std_per_iteration_bofire,
    color='#2ca02c',
    alpha=0.2,
    #label="±1 standard deviation"
)
plt.scatter(iterations, hv_per_iteration_bofire, facecolor="none", edgecolor='#2ca02c', marker="p")
#plt.title("Hypervolume Convergence with Standard Deviation")
plt.minorticks_on()
plt.grid(True)
plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   # Lighter gridlines for y-axis
plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.xlabel("Iterations", fontsize = 13)
plt.ylabel("Hypervolume", fontsize = 13)
#plt.legend()
plt.show()

terminal_hypervolume_bofire = hv_per_iteration_bofire[-1]
terminal_hypervolume_std_bofire = hv_std_per_iteration_bofire[-1]

print(f"Terminal Hypervolume BOFire: {terminal_hypervolume_bofire}")
print(f"Terminal Hypervolume Standard Deviation BOFire: {terminal_hypervolume_std_bofire}")



'''

combined hypervolume plot

'''


#plotting the average hypervolume with standard deviation as shaded area
plt.figure(figsize=(8, 6))


plt.plot(iterations, hv_per_iteration_bofire, color='#2ca02c', lw=1.5, label="BoFire")
plt.fill_between(
    iterations,
    hv_per_iteration_bofire - hv_std_per_iteration_bofire,
    hv_per_iteration_bofire + hv_std_per_iteration_bofire,
    color='#2ca02c',
    alpha=0.2,
    #label="±1 standard deviation"
)

plt.plot(iterations, hv_per_iteration_mobo, color='#ff7f0e', lw=1.5, label="BayBE")
plt.fill_between(
    iterations,
    hv_per_iteration_mobo - hv_std_per_iteration_mobo,
    hv_per_iteration_mobo + hv_std_per_iteration_mobo,
    color='#ff7f0e',
    alpha=0.2,
    #label="±1 standard deviation"
)
plt.scatter(iterations, hv_per_iteration_mobo, facecolor="none", edgecolor='#ff7f0e', marker="p")

plt.scatter(iterations, hv_per_iteration_bofire, facecolor="none", edgecolor='#2ca02c', marker="p")
#plt.title("Hypervolume Convergence with Standard Deviation")
plt.minorticks_on()
plt.grid(True)
plt.grid(axis='y', which='major', linestyle=':', alpha=0.3)   # Lighter gridlines for y-axis
plt.grid(axis='y', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.grid(axis='x', which='minor', linestyle=':', alpha=0.1, color='grey')
plt.xlabel("Iterations", fontsize = 13)
plt.ylabel("Hypervolume", fontsize = 13)
plt.legend(loc="lower right")
plt.show()


'''
bar chart of terminal hypervolume vs iteration time

'''


models = ['BayBE', 'BoFire']


hypervolume_means = [terminal_hypervolume_mobo, terminal_hypervolume_bofire]
hypervolume_stds = [terminal_hypervolume_std_mobo, terminal_hypervolume_std_bofire]


time_means = [average_time_mobo, average_time_bofire]
time_stds = [std_mobo, std_bofire]


x = np.arange(len(models)) 
bar_width = 0.4


fig, ax1 = plt.subplots(figsize=(10, 6))


bars_hypervolume = ax1.bar(
    x - bar_width / 2, hypervolume_means, bar_width,
    yerr=hypervolume_stds, capsize=5, label='Terminal Hypervolume', color='blue'
)
ax1.set_ylabel('Terminal Hypervolume', color='blue', fontsize = 13)
ax1.tick_params(axis='y', labelcolor='blue')


ax2 = ax1.twinx()
bars_time = ax2.bar(
    x + bar_width / 2, time_means, bar_width,
    yerr=time_stds, capsize=5, label='Average Time per Iteration', color='#ff7f0e'
)
ax2.set_ylabel('Average Time per Iteration (s)', color='#ff7f0e', fontsize = 14)
ax2.tick_params(axis='y', labelcolor='#ff7f0e')


ax1.set_xlabel('Models', fontsize = 14)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
plt.title('Terminal Hypervolume and Average Time per Iteration')


fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
#plt.legend(loc="lower right")
# Show the plot
plt.tight_layout()
plt.show() 

models = ['BayBE', 'BoFire']
hypervolume_means = [terminal_hypervolume_mobo, terminal_hypervolume_bofire]
hypervolume_stds = [terminal_hypervolume_std_mobo, terminal_hypervolume_std_bofire]
time_means = [average_time_mobo, average_time_bofire]
time_stds = [std_mobo, std_bofire]


x = np.arange(len(models))  
bar_width = 0.3

fig, ax1 = plt.subplots(figsize=(10, 6))


bars_hypervolume = ax1.bar(
    x - bar_width / 2, hypervolume_means, bar_width,
    yerr=hypervolume_stds, capsize=5, label='Terminal Hypervolume', color='#0c4a6e'
)
ax1.set_ylabel('Terminal Hypervolume', color='black', fontsize=14)
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(0, 8000)
ax1.set_yticks(np.arange(0, 8001, 2000)) 


ax2 = ax1.twinx()
bars_time = ax2.bar(
    x + bar_width / 2, time_means, bar_width,
    yerr=time_stds, capsize=5, label='Average Time per Iteration', color='#9e2a2f'
)
ax2.set_ylabel('Average Time per Iteration (s)', color='black', fontsize=14)  # Change to black
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_yscale('log')


ax1.set_xlabel('Models', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
#plt.title('Terminal Hypervolume and Average Time per Iteration')


#fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
handles, labels = ax1.get_legend_handles_labels()  # Get handles from ax1
handles2, labels2 = ax2.get_legend_handles_labels()  # Get handles from ax2


handles.extend(handles2)
labels.extend(labels2)


plt.legend(handles, labels, loc="upper left")
#plt.legend(loc="upper left")
# Show the plot
plt.tight_layout()
plt.show()


#distribution metrics:

print('pareto_front_baybe', pareto_front_mobo)
print('pareto_front_baybe no. points', len(pareto_front_mobo))
print('pareto_front_bofire', pareto_front_bofire)
print('pareto_front_bofire no. points', len(pareto_front_bofire))


kde = gaussian_kde(pareto_front_mobo.T)
densities = kde(pareto_front_mobo.T)

print(densities)


def spacing_metric(pareto_points):
    
    pareto_points = np.array(sorted(pareto_points, key=lambda x: x[0]))
    
    
    distances = np.linalg.norm(np.diff(pareto_points, axis=0), axis=1)
    
    
    d_mean = np.mean(distances)
    
    
    S = np.sqrt(np.sum((distances - d_mean) ** 2) / (len(distances) - 1))
    
    return S

baybe_spacing_metric = spacing_metric(pareto_front_mobo)
print('baybe_spacing_metric', baybe_spacing_metric)
bofire_spacing_metric = spacing_metric(pareto_front_bofire)
print('bofire_spacing_metric', bofire_spacing_metric)