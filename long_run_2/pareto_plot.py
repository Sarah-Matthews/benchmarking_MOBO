from open_file import yld_values_array_mobo, ton_values_array_mobo, yld_values_array_bofire, ton_values_array_bofire
import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

mobo_yld = yld_values_array_mobo[:,-40:]
mobo_ton = ton_values_array_mobo[:,-40:]
bofire_yld = yld_values_array_bofire[:,-40:]
bofire_ton = ton_values_array_bofire[:,-40:]


def find_pareto_front(yield_values, ton_values):
    points = np.array(list(zip(yield_values, ton_values)))  # Combine yield and ton into a 2D array
    is_pareto = np.ones(points.shape[0], dtype=bool)  # Assume all points are Pareto
    
    # Check each point against all other points
    for i, p in enumerate(points):
        if is_pareto[i]:  # Check if the point is currently Pareto
            # A point is dominated if any other point is strictly better in both yield and ton
            is_pareto[i] = not np.any(np.all(points > p, axis=1))  # No other point dominates this one
    
    return points[is_pareto]  # Return only the non-dominated (Pareto) points


all_yields_mobo = mobo_yld.flatten() 
all_tons_mobo = mobo_ton.flatten()  
all_yields_bofire = bofire_yld.flatten() 
all_tons_bofire = bofire_ton.flatten()  



# Create an array for the iteration values (1 to 30) for each column
iterations = np.tile(np.arange(1, 41), 15)
# Create the scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(all_yields_mobo, all_tons_mobo, c=iterations, cmap='viridis', s=50)

# Add a colorbar
plt.colorbar(scatter, label='Iteration')

# Labels and title
plt.xlabel('Yield')
plt.ylabel('Turnover number')
plt.title('Yield vs TON for BayBE MOBO optimisation')

# Show the plot
plt.show()


# Create an array for the iteration values (1 to 30) for each column
iterations = np.tile(np.arange(1, 41), 15)
# Create the scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(all_yields_bofire, all_tons_bofire, c=iterations, cmap='viridis', s=50)

# Add a colorbar
plt.colorbar(scatter, label='Iteration')

# Labels and title
plt.xlabel('Yield')
plt.ylabel('Turnover number')
plt.title('Yield vs TON for BoFire optimisation')

# Show the plot
plt.show()




# Combine into a 2D array of points (YLD, TON)
all_points_mobo = np.column_stack((all_yields_mobo, all_tons_mobo))
all_points_bofire = np.column_stack((all_yields_bofire, all_tons_bofire))
'''

# Revised Pareto front function
def pareto_front(points):
    is_pareto = np.ones(points.shape[0], dtype=bool)  # Start assuming all points are on the Pareto front

    for i, point in enumerate(points):
        if is_pareto[i]:  # Only check points that are currently considered Pareto
            # Mark dominated points
            is_pareto[is_pareto] = ~(
                (points[is_pareto][:, 0] >= point[0]) &  # YLD: other point >= current point
                (points[is_pareto][:, 1] >= point[1]) &  # TON: other point >= current point
                ((points[is_pareto][:, 0] > point[0]) |  # At least one strict dominance
                 (points[is_pareto][:, 1] > point[1]))
            )

    return points[is_pareto]

# Find the Pareto front
pareto_points = pareto_front(all_points)

# Sort the Pareto front by YLD for plotting purposes
pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

# Plot all points and the Pareto front
plt.scatter(all_yields_mobo, all_tons_mobo, alpha=0.5, label="All Points")
plt.plot(pareto_points[:, 0], pareto_points[:, 1], color="r", label="Pareto Front", linewidth=2)
plt.xlabel("YLD")
plt.ylabel("TON")
plt.legend()
plt.title("Pareto Front Across All Repeats")
plt.show()

'''

def pareto_front(points):
    # Sort points by the first objective in ascending order
    sorted_points = points[points[:, 0].argsort()]
    pareto = []

    for point in sorted_points:
        # Check if the current point is dominated by any point in the current Pareto front
        dominated = False
        for pareto_point in pareto:
            if (pareto_point[0] <= point[0] and pareto_point[1] <= point[1]) and (pareto_point[0] < point[0] or pareto_point[1] < point[1]):
                dominated = True
                break
        
        if not dominated:
            pareto.append(point)

    return np.array(pareto)



    

pareto_points_mobo = pareto_front(all_points_mobo)


pareto_points_mobo = pareto_points_mobo[np.argsort(pareto_points_mobo[:, 0])]


plt.scatter(all_yields_mobo, all_tons_mobo, alpha=0.5, label="All Points")
plt.plot(pareto_points_mobo[:, 0], pareto_points_mobo[:, 1], color="r", label="Pareto Front", linewidth=2)
plt.xlabel("YLD")
plt.ylabel("TON")
plt.legend()
plt.title("Pareto Front Across All Repeats")
plt.show()





pareto_points_bofire = pareto_front(all_points_bofire)
pareto_points_bofire = pareto_points_bofire[np.argsort(pareto_points_bofire[:, 0])]


plt.scatter(all_yields_bofire, all_tons_bofire, alpha=0.5, label="All Points")
plt.plot(pareto_points_bofire[:, 0], pareto_points_bofire[:, 1], color="r", label="Pareto Front", linewidth=2)
plt.xlabel("YLD")
plt.ylabel("TON")
plt.legend()
plt.title("Pareto Front Across All Repeats")
plt.show()

