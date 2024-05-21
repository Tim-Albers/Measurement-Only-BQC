import pandas as pd
import matplotlib.pyplot as plt


def load_plot_and_save_fig(data, x, y, yerr, title, xlabel, ylabel, save_path):
    # Extract data
    x_data = data[x]
    y_data = data[y]
    yerr_data = data[yerr]

    # Create the error bar plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(x_data, y_data, yerr=yerr_data, fmt='o', capsize=3)
    plt.axhline(y=0.7, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(save_path)

# Load the data
data = pd.read_csv("/home/tim/Desktop/Desktop/DELFT/BEP/DATA/length_successrate_error.csv")

# Plot the data and save the figure
load_plot_and_save_fig(
    data,
    'length',
    'successrate',
    'error',
    '',
    'Length (km)',
    'Success rate',
    '/home/tim/Desktop/Desktop/DELFT/BEP/DATA/success_rate_plot.png'  # Ensure you include the filename in the path
)
