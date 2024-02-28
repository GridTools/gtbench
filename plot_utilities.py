import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import math

def parse_input_file(file_path):
    # Initialize dictionaries to store the parsed data
    domain_data = {
        'Domain size': [],
        'Domain X size': [],
        'Domain Y size': [],
        'Domain Z size': [],
        'Median runtime': [],
        'Median Time 95% Confidence Lower': [],
        'Median Time 95% Confidence Upper': [],
        'Columns per Second': [],
        'Columns per Second 95% Confidence Lower': [],
        'Columns per Second 95% Confidence Upper': [],
        'Elements per Second': [],
        'Elements per Second 95% Confidence Lower': [],
        'Elements per Second 95% Confidence Upper': []
    }

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Regular expressions to extract relevant information
    domain_size_pattern = re.compile(r'Domain size:\s+(\d+)x(\d+)x(\d+)')
    median_time_pattern = re.compile(r'Median time:\s+(\d+(?:\.\d+)?)s \(95% confidence: (\d+(?:\.\d+)?)s - (\d+(?:\.\d+)?)s\)')
    cols_per_second_pattern = re.compile(r'Columns per second:\s+(\d+(?:\.\d+)?) \(95% confidence: (\d+(?:\.\d+)?) - (\d+(?:\.\d+)?)\)')

    # Parsing the file content
    current_domain_size = None
    for line in lines:
        domain_size_match = domain_size_pattern.search(line)
        median_time_match = median_time_pattern.search(line)
        cols_per_second_match = cols_per_second_pattern.search(line)

        if domain_size_match:
            current_domain_x_size = int(domain_size_match.group(1))
            current_domain_y_size = int(domain_size_match.group(2))
            current_domain_z_size = int(domain_size_match.group(3))
            current_total_domain_size = current_domain_x_size * current_domain_y_size * current_domain_z_size
            current_domain_size = "{}x{}x{}".format(current_domain_x_size, current_domain_y_size, current_domain_z_size)
            domain_data['Domain size'].append(current_domain_size)
            domain_data['Domain X size'].append(current_domain_x_size)
            domain_data['Domain Y size'].append(current_domain_y_size)
            domain_data['Domain Z size'].append(current_domain_z_size)
        elif median_time_match and current_domain_size:
            median_time = float(median_time_match.group(1))
            conf_lower = float(median_time_match.group(2))
            conf_upper = float(median_time_match.group(3))
            domain_data['Median runtime'].append(median_time)
            domain_data['Median Time 95% Confidence Lower'].append(conf_lower)
            domain_data['Median Time 95% Confidence Upper'].append(conf_upper)
            domain_data['Elements per Second'].append(current_total_domain_size/median_time)
            domain_data['Elements per Second 95% Confidence Lower'].append(current_total_domain_size/conf_upper)
            domain_data['Elements per Second 95% Confidence Upper'].append(current_total_domain_size/conf_lower)
        elif cols_per_second_match and current_domain_size:
            cols_per_second = float(cols_per_second_match.group(1))
            conf_lower = float(cols_per_second_match.group(2))
            conf_upper = float(cols_per_second_match.group(3))
            domain_data['Columns per Second'].append(cols_per_second)
            domain_data['Columns per Second 95% Confidence Lower'].append(conf_lower)
            domain_data['Columns per Second 95% Confidence Upper'].append(conf_upper)

    return domain_data

def create_plot(data, labels, output_name, title, x_key, y_key, upper_key, lower_key, xlabel, ylabel, xscale='linear', yscale='linear'):
    min_x = 0
    max_x = 0
    # Plotting with for loop
    plt.figure(figsize=(10, 6))
    for idx, (cpu, cpu_data) in enumerate(data.items()):
        for idx2, (key, value) in enumerate(cpu_data.items()):
            df = pd.DataFrame(value)
            min_x = min(df[x_key])
            max_x = max(df[x_key])
            sns.lineplot(data=df, x=x_key, y=y_key, marker='o', label=labels[idx] + ' ' + key)
            plt.errorbar(df[x_key], df[y_key], yerr=[df[y_key] - df[lower_key], df[upper_key] - df[y_key]], fmt='o', capsize=5)

    # Setting labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Setting the scale to linear for the y-axis
    plt.yscale(yscale)

    # Displaying the plot
    plt.xscale(xscale)

    # plt.xticks([16, 32, 64, 128, 256, 512, 1024], rotation=0, fontsize=7)
    # Dynamically determining the range of xticks
    min_pow = math.floor(math.log2(min_x))
    max_pow = math.ceil(math.log2(max_x))
    xticks = [2 ** i for i in range(min_pow, max_pow + 1)]
    plt.xticks(xticks)

    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(output_name, dpi=300)

