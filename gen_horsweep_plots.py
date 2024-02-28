from plot_utilities import create_plot, parse_input_file

def create_time_plot(data, labels, output_name):
    create_plot(data, labels, output_name, 'GTBENCH Median Runtime with 95% Confidence Intervals', 'Domain X size', 'Median runtime', 'Median Time 95% Confidence Upper', 'Median Time 95% Confidence Lower', 'Domain Size (NxNx64)', 'Median Runtime')

def create_columns_plot(data, labels, output_name):
    create_plot(data, labels, output_name, 'GTBENCH Columns per Second with 95% Confidence Intervals', 'Domain X size', 'Columns per Second', 'Columns per Second 95% Confidence Upper', 'Columns per Second 95% Confidence Lower', 'Domain Size (NxNx64)', 'Columns per Second')

def create_elements_plot(data, labels, output_name):
    create_plot(data, labels, output_name, 'GTBENCH Elements per Second with 95% Confidence Intervals', 'Domain X size', 'Elements per Second', 'Elements per Second 95% Confidence Upper', 'Elements per Second 95% Confidence Lower', 'Domain Size (NxNx64)', 'Elements per Second')


data = {}

systems = ["GH200"]
input_files = ["cpu_ihorswp.out", "cpu_khorswp.out"]
labels = ["cpu_ifirst", "cpu_kfirst"]

for system in systems:
    data[system] = {}
    for file, variant in zip(input_files, labels):
        data[system][variant] = parse_input_file(file)
        print(data[system][variant])

create_time_plot(data, systems, "santis_time_horswp.png")
create_columns_plot(data, systems, "santis_columns_horswp.png")
create_elements_plot(data, systems, "santis_elements_horswp.png")
