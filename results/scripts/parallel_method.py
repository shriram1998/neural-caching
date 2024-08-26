import pandas as pd
import plotly.graph_objects as go

# Define the data
data = {
    'Training method': ['Complete retraining', 'Incremental', 'EWC', 'Replay (100%)', 'Replay (50%)'],
    'Online Accuracy': [0.747, 0.739, 0.741, 0.743, 0.746],
    'Final Accuracy': [0.705, 0.693, 0.698, 0.700, 0.704],
    'Training Time (s)': [2055, 797, 852, 1803, 1416],
    'FLOPS (E+15)': [5.62, 2.12, 2.17, 4.44, 3.81]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Normalize the data for better visualization
columns_to_normalize = ['Online Accuracy', 'Final Accuracy', 'Training Time (s)', 'FLOPS (E+15)']
df_normalized = df.copy()
for column in columns_to_normalize:
    df_normalized[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

# Create the parallel coordinates plot
fig = go.Figure()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Add a trace for each training method
for i, method in enumerate(df_normalized['Training method']):
    fig.add_trace(
        go.Parcoords(
            line=dict(color=colors[i]),
            dimensions=[
                dict(range=[df_normalized['Online Accuracy'].min(), df_normalized['Online Accuracy'].max()],
                     label='Online Accuracy', values=[df_normalized['Online Accuracy'][i]]),
                dict(range=[df_normalized['Final Accuracy'].min(), df_normalized['Final Accuracy'].max()],
                     label='Final Accuracy', values=[df_normalized['Final Accuracy'][i]]),
                dict(range=[df_normalized['Training Time (s)'].min(), df_normalized['Training Time (s)'].max()],
                     label='Training Time (s)', values=[df_normalized['Training Time (s)'][i]]),
                dict(range=[df_normalized['FLOPS (E+15)'].min(), df_normalized['FLOPS (E+15)'].max()],
                     label='FLOPS (E+15)', values=[df_normalized['FLOPS (E+15)'][i]])
            ],
            name=method,
            legendrank=i,
        )
    )

# Update the layout to remove the gradient sidebar and add a legend
fig.update_layout(
    title='Parallel Coordinates Plot of Training Methods',
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(title="Training Methods", orientation="h", x=0.5, xanchor="center", y=1.1, yanchor="top"),
)

# Show the plot
fig.show()

# Save the plot as an HTML file
# fig.write_html("parallel_plot.html")
