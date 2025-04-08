import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = [
    {
        "parameters": "1.5B",
        "finetuned": "No",
        "performance": 0.24347826086956523,
        "performance_std": 0.02836109930007507,
    },
    {
        "parameters": "1.5B",
        "finetuned": "Yes",
        "performance": 0.28695652173913044,
        "performance_std": 0.029891541673635464,
    },
    {
        "parameters": "7B",
        "finetuned": "No",
        "performance": 0.30434782608695654,
        "performance_std": 0.030406290061389892,
    },
    {
        "parameters": "7B",
        "finetuned": "Yes",
        "performance": 0.3347826086956522,
        "performance_std": 0.031184982222272187,
    },
]

dataframe = pd.DataFrame(data)

dataframe

sns.barplot(
    data=dataframe,
    x="parameters",
    y="performance",
    ci=None,
    yerr=dataframe["performance_std"],
    # errorbar=("sd",dataframe["performance_std"]),
    hue="finetuned",
)

# Create figure
plt.figure(figsize=(8, 5))

# Plot bars (without confidence intervals)
ax = sns.barplot(
    data=dataframe,
    x="parameters",
    y="performance",
    ci=None,  # Disable built-in error bars
    hue="finetuned",  # Set bar color
)
# Extract the x-positions of bars (bar centers)
bar_positions = []
for bar in ax.containers:  # Each container holds bars for one hue category
    for b in bar:
        bar_positions.append(b.get_x() + b.get_width() / 2)  # Get center of each bar

# Sort bars according to data order
bar_positions = sorted(set(bar_positions))  # Unique x positions

# Flattened list of y-values (performance scores) and std deviations
y_values = dataframe["performance"].values
y_errors = dataframe["performance_std"].values

# Add error bars manually
plt.errorbar(
    x=bar_positions,
    y=y_values,
    yerr=y_errors,
    fmt="none",  # No marker
    capsize=5,  # Cap size
    color="black",  # Error bar color
    elinewidth=1,  # Line width
)

# Labels and title
plt.ylabel("Performance Score")
plt.title("Model Performance with Standard Deviation")
plt.show()
