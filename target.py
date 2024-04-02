import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('drone.csv')

# Define the new column name and the value you want to assign to all rows
new_column_name = 'target'
new_column_value = "1"

# Add a new column with the same value to the DataFrame
df[new_column_name] = new_column_value

# Save the updated DataFrame to a new CSV file
df.to_csv('drone.csv', index=False)

print("New column added successfully to 'output_file.csv'.")
