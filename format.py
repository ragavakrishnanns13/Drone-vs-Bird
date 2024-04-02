with open('helicopter.txt', 'r') as file:
    # Read the content of the file
    file_content = file.read()

csv_content = file_content.replace(' ', ',')

# Open a new file in write mode and write the CSV content
with open('helicopter.csv', 'w') as output_file:
    output_file.write(csv_content)

print("File converted successfully from spaces to commas.")
