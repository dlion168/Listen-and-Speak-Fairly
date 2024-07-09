import pandas as pd

# Step 1: Read the TSV file and transform the 'id' column
tsv_file_path = '/home/ycevan/speech_llm_bias_eval/mt/en_subset.tsv'
tsv_data = pd.read_csv(tsv_file_path, sep='\t')

# Assuming the column name is 'id'
tsv_data['id'] = tsv_data['id'] - 1
id_list = tsv_data['id'].tolist()

# Step 2: Read the text file and keep only the id-th lines
text_file_path = '/home/ycevan/speech_llm_bias_eval/mt/mt_gender/translations/google/en-it.txt'
with open(text_file_path, 'r') as file:
    lines = file.readlines()

# Filter lines based on the transformed id list
filtered_lines = [lines[i] for i in id_list if i < len(lines)]

# Step 3: Save the results to a new file or print them
output_file_path = '/home/ycevan/speech_llm_bias_eval/mt/mt_gender/translations/test/en-it.txt'
with open(output_file_path, 'w') as output_file:
    output_file.writelines(filtered_lines)

# Optionally, print the filtered lines
# for line in filtered_lines:
#     print(line.strip())
