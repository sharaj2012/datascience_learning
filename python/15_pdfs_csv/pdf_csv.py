#%%
import csv
data_set = open('data/example.csv', encoding='utf-8')
csv_data = csv.reader(data_set)
data_lines = list(csv_data)

# %%
data_lines[0]

# %%
email_id = []
for line in data_lines[:15]:
    email_id.append(line[3])


# %%
email_id
# %%
file_to_output = open('xyz.csv',mode = 'w',newline='')
csv_fileoutput = csv.writer(file_to_output)
csv_fileoutput.writerow([1,2,3])
# %%
pwd
# %%
csv_fileoutput.writerows([[1,32,4],[5,6,7]])
# %%
file_to_output.close()
# %%
###########################################
