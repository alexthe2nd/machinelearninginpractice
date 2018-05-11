import csv
import os

row_count = int(os.popen('wc -l < df_final.csv').read()[:-1])
keys = [x for x in range (1, 229)]

results = {}

with open ('df_final.csv', 'r') as table:
    rows = list(csv.reader(table))
    for i in range (1, 229):
        for j in range (1, row_count):
            if (str(rows[j][i]) == '1'):
                results.setdefault(rows[0][i], []).append(rows[j][0])
        print(rows[0][i])

with open('output.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in results.items():
       writer.writerow([key, value])
