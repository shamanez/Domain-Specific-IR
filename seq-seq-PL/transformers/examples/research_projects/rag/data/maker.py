import csv
import re
import regex
csvfile = open('nq-train.qa.csv', 'r')

csv_reader = csv.reader(csvfile, delimiter='\t')

source_file=open('train.source', 'a')
target_file=open('train.target', 'a')

line_count = 0
for row in csv_reader:

 
    src= row[0]

    tgt=row[1]

    tgt=regex.sub(r'\p{P}','', tgt)


    source_text=src.replace('\n', ' ')
    target_text=tgt.replace('\n', ' ')



    
    line_count=line_count+1
    source_file.write(source_text+'\n')
    target_file.write(target_text+'\n')


print(f'Processed {line_count} lines.')