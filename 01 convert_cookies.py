import csv
import re
import os
import sys
import httpagentparser


outf = open('converted_cookies.csv', 'w')
writer= csv.writer(outf)

infilename = 'cookies.txt'
inf = open(infilename, 'r')
print infilename
reader=csv.reader(inf, delimiter='\t')
for row in reader:
    ob_id = abs(int(row[0]))
    info = sorted(row[1].split(';')) #why sort here ?
    #info = row[1].split(';')
    #should make alist of strings here
    age, gender, income, education, employment, children, household = 'unknown', 'unknown', 'unknown', 'unknown', 'unknown', 'unknown', 'unknown'
    for i in range(len(info)):
        name=info[i].split('_')[0]
        value=info[i].split('_')[1]
        if name == 'age':
            age = value
            continue
        elif name == 'gender':
            gender = value
            continue
        elif name == 'income':
            income = value
            continue
        elif name == 'education':
            education = value
            continue
        elif name == 'employment':
            employment = value
            continue
        elif name == 'children':
            children = value
            continue
        elif name == 'household':
            household = value
            continue



    #getting rid of 'unknown' gender entries.
    if gender=='unknown':
        continue


    
    user_agent = httpagentparser.detect(row[2])
    if 'os' in user_agent:
        os = user_agent['os']['name']
    elif 'dist' in user_agent:
        os = user_agent['dist']['name']
    elif 'flavour' in user_agent:
        os = user_agent['flavour']['name']
    else:
        os = 'unknown'
    try: 
        browser = user_agent['browser']['name']
    except:
        browser = 'unknown'
    writer.writerow([ob_id, age, gender, income, education, employment, children, household , os, browser])

inf.close()
outf.close()    
            
