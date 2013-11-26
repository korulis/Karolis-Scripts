import csv
import sys

#7057620,55,female,529,5,10,0,0.5,go.tv2.dk,samvirke.dk,www.degulesider.dk
    

def write_label_table_to_file(label_table,out_filename):
    print 'Working on writing to file '+out_filename+' ...'
    outf= open(out_filename,'w')
    writer = csv.writer(outf)
    for i in range(1,len(label_table)):
        writer.writerow(label_table[i])
    outf.close()
    print 'done'

def make_label_table(in_filename='uvd_cookies.csv',out_filename='cookie_table_of_labels.csv'):
    inf=  open(in_filename, 'r')
    contents = inf.readlines()
    inf.close()
    data_dimension =len(contents[0].split(','))
    label_table = [[] for i in range(data_dimension) ]
    print label_table, 'empty label_table', data_dimension
    for i in contents:
        cur_line =i.split(',')
        for j in range(data_dimension):
            label_table[j].append(cur_line[j])
    label_table = [sorted(list(set(label_table[i]))) for i in range(data_dimension)]
    label_table[-1] = [ label_table[-1][i][:-2] for i in range(len(label_table[-1]))]
##    print 'Printing table of labels:'
##    for i in range(1,data_dimension):
##        print label_table[i]
    all_domains= sorted(list(set(label_table[-1] + label_table[-2] +label_table[-3])))
    label_table[-1]=label_table[-2]=label_table[-3]= all_domains                                 
    write_label_table_to_file(label_table,out_filename)
#cookies
make_label_table()
#history
