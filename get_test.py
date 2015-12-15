import os, csv, json, codecs
import csv, getpass, json, os, time, sys



def main():
    input_file = 'testdata.manual.2009.06.14.csv' #'resources/test_tweets_new.csv'
    output_file = 'test'
    
    # read test file
    data = []
    with open(input_file, 'r') as input_fh:
        for line in input_fh:
            fields = line.decode('utf8').split(u'","')
            if len(fields) != 6:
                continue
            temp = [fields[i].strip().replace('"', '') for i in [0, 5]]
            # Positve/Negative
            if temp[0] == '0' or temp[0] == '4':
                data.append(u'\t'.join([unicode(r) for r in temp]) + u'\n')
    write_file(output_file, data)

def write_file(filename, lines):
    with open(filename, 'wb') as file_fh:
        for line in lines:
            try:
                file_fh.write(line.encode('utf8'))
            except TypeError as exc:
                #print row[4]
                raise exc

if __name__ == '__main__':
    main()