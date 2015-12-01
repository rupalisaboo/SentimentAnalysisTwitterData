import os

def main():
    input_file = 'resources/tweets_new.csv' #'resources/test_tweets_new.csv'
    input_file_pos = 'resources/tweets_new_pos'
    input_file_neg = 'resources/tweets_new_neg'
    positive_label_count = 0
    negative_label_count = 0
    #max_positive = 2500
    #max_negative = 2500
    # read input corpus file
    pos_lines = []
    neg_lines = []
    with open(input_file, 'r') as input_fh:
        for line in input_fh:
            row = line.split('","')
            if row[0].replace('"', '') == '0':
                neg_lines.append(line)
            elif row[0].replace('"', '') == '4':
                pos_lines.append(line)
    pos = len(pos_lines)
    neg = len(neg_lines)
    with open(input_file_neg + '1.csv', 'wb') as input_fh_neg1:
        input_fh_neg1.write('\n'.join(neg_lines[0: (neg / 3)]))
    with open(input_file_neg + '2.csv', 'wb') as input_fh_neg2:
        input_fh_neg2.write('\n'.join(neg_lines[(neg / 3): (2 * neg / 3)]))
    with open(input_file_neg + '3.csv', 'wb') as input_fh_neg3:
        input_fh_neg3.write('\n'.join(neg_lines[(2 * neg / 3): ]))

    
    with open(input_file_pos + '1.csv', 'wb') as input_fh_pos1:
        input_fh_pos1.write('\n'.join(pos_lines[0:(pos / 3)]))
    with open(input_file_pos + '2.csv', 'wb') as input_fh_pos2:
        input_fh_pos2.write('\n'.join(pos_lines[(pos / 3): (2 * pos / 3)]))
    with open(input_file_pos + '3.csv', 'wb') as input_fh_pos3:
        input_fh_pos3.write('\n'.join(pos_lines[(2 * pos / 3):]))

if __name__ == '__main__':
    main()