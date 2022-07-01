import pandas as pd 
import sys, getopt

def process_data(argv):

    opts, args = getopt.getopt(argv,"n:",["min_num_products="])
    
    min_num_products = 500

    for opt, arg in opts:
        if opt == '-n':
            min_num_products = int(arg)
    
    with open('/workspace/datasets/fasttext/labeled_products.txt', 'r') as file:
        data = file.readlines()

    processed_data = []
    for line in data:
        items = line.split(' ')
        label = items[0]
        title = ' '.join(items[1:]).replace('\n','')
        processed_data.append([label, title])

    df = pd.DataFrame(data=processed_data, columns = ['label', 'title'])
    category_counts = df.groupby('label').count().reset_index().sort_values('title', ascending=False)
    valid_labels = category_counts[category_counts.title >= min_num_products].label.values

    df = df[df.label.isin(valid_labels)]
    
    with open('/workspace/datasets/fasttext/pruned_labeled_products.txt', 'w+') as file:
        for i,row in df.iterrows():
            file.write(f'{row.label} {row.title}\n')
if __name__ == '__main__':
    process_data(sys.argv[1:])