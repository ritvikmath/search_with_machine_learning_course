import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import string

from nltk.stem.snowball import SnowballStemmer

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

#map of category to its parent category
parents_lookup = dict(zip(parents_df.category, parents_df.parent))
parents_lookup[root_category_id] = root_category_id

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
print('lowercasing')
df['query'] = df['query'].apply(lambda q: q.lower())
print('removing punctuation')
df['query'] = df['query'].apply(lambda q: q.translate(str.maketrans('', '', string.punctuation)))
print('stemming')
stemmer = SnowballStemmer("english")
df['query'] = df['query'].apply(lambda s: ' '.join([stemmer.stem(w) for w in s.split()]))

print('rolling')
# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
min_products_per_category = int(args.min_queries)
while True:
    grouped = df.groupby('category').count().reset_index()[['category', 'query']]
    grouped.columns = ['category', 'num']
    categories_to_roll = set(grouped[grouped.num < min_products_per_category].category.values)
    if len(categories_to_roll) == 0:
        break
    curr_mapping = {c:(c if c not in categories_to_roll else parents_lookup[c]) for c in grouped.category.values}
    df.category = df.category.apply(lambda c: curr_mapping[c])
    print(f'num_categories: {len(set(df.category.values))}')

# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
