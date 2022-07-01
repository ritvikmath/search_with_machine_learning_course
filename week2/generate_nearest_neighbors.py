import fasttext
import sys, getopt

model = fasttext.load_model('/workspace/datasets/fasttext/title_model.bin')

def generate_similar_words(argv):

    opts, args = getopt.getopt(argv,"t:",["thresh="])

    thresh = 0.75

    for opt, arg in opts:
        if opt == '-t':
            thresh = float(arg)

    with open('/workspace/datasets/fasttext/top_words.txt', 'r') as file, open('/workspace/datasets/fasttext/synonyms.csv', 'w+') as synonyms:
        for line in file.readlines():
            word = line.replace('\n', '')
            closest_neighbors = model.get_nearest_neighbors(word, k=100)
            closest_neighbors = [c[1] for c in closest_neighbors if c[0] >= thresh]
            if len(closest_neighbors) >= 1:
                all_syns = ' '.join(closest_neighbors)
                synonyms.write(f'{word}, {all_syns}\n')

                

if __name__ == '__main__':
    generate_similar_words(sys.argv[1:])