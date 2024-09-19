import os
import re
import math
from collections import defaultdict
from MRE import Job  # Import the MRE simulator

##########################
#  MAP REDUCE FUNCTIONS  #
##########################

def map_tf(review_id, review, context):
    """
    Function that performs the Map process of MapReduce. It calculates term frequencies (TF) for each word of each review.
    """
    words = re.split(';|,| |\.|-|!|\t|\"|&|\(|\)|\*|<|>|/|:|\'', review.lower())
    word_count = len(words)
    TF = defaultdict(int)
    
    for word in words:
        if word:
            TF[word] += 1
    
    for word, count in TF.items():
        context.write(word, (review_id, count, word_count))  # Emit (term, (Review ID, TF, total words))

def reduce_tf_idf(word, values_iterator, context):
    """
    Reduction function that calculates the TF-IDF of a word based on the number of documents it appears in.
    """
    documents = []
    for value in values_iterator:
        documents.append(value)
    
    # Calculate IDF
    total_docs = context[0]  # context[0] has the total number of documents
    idf = math.log(total_docs / len(documents))
    
    # Calculate TF-IDF
    for review_id, tf, total_words in documents:
        tfidf = (tf / total_words) * idf
        context.write(review_id, (word, tfidf))  # Emit (Review ID, (word, TF-IDF))

##########################
# AUXILIARY FUNCTIONS    #
##########################

def read_documents(folder_path):
    """
    Function that reads .txt documents based on the provided path.
    """
    reviews = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                review_id = filename.split('.')[0]  # Use the filename as review_id
                review_text = f.read().strip().split('\t')[1]  # The actual review text
                reviews.append((review_id, review_text))
    return reviews

# Filter results for a specific term
def select_word(tfidf_results, term):
    """
    Filters the results for a specific term, returning only exact matches.
    """
    filtered_results = [(review_id, tfidf) for (review_id, word, tfidf) in tfidf_results if word == term]
    return filtered_results

##########################
# JOB CONFIGURATION AND EXECUTION #
##########################

# Input path (folder with text files)
input_path = 'data'  # Directory where your review files are located
output_path = 'output'  # Output directory where results will be saved

# Initialize the MapReduce job
job = Job(input_path, output_path, map_tf, reduce_tf_idf)

# Set the total number of documents (context for calculating IDF)
total_docs = len([name for name in os.listdir(input_path) if name.endswith(".txt")])
job.setParams([total_docs])

# Execute the job
job.waitForCompletion()

##########################
# RESULTS FILTERING      #
##########################

def filter_results_by_word(output_path, term):
    output_file = os.path.join(output_path, 'output.txt')
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    print(f"TF-IDF results for the term '{term}':")
    for line in lines:
        # Split the line into columns to get the exact term
        columns = line.strip().split('\t')
        if len(columns) > 1 and columns[1] == term:  # Compare the exact term
            print(line.strip())

# Filter the results for the search term
filter_results_by_word(output_path, 'after')
