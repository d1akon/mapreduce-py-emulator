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
    Map function for MapReduce. It calculates the term frequency (TF) for each word in a review.
    
    Args:
        review_id (str): The ID of the review.
        review (str): The text of the review.
        context (object): Context object to emit key-value pairs during the MapReduce job.
    """
    words = re.split(';|,| |\.|-|!|\t|\"|&|\(|\)|\*|<|>|/|:|\'', review.lower())
    word_count = len(words)
    TF = defaultdict(int)
    
    for word in words:
        if word:
            TF[word] += 1
    
    for word, count in TF.items():
        context.write(word, (review_id, count, word_count))  # Emit (word, (review_id, TF, total words))

def reduce_tf_idf(word, values_iterator, context):
    """
    Reduce function that calculates the TF-IDF of a word based on the number of documents it appears in.
    
    Args:
        word (str): The word for which TF-IDF is calculated.
        values_iterator (iterable): An iterator over the values associated with the word.
        context (object): Context object to emit results during the Reduce phase.
    """
    docs = []
    for value in values_iterator:
        docs.append(value)
    
    # Calculate IDF
    total_docs = context[0]  # context[0] contains the total number of documents
    idf = math.log(total_docs / len(docs))
    
    # Calculate TF-IDF
    for review_id, tf, total_words in docs:
        tfidf = (tf / total_words) * idf
        context.write(review_id, (word, tfidf))  # Emit (review_id, (word, TF-IDF))

##########################
#  HELPER FUNCTIONS  #
##########################

def read_documents(folder_path):
    """
    Reads .txt documents from the specified folder and returns a list of reviews.
    
    Args:
        folder_path (str): Path to the folder containing .txt files.
    
    Returns:
        list: A list of tuples, each containing the review ID and the review text.
    """
    reviews = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                review_id = filename.split('.')[0]  # Take the filename as review_id
                review_text = f.read().strip().split('\t')[1]  # The actual review text
                reviews.append((review_id, review_text))
    return reviews

def filter_results_by_word(tfidf_results, term):
    """
    Filters the results to display only the exact matches for a specific word.
    
    Args:
        tfidf_results (list): A list of TF-IDF results.
        term (str): The term to filter results for.
    
    Returns:
        list: A list of tuples containing the review ID and the TF-IDF score for the given term.
    """
    filtered_results = [(review_id, tfidf) for (review_id, word, tfidf) in tfidf_results if word == term]
    return filtered_results

##########################
# JOB CONFIGURATION AND EXECUTION #
##########################
# Input and output paths
input_path = 'data'  
output_path = 'output'  
word_to_filter = 'after'

# Initialize the MapReduce job
job = Job(input_path, output_path, map_tf, reduce_tf_idf)

# Set the total number of documents (context for calculating IDF)
total_docs = len([name for name in os.listdir(input_path) if name.endswith(".txt")])
job.setParams([total_docs])

# Execute the job
job.waitForCompletion()

##########################
# RESULT FILTERING #
##########################

def filter_results_by_term(output_path, term):
    """
    Filters the output file to display TF-IDF results for a specific term.
    
    Args:
        output_path (str): Path to the output file.
        term (str): The term to filter results for.
    """
    output_file = os.path.join(output_path, 'output.txt')
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    print(f"TF-IDF results for the term '{term}':")
    for line in lines:
        columns = line.strip().split('\t')
        if len(columns) > 1 and columns[1] == term:  # Compare exact term
            print(line.strip())

# Filter results for a specific search term
filter_results_by_term(output_path, word_to_filter)
