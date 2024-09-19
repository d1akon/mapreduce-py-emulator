import os
import re
import math
from collections import defaultdict
import pickle

##########################
#  MAP REDUCE FUNCTIONS  #
##########################

def map_tf(id_review, review):
    """
    Function that performs the MapReduce mapping process. It calculates the term frequencies (TF) for each word in each review.
    EX: [(('if', '10_7'), (1, 181)), 
         (('you', '10_7'), (2, 181))]
    """
    words = re.split(';|,| |\.|-|!|\t|\"|&|\(|\)|\*|<|>|/|:|\'', review.lower())
    word_count = len(words)
    TF = defaultdict(int)
    
    for word in words:
        if word:
            TF[word] += 1
    
    results = []
    
    for word, count in TF.items():
        results.append(((word, id_review), (count, word_count)))

    return results

def map_tf_on_reviews(reviews):
    """
    Applies the TF calculation to each review, obtaining the frequency of each.
    """
    results = []
    for review_id, review_text in reviews:
        results.extend(map_tf(review_id, review_text))
    return results

def map_idf_from_tf(tf_results):
    """
    Maps the review IDs into a dictionary.
    EX: {'if': {'10_7', '1_10', '2_7', '5_7'}, 
         'you': {'7_9', '10_7', '2_7', '3_7', '6_7'}}
    """
    idf_input = defaultdict(set)
    for (word, id_review), _ in tf_results:
        idf_input[word].add(id_review)
    return idf_input

def calculate_idf(idf_input, total_docs):
    """
    Calculates the importance of the word within the set of documents, penalizing words that appear in many documents.
    EX: {'if': 0.9162907318741551, 
         'you': 0.6931471805599453}
    """
    idf_values = {}
    for word, doc_ids in idf_input.items():
        idf = math.log(total_docs / len(doc_ids))
        idf_values[word] = idf
    return idf_values

def calculate_tfidf(tf_results, idf_values):
    """
    Calculates TF-IDF by combining the frequency of a word in a document (TF) with the importance of that word in the set of documents (IDF).
    
    EX: [('10_7', 'if', 0.005062379734111354), 
         ('10_7', 'you', 0.007659084868065693)]
    """
    tfidf_results = []
    for (term, id_review), (tf, word_count) in tf_results:
        if term in idf_values:
            tfidf = (tf / word_count) * idf_values[term]
            tfidf_results.append((id_review, term, tfidf))
    return tfidf_results

##########################
#  SECONDARY FUNCTIONS   #
##########################

def read_documents(folder_path):
    """
    Function that reads .txt documents from the specified folder path.
    """
    reviews = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                review_id = filename.split('.')[0]  # Use the filename as id_review
                review_text = f.read().strip().split('\t')[1]  # The actual review text
                reviews.append((review_id, review_text))
    return reviews

# Filter the results for a specific term
def filter_term_tfidf(tfidf_results, term):
    """
    Filters the TF-IDF results to only include entries for the specified term.
    """
    filtered_results = [(doc_id, tfidf) for (doc_id, term_result, tfidf) in tfidf_results if term_result == term]
    return filtered_results

##########################
#    EXECUTION PIPELINE   #
##########################

# Folder containing review files
folder_path = 'data'
output_path = 'output/tf_results.pkl'

# Term you want to search for (e.g., "after")
search_term = 'after'

# Execute the full pipeline
def process_reviews(folder_path, search_term):
    # Read the reviews
    reviews = read_documents(folder_path)
    
    # Execute the Map task for TF
    tf_results = map_tf_on_reviews(reviews)
    
    # Execute the Map task for IDF
    idf_input = map_idf_from_tf(tf_results)
    
    # Calculate IDF values
    idf_values = calculate_idf(idf_input, len(reviews))
    
    # Calculate TF-IDF values
    tfidf_results = calculate_tfidf(tf_results, idf_values)
    
    # Filter the results for the specified term
    filtered_results = filter_term_tfidf(tfidf_results, search_term)
    
    # Save results
    save_tf_results(tf_results, output_path)
    
    return filtered_results

# Save the TF results to a file
def save_tf_results(tf_results, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(tf_results, f)


# Process the reviews and calculate TF-IDF for the specified term
tfidf_results = process_reviews(folder_path, search_term)

# Print TF-IDF results for the specified term
print(f"TF-IDF results for the term '{search_term}':")
for result in tfidf_results:
    print(f"Review ID: {result[0]}, TF-IDF: {result[1]}")
