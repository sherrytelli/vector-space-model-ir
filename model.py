import os
import math
import nltk
import pickle
import natsort
import collections

def create_doc_vector():
    """helper to initilaize document vector"""
    return { 'terms': collections.defaultdict(float), 'magnitude': 0.0 }

class VSM:
    def __get_stop_words(self):
        """read the stop words from the text file"""
        file = open(file=".\\Stopword-List.txt")
        self.__stop_words = file.read()
        self.__stop_words = nltk.word_tokenize(self.__stop_words)
        file.close()

    def __preprocess(self, content):
        """function to preprocess a document"""
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokens = nltk.word_tokenize(content.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in self.__stop_words]
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmas
    
    def __read_documents(self):
        """function to read all documents"""
        docs = {}
        files = ["Abstracts\\"+str(document) for document in os.listdir(path=".\\Abstracts\\") if document.endswith(".txt")]
        files = natsort.natsorted(files)
        self.__N = len(files)
        for file_name in files:
            file = open(file=file_name)
            content = file.read()
            docs[file_name] = content
            file.close()
        return docs
    
    def __build_inverted_index(self):
        """function to build inverted index and calculate term document frequencies"""
        docs = self.__read_documents()

        self.__inverted_index = collections.defaultdict(dict)
        self.__document_frequencies = collections.defaultdict(int)
        
        #loading preexisting inverted index and document frequencies from files if they exist
        if(os.path.exists(".\\inverted_index.pkl") and os.path.exists(".\\document_frequencies.pkl")):
            inverted_index_file = open(file=".\\inverted_index.pkl", mode="rb")
            self.__inverted_index = pickle.load(inverted_index_file)
            inverted_index_file.close()
            
            document_frequencies_file = open(file=".\\document_frequencies.pkl", mode="rb")
            self.__document_frequencies = pickle.load(document_frequencies_file)
            document_frequencies_file.close()
            
        #creating inverted index and calculating document frequencies    
        else: 
            for doc_id, content in docs.items():
                terms = self.__preprocess(content)
                term_counts = collections.defaultdict(int)
                #calculating term counts
                for term in terms:
                    term_counts[term] += 1
                #creating inverted index and document frequencies
                for term, count in term_counts.items():
                    self.__inverted_index[term][doc_id] = count
                    self.__document_frequencies[term] += 1
            
            #storing inverted index        
            inverted_index_file = open(file=".\\inverted_index.pkl", mode="wb")
            pickle.dump(obj=self.__inverted_index, file=inverted_index_file)
            inverted_index_file.close()
            
            #storing document frequencies
            document_frequency_file = open(file=".\\document_frequency.pkl", mode="wb")
            pickle.dump(obj=self.__document_frequencies, file=document_frequency_file)
            document_frequency_file.close()

    def __compute_doc_vectors(self):
        """fucntion to create document vectors and calculate idf"""
        self.__doc_vectors = collections.defaultdict(create_doc_vector)
        self.__idf_cache = {}
        
        #loading document vectors and idf values from files if they exist
        if(os.path.exists(".\\doc_vectors.pkl") and os.path.exists(".\\idf_cache.pkl")):
            doc_vectors_file = open(file=".\\doc_vectors.pkl", mode="rb")
            self.__doc_vectors = pickle.load(doc_vectors_file)
            doc_vectors_file.close()
            
            idf_cache_file = open(file=".\\idf_cache.pkl", mode="rb")
            self.__idf_cache = pickle.load(idf_cache_file)
            idf_cache_file.close()
            
        #creating document vectors and calculating idfs
        else:
            #calculating idf
            for term, df in self.__document_frequencies.items():
                self.__idf_cache[term] = math.log(self.__N/df)
            
            #storing tfidf values in documents
            for term, doc_counts in self.__inverted_index.items():
                idf = self.__idf_cache.get(term, 0.0)
                for doc_id, tf in doc_counts.items():
                    tfidf = tf * idf
                    self.__doc_vectors[doc_id]['terms'][term] += tfidf
            
            #storing magnitute of document vectors
            for doc_id in self.__doc_vectors:
                terms = self.__doc_vectors[doc_id]['terms']
                sum_squares = sum(tfidf ** 2 for tfidf in terms.values())
                self.__doc_vectors[doc_id]['magnitude'] = math.sqrt(sum_squares) if sum_squares != 0 else 0.0
                
            #storing document vectors
            doc_vectors_file = open(file=".\\doc_vectors.pkl", mode="wb")
            pickle.dump(obj=self.__doc_vectors, file=doc_vectors_file)
            doc_vectors_file.close()
            
            #storing idf values
            idf_cache_file = open(file=".\\idf_cache.pkl", mode="wb")
            pickle.dump(obj=self.__idf_cache, file=idf_cache_file)
            idf_cache_file.close()

    def process_query(self, query, alpha=0.05):
        """function to process query and return relevent documents"""
        #precprocessing query
        terms = self.__preprocess(query) 

        #if query empty return empty set
        if not terms:
            return []
        
        #calculating query term frequencies
        query_tf = collections.defaultdict(int)
        for term in terms:
            query_tf[term] += 1
        
        #creating query tfidf vector
        query_vector = {}
        for term, tf in query_tf.items():
            if term in self.__idf_cache:
                query_vector[term] = tf * self.__idf_cache[term]
        
        #if query_vector empty return empty set
        if not query_vector:
            return []
        
        #calculating query magnitude
        query_mag = math.sqrt(sum(score ** 2 for score in query_vector.values()))
        if query_mag == 0:
            return []
        
        #creating a relevent doc set based on the terms in the query
        relevant_docs = set()
        for term in query_vector:
            relevant_docs.update(self.__inverted_index.get(term, {}).keys())
        
        #scoring the documents in the relevent set using cosine similarity
        scores = []
        for doc_id in relevant_docs:
            doc_info = self.__doc_vectors.get(doc_id, {'terms': collections.defaultdict(float), 'magnitude': 0.0})
            doc_terms = doc_info['terms']
            doc_mag = doc_info['magnitude']
            if doc_mag == 0:
                continue
            
            dot_product = sum(doc_terms.get(term, 0.0) * q_score for term, q_score in query_vector.items())
            cosine = dot_product / (doc_mag * query_mag) if (doc_mag * query_mag) != 0 else 0.0
            
            #if scores greater than alpha only then add them to result set
            if cosine >= alpha:
                scores.append((doc_id, cosine))
        
        #sorting result set
        scores = sorted(scores, key=lambda x: (-x[1], x[0]))
        
        #extracting document numbers from the result set scores
        doc_numbers = []
        for doc_id, _ in scores:
            doc_num = doc_id.split('\\')[-1].replace('.txt', '')
            doc_numbers.append(int(doc_num))
        
        #returning relevent document numbers
        return doc_numbers
    
    def __init__(self):
        self.__get_stop_words()
        self.__build_inverted_index()
        self.__compute_doc_vectors()