import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BookRecommender:
    def __init__(self):
        self.books_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None

    def load_data(self, filepath):
        try:
            self.books_df = pd.read_csv(filepath)
            return True
        except FileNotFoundError:
            print(f"Error: Could not find {filepath}")
            return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def preprocess_data(self):
        """Combine relevant features for content-based filtering"""
        if self.books_df is None:
            raise ValueError("No data loaded. Please load data first.")
            
        self.books_df['features'] = (self.books_df['title'] + ' ' + 
                                   self.books_df['author'] + ' ' + 
                                   self.books_df['genre'] + ' ' + 
                                   self.books_df['year'].astype(str))

    def create_similarity_matrix(self):
        if self.books_df is None:
            raise ValueError("No data loaded. Please load data first.")
            
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.books_df['features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_recommendations(self, book_title, n_recommendations=5, genre_filter=None):
        """
        Get book recommendations based on similarity
        Args:
            book_title (str): Title of the book to base recommendations on
            n_recommendations (int): Number of recommendations to return
            genre_filter (str, optional): Filter recommendations by genre
        """
        try:
            # Find the index of the book
            idx = self.books_df[self.books_df['title'] == book_title].index[0]
        except IndexError:
            return f"Book '{book_title}' not found in database."

        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Filter by genre if specified
        if genre_filter:
            genre_indices = self.books_df[self.books_df['genre'] == genre_filter].index
            sim_scores = [s for s in sim_scores if s[0] in genre_indices]

        # Sort books by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N most similar books (excluding the input book)
        sim_scores = [s for s in sim_scores if s[0] != idx][:n_recommendations]
        
        # Get book indices and similarity scores
        book_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]

        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'title': self.books_df['title'].iloc[book_indices],
            'author': self.books_df['author'].iloc[book_indices],
            'genre': self.books_df['genre'].iloc[book_indices],
            'year': self.books_df['year'].iloc[book_indices],
            'similarity_score': similarity_scores
        })

        return recommendations

    def get_available_genres(self):
        """Return list of all available genres"""
        return sorted(self.books_df['genre'].unique())
