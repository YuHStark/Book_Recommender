import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    def __init__(self, processed_data_path):
        """Initialize the recommendation engine with processed book data."""
        print(f"Loading data from {processed_data_path}...")
        
        # Handle compressed files if needed
        if processed_data_path.endswith('.gz'):
            self.df = pd.read_csv(processed_data_path, compression='gzip')
        else:
            self.df = pd.read_csv(processed_data_path)
        
        print(f"Loaded {len(self.df)} books")
        
        # Load the model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knn_model = None
        self.embeddings = None
        self.decision_tree = None
        
        # Generate embeddings for the dataset
        print("Generating embeddings for book descriptions...")
        self.embeddings = self.model.encode(self.df['book_desc'].tolist(), show_progress_bar=True)
        print(f"Generated embeddings with shape: {self.embeddings.shape}")
        
        # Initialize KNN model for similar book recommendations
        self._init_knn_model()
    
    def _init_knn_model(self):
        """Initialize the K-nearest neighbors model for finding similar books."""
        if self.embeddings is None:
            # If embeddings weren't generated earlier, do it now
            self.embeddings = self.model.encode(self.df['book_desc'].tolist(), show_progress_bar=True)
        
        self.knn_model = NearestNeighbors(n_neighbors=6, algorithm='auto', metric='cosine')
        self.knn_model.fit(self.embeddings)
    
    def content_based_filter(self, genre=None, style=None, min_rating=0, max_pages=None):
        """
        Filter books based on content criteria.
        
        Args:
            genre (str): The genre to filter for
            style (str): The writing style to look for
            min_rating (float): Minimum rating threshold
            max_pages (int): Maximum number of pages
            
        Returns:
            DataFrame containing filtered book recommendations
        """
        filtered_df = self.df.copy()
        
        # Filter by genre if specified
        if genre and genre != "":
            genre_pattern = re.compile(genre, re.IGNORECASE)
            filtered_df = filtered_df[filtered_df['genres'].apply(
                lambda x: bool(genre_pattern.search(str(x))))]
        
        # Filter by minimum rating
        if min_rating > 0:
            filtered_df = filtered_df[filtered_df['book_rating'] >= min_rating]
        
        # Filter by maximum page count
        if max_pages and max_pages > 0:
            filtered_df = filtered_df[filtered_df['book_pages'] <= max_pages]
        
        # Filter by style (text search in description) if specified
        if style and style != "":
            style_pattern = re.compile(style, re.IGNORECASE)
            filtered_df = filtered_df[filtered_df['book_desc'].apply(
                lambda x: bool(style_pattern.search(str(x))))]
        
        # Sort by rating
        filtered_df = filtered_df.sort_values(by='book_rating', ascending=False)
        
        return filtered_df.head(5)  # Return top 5 books
    
    def popularity_rank_recommend(self, min_rating=0, max_pages=None, top_n=5):
        """
        Recommend books based on popularity score.
        
        Args:
            min_rating (float): Minimum rating threshold
            max_pages (int): Maximum number of pages
            top_n (int): Number of recommendations to return
            
        Returns:
            DataFrame containing popular book recommendations
        """
        # Calculate popularity score if not already in the dataframe
        if 'popularity_score' not in self.df.columns:
            self.df['popularity_score'] = (
                self.df['book_rating'] * self.df['book_rating_count']
            ) / (self.df['book_rating_count'] + 10)
        
        filtered_df = self.df.copy()
        
        # Apply filters
        if min_rating > 0:
            filtered_df = filtered_df[filtered_df['book_rating'] >= min_rating]
        
        if max_pages and max_pages > 0:
            filtered_df = filtered_df[filtered_df['book_pages'] <= max_pages]
        
        # Sort by popularity score
        filtered_df = filtered_df.sort_values(by='popularity_score', ascending=False)
        
        return filtered_df.head(top_n)
    
    def train_decision_tree(self):
        """Train a decision tree classifier for book recommendation."""
        # This is a simplified version for demonstration
        # In a real system, you'd need user feedback data
        
        # Creating sample labels (liked/not liked) based on rating for demonstration
        self.df['liked'] = (self.df['book_rating'] >= 4.0).astype(int)
        
        # Features for the decision tree - using genre and page count
        # In a real system, you would have more meaningful features
        X = pd.concat([
            pd.get_dummies(self.df['genres'].apply(lambda x: str(x).split('|')[0])),
            self.df[['book_pages']]
        ], axis=1)
        
        y = self.df['liked']
        
        # Train the decision tree
        self.decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.decision_tree.fit(X, y)
        
        print("Decision tree classifier trained")
    
    def decision_tree_recommend(self, genre, style=None, min_rating=0, max_pages=None, top_n=5):
        """
        Use the decision tree to recommend books.
        This is a simplified implementation.
        """
        if self.decision_tree is None:
            self.train_decision_tree()
        
        # Start with content-based filtering
        filtered_df = self.content_based_filter(genre, style, min_rating, max_pages)
        
        # Sort by predicted likelihood using our simple model
        # In a real system, you'd use actual decision tree predictions
        filtered_df = filtered_df.sort_values(by=['book_rating', 'book_rating_count'], 
                                              ascending=[False, False])
        
        return filtered_df.head(top_n)
    
    def find_similar_books_knn(self, book_title, n=3):
        """
        Find books similar to the given book title using KNN.
        
        Args:
            book_title (str): Title of the reference book
            n (int): Number of similar books to return
            
        Returns:
            DataFrame containing similar book recommendations
        """
        # Find the book index
        book_indices = self.df[self.df['book_title'].str.contains(book_title, case=False, na=False)].index
        
        if len(book_indices) == 0:
            return pd.DataFrame()  # Book not found
        
        book_idx = book_indices[0]  # Take the first match
        
        # Get the embedding for the query book
        book_embedding = self.embeddings[book_idx].reshape(1, -1)
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(book_embedding)
        
        # Skip the first result as it's the query book itself
        similar_books = self.df.iloc[indices[0][1:n+1]]
        
        return similar_books
    
    def ensemble_recommendations(self, genre=None, style=None, min_rating=0, max_pages=None, top_n=5):
        """
        Combine multiple recommendation algorithms for better results.
        
        Args:
            genre (str): Preferred book genre
            style (str): Preferred writing style
            min_rating (float): Minimum book rating
            max_pages (int): Maximum book pages
            top_n (int): Number of recommendations to return
            
        Returns:
            DataFrame containing ensemble book recommendations
        """
        # Get recommendations from each algorithm
        content_recs = self.content_based_filter(genre, style, min_rating, max_pages)
        popularity_recs = self.popularity_rank_recommend(min_rating, max_pages)
        decision_recs = self.decision_tree_recommend(genre, style, min_rating, max_pages)
        
        # Combine and deduplicate recommendations
        # Using a scoring system that prioritizes books recommended by multiple algorithms
        all_books = pd.concat([
            content_recs.assign(algorithm='content'),
            popularity_recs.assign(algorithm='popularity'),
            decision_recs.assign(algorithm='decision_tree')
        ])
        
        # Count occurrences of each book across algorithms
        book_counts = all_books.groupby('book_id').size().reset_index(name='algorithm_count')
        
        # Join with the original recommendations to get all metadata
        scored_recs = all_books.drop_duplicates(subset='book_id').merge(
            book_counts, on='book_id', how='left')
        
        # Sort by count (number of algorithms that recommended it) then by rating
        final_recs = scored_recs.sort_values(
            by=['algorithm_count', 'book_rating', 'popularity_score'], 
            ascending=[False, False, False]
        )
        
        return final_recs.head(top_n)[['book_id', 'book_title', 'book_authors', 
                                      'book_rating', 'book_pages', 'genres']]
    
    def get_book_details(self, book_title):
        """
        Get detailed information about a specific book.
        
        Args:
            book_title (str): Title of the book
            
        Returns:
            dict containing book details or None if not found
        """
        book = self.df[self.df['book_title'].str.contains(book_title, case=False, na=False)]
        
        if len(book) == 0:
            return None
        
        # Return the first matching book's details
        book = book.iloc[0]
        
        details = {
            'title': book['book_title'],
            'author': book['book_authors'],
            'description': book['book_desc'],
            'genres': book['genres'],
            'rating': book['book_rating'],
            'rating_count': book['book_rating_count'],
            'pages': book['book_pages'],
            'format': book.get('book_format', 'Unknown')
        }
        
        return details

# Example usage
if __name__ == "__main__":
    engine = RecommendationEngine("processed_books.csv")
    
    
