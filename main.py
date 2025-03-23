from recommender import BookRecommender

def main():
    # Initialize recommender
    recommender = BookRecommender()
    
    # Load data
    if not recommender.load_data('books.csv'):
        return
    
    # Preprocess data and create similarity matrix
    try:
        recommender.preprocess_data()
        recommender.create_similarity_matrix()
    except Exception as e:
        print(f"Error preparing recommendation system: {str(e)}")
        return

    while True:
        print("\nBook Recommendation System")
        print("1. Get recommendations")
        print("2. Get recommendations by genre")
        print("3. Show available genres")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            book_title = input("Enter book title: ")
            recommendations = recommender.get_recommendations(book_title)
            if isinstance(recommendations, str):
                print(recommendations)
            else:
                print(f"\nRecommendations for '{book_title}':")
                for _, row in recommendations.iterrows():
                    print(f"\nTitle: {row['title']}")
                    print(f"Author: {row['author']}")
                    print(f"Genre: {row['genre']}")
                    print(f"Year: {row['year']}")
                    print(f"Similarity Score: {row['similarity_score']:.2f}")
                    
        elif choice == '2':
            book_title = input("Enter book title: ")
            genre = input("Enter genre to filter by: ")
            recommendations = recommender.get_recommendations(book_title, genre_filter=genre)
            if isinstance(recommendations, str):
                print(recommendations)
            else:
                print(f"\nRecommendations for '{book_title}' in genre '{genre}':")
                for _, row in recommendations.iterrows():
                    print(f"\nTitle: {row['title']}")
                    print(f"Author: {row['author']}")
                    print(f"Genre: {row['genre']}")
                    print(f"Year: {row['year']}")
                    print(f"Similarity Score: {row['similarity_score']:.2f}")
                    
        elif choice == '3':
            genres = recommender.get_available_genres()
            print("\nAvailable genres:")
            for genre in genres:
                print(f"- {genre}")
                
        elif choice == '4':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
