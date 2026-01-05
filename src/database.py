import pymysql
import pandas as pd
from datetime import datetime

class IrisDatabase:
    def __init__(self, host='localhost', user='root', password='Abhi@8888', database='iris_db'):
        """Initialize database connection"""
        try:
            self.connection = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=database,
                cursorclass=pymysql.cursors.DictCursor
            )
            self.cursor = self.connection.cursor()
            print("✓ Connected to MySQL database successfully\n")
        except Exception as e:
            print(f"✗ Error connecting to database: {e}")
            raise
    
    def load_iris_data(self, csv_path='data/iris.csv'):
        """Load Iris dataset into database"""
        print("Loading data from CSV into database...")
        df = pd.read_csv(csv_path)
        
        insert_query = """
        INSERT INTO iris_data (sepal_length, sepal_width, petal_length, petal_width, species)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        count = 0
        for _, row in df.iterrows():
            self.cursor.execute(insert_query, (
                row['sepal length (cm)'],
                row['sepal width (cm)'],
                row['petal length (cm)'],
                row['petal width (cm)'],
                row['species_name']
            ))
            count += 1
        
        self.connection.commit()
        print(f"✓ Loaded {count} records into iris_data table\n")
    
    def insert_prediction(self, sepal_length, sepal_width, petal_length, petal_width, 
                          predicted_species, confidence):
        """Store prediction result"""
        insert_query = """
        INSERT INTO predictions (sepal_length, sepal_width, petal_length, petal_width, 
                                 predicted_species, confidence)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        self.cursor.execute(insert_query, (
            sepal_length, sepal_width, petal_length, petal_width, 
            predicted_species, confidence
        ))
        self.connection.commit()
        print("✓ Prediction saved to database")
    
    def get_species_statistics(self):
        """Get statistics by species"""
        query = """
        SELECT species, 
               COUNT(*) as count,
               ROUND(AVG(sepal_length), 2) as avg_sepal_length,
               ROUND(AVG(petal_length), 2) as avg_petal_length
        FROM iris_data
        GROUP BY species
        """
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        return pd.DataFrame(results)
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        query = f"""
        SELECT predicted_species, 
               ROUND(confidence * 100, 2) as confidence_percent, 
               prediction_date
        FROM predictions
        ORDER BY prediction_date DESC
        LIMIT {limit}
        """
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        return pd.DataFrame(results)
    
    def close(self):
        """Close database connection"""
        self.cursor.close()
        self.connection.close()
        print("\n✓ Database connection closed")

# Test the database
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DATABASE CONNECTION")
    print("=" * 60)
    print()
    
    # Initialize database
    db = IrisDatabase(host='localhost', user='root', password='Abhi@8888', database='iris_db')
    
    # Load Iris data
    db.load_iris_data()
    
    # Get statistics
    print("Species Statistics:")
    print("-" * 60)
    print(db.get_species_statistics())
    print()
    
    # Insert sample predictions
    print("Inserting sample predictions...")
    db.insert_prediction(5.1, 3.5, 1.4, 0.2, 'Setosa', 0.98)
    db.insert_prediction(6.3, 2.5, 4.9, 1.5, 'Versicolor', 0.85)
    db.insert_prediction(7.2, 3.6, 6.1, 2.5, 'Virginica', 0.99)
    print()
    
    # Get recent predictions
    print("Recent Predictions:")
    print("-" * 60)
    print(db.get_recent_predictions())
    
    # Close connection
    db.close()
    
    print("\n" + "=" * 60)
    print("✓ DATABASE TEST COMPLETE!")
    print("=" * 60)
