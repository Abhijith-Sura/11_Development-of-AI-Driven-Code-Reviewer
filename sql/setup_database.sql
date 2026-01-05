-- Create database
CREATE DATABASE IF NOT EXISTS iris_db;
USE iris_db;

-- Table 1: Store Iris training data
CREATE TABLE IF NOT EXISTS iris_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sepal_length FLOAT NOT NULL,
    sepal_width FLOAT NOT NULL,
    petal_length FLOAT NOT NULL,
    petal_width FLOAT NOT NULL,
    species VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: Store prediction history
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sepal_length FLOAT NOT NULL,
    sepal_width FLOAT NOT NULL,
    petal_length FLOAT NOT NULL,
    petal_width FLOAT NOT NULL,
    predicted_species VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample query 1: Get statistics by species
SELECT 
    species,
    COUNT(*) as total_count,
    ROUND(AVG(sepal_length), 2) as avg_sepal_length,
    ROUND(AVG(petal_length), 2) as avg_petal_length
FROM iris_data
GROUP BY species;

-- Sample query 2: Get recent predictions
SELECT 
    predicted_species,
    ROUND(confidence * 100, 2) as confidence_percent,
    prediction_date
FROM predictions
ORDER BY prediction_date DESC
LIMIT 10;
