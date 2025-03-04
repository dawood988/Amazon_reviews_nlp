# Amazon Product Reviews Analysis

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation & Dependencies](#installation--dependencies)
4. [Running the Project](#running-the-project)
5. [Functionality](#functionality)
6. [Results & Visualizations](#results--visualizations)
7. [Future Enhancements](#future-enhancements)
8. [Author](#author)
9. [License](#license)

## Overview

This project is a web application built using Streamlit that scrapes and analyzes Amazon product reviews. The application extracts user reviews, ratings, and dates from Amazon product pages and performs text analysis, sentiment analysis, and visualization using Natural Language Processing (NLP) techniques.

## Features

- **Web Scraping**: Extracts Amazon product reviews, ratings, and dates.
- **Text Cleaning**: Processes review text by removing punctuation, stopwords, and lemmatizing words.
- **Sentiment Analysis**: Uses VADER (Valence Aware Dictionary and sentiment Reasoner) to determine sentiment scores.
- **Feature Engineering**: Implements TF-IDF vectorization and Doc2Vec representation.
- **Visualization**: Generates word clouds, sentiment distributions, and summary statistics.
- **Streamlit Interface**: Provides an interactive UI for users to scrape and analyze reviews.

## Installation & Dependencies

### Prerequisites
Ensure you have Python installed along with the following libraries:

```sh
pip install streamlit pandas numpy matplotlib seaborn nltk gensim wordcloud requests beautifulsoup4
```

## Running the Project

1. Clone this repository:
   ```sh
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```sh
   cd amazon-reviews-analysis
   ```
3. Run the Streamlit application:
   ```sh
   streamlit run app.py
   ```

## Functionality

- **Scraping Reviews**: Extracts reviews from Amazon product pages.
- **Data Cleaning**: Preprocesses text data for analysis.
- **Sentiment Analysis**: Computes sentiment scores for each review.
- **Feature Engineering**: Generates numerical representations for machine learning.
- **Visualization**: Displays word clouds, positive/negative reviews, and summary statistics.

## Results & Visualizations

- **Word Clouds**: Show common words in reviews.
- **Sentiment Distribution**: Analyzes positive, neutral, and negative sentiments.
- **Most Positive & Negative Reviews**: Highlights reviews with extreme sentiment scores.

## Future Enhancements

- Implement machine learning models for review classification.
- Deploy the application as a web service.
- Support multiple product URLs for batch analysis.

## Author

- **Dawood M D**

## License

This project is open-source and available under the [MIT License](LICENSE).

