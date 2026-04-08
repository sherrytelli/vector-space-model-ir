# Vector Space Model (VSM) Search Engine

This project implements a Information Retrieval System based on a Vector Space Model (VSM). It allows users to search through a local corpus of text documents using natural language queries. The model calculates TF-IDF (Term Frequency-Inverse Document Frequency) scores and uses Cosine Similarity to find and rank the most relevant documents. A graphical user interface (GUI) is provided using Python's Tkinter library.

## Features
* **Text Preprocessing:** Tokenization, stopword removal, and lemmatization utilizing NLTK.
* **TF-IDF Weighting:** Computes the statistical importance of words across the document corpus.
* **Cosine Similarity Ranking:** Ranks and filters documents based on an alpha threshold (0.05).
* **Performance Optimization:** Caches the inverted index, document frequencies, and document vectors using `pickle` to significantly reduce load times on subsequent runs.
* **Desktop GUI:** A simple and intuitive interface to input queries and view results.

## Project Structure
To run the application successfully, ensure your repository follows this exact structure:

```text
Project_Root/
│
├── Abstracts/               # Directory containing your corpus of .txt files (e.g., 1.txt, 2.txt)
├── Stopword-List.txt        # Text file containing stopwords to be filtered out
├── main.py                  # The Tkinter GUI application and entry point
└── model.py                 # Core Vector Space Model logic and classes
```
*Note: Upon the first successful execution, the application will automatically generate `.pkl` files (e.g., `inverted_index.pkl`, `doc_vectors.pkl`) in the root directory to cache the model's state.*

## Installation & Setup

It is highly recommended to run this project inside a Python Virtual Environment (`venv`) to keep the dependencies isolated from your system's global Python installation.

### 1. Clone the repository
```bash
git clone https://github.com/sherrytelli/vector-space-model-ir.git
cd vector-space-model-ir
```

### 2. Set up the Virtual Environment
Create and activate the virtual environment based on your operating system:

**For Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**For macOS and Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
*(You will know it is activated when you see `(venv)` appear at the start of your terminal prompt.)*

### 3. Install Dependencies
With the virtual environment activated, install the required third-party packages:
```bash
pip install nltk natsort
```

### 4. Download NLTK Data
The model relies on NLTK for natural language processing, which requires specific datasets to function (specifically `punkt` for tokenization and `wordnet` for lemmatization). 

Run the following commands in your terminal to download them:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

## Usage

1. Ensure your virtual environment is currently active.
2. Launch the application by running:
   ```bash
   python main.py
   ```
3. The GUI window will open. Type your search query into the input box and click **Enter** (or use the designated button) to retrieve relevant document numbers.
4. Type `!help` in the search bar to clear the screen and reset the view.

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.