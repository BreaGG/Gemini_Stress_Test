# Gemini Stress Test

## Project Overview

The **Gemini Stress Test** project focuses on analyzing genomic sequences and evaluating text prompts using BERT embeddings. It leverages Google Generative AI to generate insights from DNA sequences and assess the coherence and similarity of generated responses against the original prompts.

## Contents

1. **Data Preparation**: Scripts to download genomic data, preprocess it, and compute various metrics.
2. **Model Interaction**: Utilizes a generative AI model for generating responses based on the analyzed genomic data.
3. **Evaluation Metrics**: Computes similarity and keyword coverage metrics using BERT embeddings and presents aggregated scores.

## Requirements

To run this project, you need to have the following packages installed:

- Python 3.x
- `torch`
- `transformers`
- `datasets`
- `google-generativeai`
- `pandas`
- `scikit-learn`
- `numpy`
- `spacy`
- `dotenv`
- `matplotlib`

You can install the required packages using pip:

```bash
pip install torch transformers datasets google-generativeai pandas scikit-learn numpy spacy python-dotenv matplotlib
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/BreaGG/Gemini_Stress_Test
   ```

2. Ensure you have your Google Generative AI API key set up in a `.env` file:

   ```
   GENAI_API_KEY=your_api_key_here
   ```

3. Download and prepare the genomic data. The script will automatically fetch the necessary files from the Ensembl database.

## Usage

### Data Preparation

The `Data_Preparation.py` script handles the downloading of genomic data and calculating similarity scores.

```bash
python Data_Preparation.py
```

### Model Interaction

The `Gemini_Stress_Test.ipynb` script uses the genomic data to generate insights and evaluate responses.


### Metrics Calculation

The scripts will compute the following metrics for each prompt:

- **Similarity**: Calculated using BERT embeddings.
- **Coverage**: Assesses the overlap of keywords between the prompt and the generated response.
- **Aggregate Score**: A weighted combination of similarity and coverage metrics.

### Output

The output DataFrame containing scores will be saved as `scored_df.xlsx` in the `data` directory, and the mean scores will be displayed using matplotlib.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Generative AI
- Hugging Face Transformers
- Ensembl Database for genomic data