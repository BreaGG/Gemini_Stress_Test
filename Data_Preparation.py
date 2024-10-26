from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import google.generativeai as genai
from datasets import load_dataset
from time import sleep

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bertModel = BertModel.from_pretrained('bert-base-uncased')

# Function to generate BERT embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = bertModel(**tokens)
    return output.last_hidden_state.mean(dim=1).numpy()

# Function to calculate similarity using BERT embeddings
def calculate_similarity_bert(text1, text2):
    embedding1 = get_bert_embedding(text1)
    embedding2 = get_bert_embedding(text2)
    return cosine_similarity(embedding1, embedding2)[0][0]

# Function to calculate keyword coverage
def keyword_coverage(prompt, response):
    prompt_words = prompt.lower().split()
    response_words = response.lower().split()
    
    prompt_word_count = Counter(prompt_words)
    response_word_count = Counter(response_words)
    
    coverage = sum(min(prompt_word_count[word], response_word_count[word]) for word in prompt_word_count) / len(prompt_words)
    return coverage

# Function to aggregate coherence score
def aggregate_coherence_score(similarity_score, coverage_score):
    return 0.7 * similarity_score + 0.3 * coverage_score

# Configure Google Generative AI API
genai.configure(api_key='AIzaSyB6xMXvP02XOB-0NHkWjuki-C1kWyajyVU')

# Create the model configuration
generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_output_tokens": 5000,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Load dataset
forbidden_question_set = load_dataset("TrustAIRLab/forbidden_question_set", split='train')
print(f"Loaded {len(forbidden_question_set['question'])} questions.")

# Convert to pandas DataFrame and initialize metrics
df = forbidden_question_set.to_pandas()
metrics = ['similarity', 'coverage', 'aggregate']

for rating in metrics:
    df[rating] = None

print(df.columns)

# Process each question in the DataFrame
for i, prompt in enumerate(df['question']):
    try:
        response = model.generate_content(prompt)
        documentation = response.text[:50]
        print(documentation)
        category = response._result.candidates[0].safety_ratings
    except:
        continue
    
    # Calculate ratings for each metric
    for rating in metrics:
        if rating == 'similarity':
            df.at[i, rating] = calculate_similarity_bert(prompt, documentation)
        elif rating == 'coverage':
            df.at[i, rating] = keyword_coverage(prompt, documentation)
        elif rating == 'aggregate':
            similarity_score = calculate_similarity_bert(prompt, documentation)
            coverage_score = keyword_coverage(prompt, documentation)
            df.at[i, rating] = aggregate_coherence_score(similarity_score, coverage_score)
    
    sleep(5)  # Sleep to avoid rate limits
    print(type(response._result))

# Final DataFrame with scores
scored_df = df[["q_id", "content_policy_name", "similarity", "coverage", "aggregate"]]
print(scored_df)
file_path = 'data/scored_df.xlsx'
scored_df.to_excel(file_path, index=False)
print(f"Scores saved to {file_path}.")
