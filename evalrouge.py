
'''
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from rouge import Rouge

# Define the model path
model_path = 'C:\\Users\\ommeh\\Downloads\\IR-4\\model_saved'

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Ensure the tokenizer uses the correct pad token
tokenizer.pad_token = tokenizer.eos_token

def generate_summary(text, max_length=512):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded_input = tokenizer.encode_plus(
        text,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    ).to(device)

    output_sequences = model.generate(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        pad_token_id=tokenizer.pad_token_id,
        max_length=513,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return summary

def calculate_rouge_scores(actual_summary, generated_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, actual_summary, avg=True)
    return scores

# User input for review and expected summary
review_text = "The Fender CD-60S Dreadnought Acoustic Guitar is a great instrument for beginners. It has a solid construction, produces a rich sound, and feels comfortable to play. However, some users have reported issues with the tuning stability."
actual_summary = "Good for beginners but has tuning stability issues."

# Generate summary
generated_summary = generate_summary(review_text)
print("Generated Summary:", generated_summary)

# Calculate and display ROUGE scores
rouge_scores = calculate_rouge_scores(actual_summary, generated_summary)
print("ROUGE scores:", rouge_scores)
'''
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, logging
from rouge import Rouge

# Set the logging level to error to suppress the repeated pad token messages
logging.set_verbosity_error()

# Define paths for the model and tokenizer
file_path = 'C:\\Users\\ommeh\\Downloads\\IR-4\\model_saved'
tokenizer_path = 'C:\\Users\\ommeh\\Downloads\\IR-4\\tokenizer_saved'

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained(file_path)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

# Ensure the tokenizer uses the correct pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
df = pd.read_csv('C:\\Users\\ommeh\\Downloads\\IR-4\\archive (1)\\Reviews.csv')
test_data = df[['Text', 'Summary']].sample(frac=0.25, random_state=42)  # 25% for testing

# Function to generate summaries
def generate_summary(text):
    # Ensure the text is not empty
    if not text:
        return ""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move to same device as model
    generated_outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=150,  # Specifies how many new tokens are to be generated
        num_beams=5,
        no_repeat_ngram_size=2,  # Prevents repeating n-grams in the text
        early_stopping=True,
        return_dict_in_generate=True
    )
    summary = tokenizer.decode(generated_outputs.sequences[0], skip_special_tokens=True)
    return summary

# Compute ROUGE scores
rouge = Rouge()
results = {"rouge-1": {"precision": [], "recall": [], "f1": []},
           "rouge-2": {"precision": [], "recall": [], "f1": []},
           "rouge-l": {"precision": [], "recall": [], "f1": []}}

for index, row in test_data.iterrows():
    review_text = row['Text']
    actual_summary = row['Summary']
    generated_summary = generate_summary(review_text)
    score = rouge.get_scores(generated_summary, actual_summary)[0]
    for key in results:
        results[key]['precision'].append(score[key]['p'])
        results[key]['recall'].append(score[key]['r'])
        results[key]['f1'].append(score[key]['f'])

# Average the results
for key in results:
    for metric in results[key]:
        results[key][metric] = sum(results[key][metric]) / len(results[key][metric])

print("Average ROUGE scores:")
print(results)
