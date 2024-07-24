import torch
from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline
import re

def read_file(file_path):
    # Read the content of the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    return file_content

def sliding_window(text, window_size=10000, stride=5000):
    # Implement sliding window over the text
    chunks = []
    for i in range(0, len(text), stride):
        chunk = text[i:i + window_size]
        chunks.append(chunk)
    return chunks

def chatbot(context, question):
    # Load pre-trained model and tokenizer
    model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
    tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

    # Initialize variables to store answers
    answers = []

    # Process each chunk using sliding window
    for chunk in sliding_window(context):
        # Encode the question and chunk
        inputs = tokenizer(question, chunk, return_tensors='pt', truncation=True, padding=True)

        # Use the model to get the answer
        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

        # Append the answer to the list
        answers.append(answer)

    # Combine the answers from all chunks
    full_answer = " ".join(answers)

    # Remove [CLS] tokens and extra spaces
    full_answer = re.sub(r'\[CLS\]', '', full_answer).strip()

    return full_answer

# Example usage:
file_path = r'C:\Users\ssist\Desktop\MB1Text.txt'
context = read_file(file_path)

# User input
user_question = input("Ask a question: ")

# Get the answer from the chatbot function
answer = chatbot(context, user_question)

# Print the answer
print("Answer:", answer)