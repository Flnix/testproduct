import torch
from transformers import XLNetTokenizer, XLNetLMHeadModel
# Load the XLNet tokenizer and model
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')
while True:
    # Define a prompt
    prompt = input("Enter a prompt: ")
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # Generate text
    output = model.generate(input_ids, max_length=50, do_sample=True)
    # Decode the output
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Print the output
    print(output_text)
