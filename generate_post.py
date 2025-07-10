from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

try:
    model = GPT2LMHeadModel.from_pretrained("./linkedin_gpt2_model")
    tokenizer = GPT2Tokenizer.from_pretrained("./linkedin_gpt2_model")
except:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_post(topic):
    prompt = f"Write a professional LinkedIn post about {topic}."
    result = generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
    return result