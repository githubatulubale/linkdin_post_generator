from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, TextDataset,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

dataset = load_dataset("linkedin_posts.txt", tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./linkedin_gpt2_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

# Save model
model.save_pretrained("./linkedin_gpt2_model")
tokenizer.save_pretrained("./linkedin_gpt2_model")