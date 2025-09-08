import pandas as pd
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
from sacrebleu import corpus_bleu, corpus_chrf
import torch

# =============================
# Load Data
# =============================
df = pd.read_csv("Eri_Intern/DataSet/data_preprocessed.csv")
df = df.dropna(subset=["en_clean", "ar_clean"])
df = df.sample(frac=1).reset_index(drop=True)

split = int(0.9 * len(df))
train_df = df[:split]
test_df = df[split:]

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df),
})
print("Train size:", len(dataset["train"]))
print("Test size:", len(dataset["test"]))

# =============================
# Load Model & Tokenizer
# =============================
model_name = "marefa-nlp/marefa-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Force model to float32 if GPU not available
if not torch.cuda.is_available():
    model = model.to(torch.float32)

# =============================
# Preprocess Data
# =============================
max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    inputs = tokenizer(
        examples["en_clean"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )
    targets = tokenizer(
        examples["ar_clean"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# =============================
# Data Collator
# =============================
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# =============================
# Training Arguments
# =============================
training_args = Seq2SeqTrainingArguments(
    output_dir="Eri_Intern/Updated_CheckPoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,   #8
    per_device_eval_batch_size=16,    #8
    gradient_accumulation_steps=2,    #4
    weight_decay=0.01,
    num_train_epochs=30,      #15
    warmup_steps=500,
    lr_scheduler_type="linear",
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=50,         #100
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    label_smoothing_factor=0.2,  #0.1
    # generation_max_length=128,
    # generation_num_beams=6,
    fp16=torch.cuda.is_available(),
)

# =============================
# Metrics (BLEU)
# =============================
# sacrebleu = evaluate.load("sacrebleu")
#
# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#
#     # shift labels: -100 means ignore in loss, replace before decoding
#     labels = [[l if l != -100 else tokenizer.pad_token_id for l in label] for label in labels]
#
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#     decoded_labels = [[lbl] for lbl in decoded_labels]  # sacrebleu expects list of list
#
#     bleu_result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
#     return {"bleu": bleu_result["score"]}
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [[lbl] for lbl in decoded_labels]

    bleu = corpus_bleu(decoded_preds, decoded_labels)
    chrf = corpus_chrf(decoded_preds, decoded_labels)

    return {
        "bleu": bleu.score,
        "chrf": chrf.score
    }

# =============================
# Trainer
# =============================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],   #4
)

# =============================
# Train
# =============================
try:
    trainer.train(resume_from_checkpoint=True)  # try to resume if checkpoint exists
except Exception as e:
    print("⚠️ Warning:", e)
    print("No valid checkpoint found. Training from scratch instead.")
    trainer.train()

# =============================
# Save Final Model
# =============================
trainer.save_model("Eri_Intern/finetuned-en-ar")

# =============================
# Test
# =============================
text = "Hello, how are you?"
inputs = tokenizer([text], return_tensors="pt", padding=True).to(model.device)

outputs = model.generate(
    **inputs,
    max_length=128,
    num_beams=10,     #5
    length_penalty=1.2,
    early_stopping=True
)

print("English:", text)
print("Arabic:", tokenizer.decode(outputs[0], skip_special_tokens=True))
