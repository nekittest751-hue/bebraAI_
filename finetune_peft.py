# finetune_peft.py - template for LoRA fine-tuning (edit paths & hyperparams)
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import os

BASE_MODEL = os.environ.get('BASE_MODEL', 'EleutherAI/gpt-neo-1.3B')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, device_map='auto')

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['q_proj','v_proj'],
    lora_dropout=0.1,
    bias='none',
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

dataset = load_dataset('json', data_files={'train':'train.jsonl','validation':'train.jsonl'})
def preprocess(batch):
    texts = [p['prompt'] + '\\n' + p['completion'] for p in batch['train']]
    tok = tokenizer(texts, truncation=True, padding='max_length', max_length=512)
    return {'input_ids': tok['input_ids'], 'attention_mask': tok['attention_mask']}
# adapt to your dataset

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    output_dir='./bebra-lora'
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset['train'])
trainer.train()
trainer.save_model('./bebra-lora-final')
