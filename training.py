import torch
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
import json
from PIL import Image 

class ConversationDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            # Concatenate all conversation pieces into a single string
            conversation_text = ' '.join([conv['value'] for conv in example['conversations']])
            texts.append(conversation_text)
            images.append(Image.open(example['image'][1:]))

        # Process texts and images with the provided processor
        batch = self.processor(texts, images, return_tensors="pt", padding=True, truncation=True)

        # Prepare labels for model training
        labels = batch['input_ids'].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100  # Mask pad tokens in loss calculation
        batch['labels'] = labels

        return batch


model_id = "llava-hf/llava-1.5-7b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)
model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                      quantization_config=quantization_config,
                                                      torch_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer = tokenizer

data_collator = ConversationDataCollator(processor)
 
# Opening JSON file
train_file = open('train_data.json')
test_file = open('test_data.json')
# returns JSON object as 
# a dictionary
train_data = json.load(train_file)
test_data = json.load(test_file)
print(len(train_data))
print(len(test_data))


training_args = TrainingArguments(
    output_dir="out_dir",
    learning_rate=1.4e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    logging_steps=100,
    num_train_epochs=10,
    push_to_hub=False,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    fp16=True,
    bf16=False
)
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules="all-linear" 
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    peft_config=lora_config,
    dataset_text_field='conversations',  # need a dummy field
    tokenizer=tokenizer,
    data_collator=data_collator,
    max_seq_length = 100,
    dataset_kwargs={"skip_prepare_dataset": True},
)

trainer.train()
model.save_pretrained("./out_dir")


