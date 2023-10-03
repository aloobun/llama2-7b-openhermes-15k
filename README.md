## llama2-7b-openhermes-15k-mini

It is a 4-bit qlora refinement of llama-v2-guanaco, fine tuned on the 15k rows of Hermes dataset.

## Usage:

```
from transformers import AutoTokenizer
import transformers
import torch

model = "aloobun/llama2-7b-openhermes-15k-mini"
prompt = "What are large language models?"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    f'[INST] {prompt} [/INST]',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
```

### Output:

```
Result: [INST] What are large language models? [/INST] Large language models are artificial intelligence systems that can be trained on vast amounts of text to generate human-like language. Libraries of natural language processing (NLP) algorithms like BERT and GPT have allowed these systems to learn and improve their capacity for language understanding and generation. These language models have found applications in natural language translation, text summarization, chatbots, and even creative writing. They can help in tasks like predicting the next word in a sentence or even generating a whole text based on a given topic or prompt. Large language models have the potential to revolutionize many industries, from customer support to content creation and beyond. However, their use and development raise important ethical and societal questions, such as the impact on employment or the potential misuse of generated content. As AI technology continues to advance, the role and capabilities of large language models will continue to evolve.
```
