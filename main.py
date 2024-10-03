from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Phi-3-mini-4k-instruct',
    device_map='auto', # for GPU environments this value should be set to 'cuda'
    torch_dtype='auto',
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')

generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

messages = [
    {'role': 'user', 'content':'Write a funny joke about space travel.'}
]

output = generator(messages)
print(output[0]['generated_text'])

# expected output should be something like this: "Why don't astronauts get lost in space? Because they always have their "GPS" - the Great Galactic Positioning System!"