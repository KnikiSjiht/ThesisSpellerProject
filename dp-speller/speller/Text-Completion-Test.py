from transformers import pipeline

generator = pipeline('text-generation', model='HuggingFaceH4/zephyr-7b-beta')
generated_text = generator("Hi, I am", max_length=30, num_return_sequences=1)
print(generated_text)