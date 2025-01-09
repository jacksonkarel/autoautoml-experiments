import os
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer


def llama_algo_gen():
    client = InferenceClient(
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        token=os.environ["HUGGING_FACE_HUB_TOKEN"],
    )

    for message in client.chat_completion(
        messages=[{"role": "user", "content": "Create an artificial neural network architecture for text generation that no one has thought of yet that attempts to be very good at generalization"}],
        max_tokens=2000,
        stream=True,
    ):
        output = message.choices[0].delta.content
        print(output)
        
        # # Load the tokenizer for Llama 3.1 (replace 'meta-llama/Llama-3b' with the correct model path if different)
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")

        # # Input text
        # text = "Your input string here."

        # # Tokenize the text
        # tokens = tokenizer.encode(text, add_special_tokens=False)

        # # Count the number of tokens
        # num_tokens = len(tokens)
        # print(num_tokens)
