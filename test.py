from openai import OpenAI

client = OpenAI(
    base_url="https://married-ground-ribcage.ngrok-free.dev/", 
    api_key="sk-wmxaNq_l_KvBRxljcAVJsw",    
    default_headers={"ngrok-skip-browser-warning": "true"}
)

print("Sending request to  jhub  llm ...")

response = client.chat.completions.create(
    model="local-rag-agent",
    messages=[
        {"role": "system", "content": "You are a highly capable AI assistant."},
        {"role": "user", "content": "In two sentences, what is keyboard?"}
    ],
    max_tokens=100,
    temperature=0.3
)

print("\nResponse:")
print(response.choices[0].message.content)