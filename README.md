# RAG Chat Bot
## _(Retrieval Augmented Generation)_

### Introduction

As AI adoption grows, the demand for chatbots capable of performing specialized tasks with contextual understanding is rising. For instance, a business owner may wish to integrate Large Language Models (LLMs) into their customer service workflows. However, LLMs often lack the contextual knowledge (specific business information) required to deliver meaningful assistance.

Retrieval-Augmented Generation (RAG) addresses this by integrating vectorized retrieval techniques into the LLM pipeline, enriching responses with relevant context. This repository provides a basic implementation of naive RAG using an OpenAI-based chatbot.

The codebase is modular, making it easy to customize. See the **Customizability** section for details on adapting the chatbot to use other models.

---

### Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/kdukuray/rag-chat-bot.git
   ```

2. **Install dependencies**

   ```bash
   pip3 install -r requirements.txt
   ```

3. **Run the script**

   Pass your OpenAI API key as a parameter:

   ```bash
   python3 main.py <openai_api_key>
   ```

   The script will continuously prompt you to specify directories on your computer containing relevant files to use as context. Compatible file formats are:

   - PDF (with selectable text)
   - TXT

   After providing the necessary context, the documents will be saved, and the chat session will begin. The chatbot will then use the contextual information from the specified directories to provide enriched responses.

---

### Customizability

To use a different LLM with this script:

1. Update the initialization of the client in the `main` function to point to your preferred model.
2. Modify the `chat_with_bot` function to handle the chat interactions and parse responses correctly.

For example, if using OpenAI's API, the response from the model is parsed as follows:

```python
api_response = ai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=all_messages,
)
# Extract the actual response
all_messages.append({"role": "system", "content": api_response.choices[0].message.content})
```

Most other LLM providers are compatible with OpenAI's SDK. If your chosen provider supports this, simply specify the `base_url` parameter during client creation:

```python
ai_client = openai.OpenAI(api_key=sys.argv[1], base_url="your_api_provider_url")
```

---

### To-Do List

1. Integrate more advanced RAG techniques.

---

### Contact Information

For questions or feedback, feel free to reach out on [X](https://x.com/kalelodukuray).

