from typing import List, Dict
from InquirerPy import prompt
import os
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.types import Collection
import chromadb
from pypdf import PdfReader
import openai
import sys


def get_path_to_directory_containing_relevant_documents(relevant_directory_paths: List[Dict]):
    """
    Prompts the user to get information about the directory containing
    relevant documents that should be used for retrieval. This includes
    the path to the directory as well as the file type of the documents
    in the directory. This information is added to the relevant_directory_paths
    list in the form of a dictionary
    """
    directory_path = {
        "name": "path_string",
        "type": "input",
        "message": "Please enter a path that contains relevant documents.",
    }

    directory_documents_type = {
        "name": "path_docs_type",
        "type": "list",
        "choices": ["pdf", "txt"],
        "message": "Please enter the file format of the relevant documents in this path",
    }

    questions = [directory_path, directory_documents_type]
    user_response = prompt(questions)

    relevant_directory_paths.append({"path_string": user_response["path_string"],
                                     "path_docs_type": user_response["path_docs_type"]
                                     })


def menu_to_ask_for_new_path(relevant_directory_paths: List[Dict], ai_client: openai.OpenAI, collection: Collection):
    """
    Prompts the user to select if they would like to keep adding paths to
    relevant documents or not. If not, they will continue to the chatbot
    with the documents they already have.
    """
    ask_for_new_path = {
        "name": "add_new_path",
        "type": "list",
        "choices": ["Yes", "No"],
        "message": "Do you wish to add a new directory that contains relevant documents?",
    }
    while True:
        questions = [ask_for_new_path]
        user_response = prompt(questions)
        if user_response["add_new_path"] == "Yes":
            get_path_to_directory_containing_relevant_documents(relevant_directory_paths)
        else:
            populate_database_with_all_relevant_files(relevant_directory_paths, collection)
            chat_with_bot(ai_client, collection)
            break


def get_paths_to_all_files_in_directory(directory_data: Dict):
    """
    Given a dictionary representing data about a given directory,
    it returns the path to all the relevant files in that directory
    """
    paths_to_all_files_in_directory = []
    file_extension: str = ""
    match directory_data["path_docs_type"]:
        case "txt":
            file_extension = ".txt"
        case "pdf":
            file_extension = ".pdf"

    for file_name in os.listdir(directory_data.get("path_str")):
        if file_name.endswith(file_extension):

            file_path = os.path.join(directory_data.get("path_string"), file_name)
            paths_to_all_files_in_directory.append(file_path)

    return paths_to_all_files_in_directory


def populate_database_with_all_relevant_files(relevant_directory_paths: List[Dict], collection: Collection):
    """
    For each directory in relevant_directory_paths, it extracts the text from all the
    relevant files, splits them into chunks, embeds each chunk into a vector space and stores
    all embeddings in the given vector database.
    """
    print("\n-----Saving Relevant Files in the Database-----")
    for directory_data in relevant_directory_paths:
        all_directory_files = get_paths_to_all_files_in_directory(directory_data)
        for file_path in all_directory_files:
            vectorize_file_contents_and_store(file_path, collection)

    print("-----Finished Saving Relevant Files in the Database-----\n")


def vectorize_file_contents_and_store(file_path: str, collection: Collection):
    """
    Takes in the path to a file, extracts all the text from the file,
    divides the file into chunks, embeds those chunks in a vector space and saves them to a
    vector database
    """
    text_splitter = RecursiveCharacterTextSplitter(
        # Increase this post development
        chunk_size=1000,
        chunk_overlap=0,
        is_separator_regex=False,
        length_function=len,
    )
    file_content = ""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()

    elif file_path.endswith(".pdf"):
        pdf = PdfReader(file_path)
        file_content = "".join([page.extract_text(extraction_mode="layout",
                                                  layout_mode_space_vertically=False) for page in pdf.pages])
    file_content_chunks = text_splitter.split_text(file_content)
    for chunk in file_content_chunks:
        collection.add(
            ids = str(uuid.uuid4()),
            documents = chunk,
            metadatas={"file_path": file_path},
            embeddings=get_embedding(chunk),
        )

def get_embedding(chunk: str):
    """
    Takes in a chunk of text and embeds it into a vector space.
    """
    ai_client = openai.OpenAI(api_key=sys.argv[1])
    api_response = ai_client.embeddings.create(
        model="text-embedding-3-large",
        input=chunk,
    )
    return api_response.data[0].embedding

def chat_with_bot(ai_client: openai.OpenAI, collection: Collection):
    """
    Handles chat with bot. Chat is terminated by entering '$$$'
    """
    system_prompt = "You are a helpful AI assistant"
    all_messages = [{"role": "system", "content": system_prompt}]
    print_bot_message("All of the relevant documents have been saved and accessible to me. You can type in '$$$'"
                  "to end this chat session. How may I help you?")
    while True:
        user_prompt = input("")
        if user_prompt == "$$$":
            break
        else:
            relevant_chunks = collection.query(query_embeddings=get_embedding(user_prompt), n_results=2)
            extra_context = (f"Additional Context: {"\n\n".join(relevant_chunks.get("documents")[0])}"
                             f"\nEnd of context.\n")
            all_messages.append({"role": "user", "content": extra_context + user_prompt})
            api_response = ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages = all_messages,
            )
            all_messages.append({"role": "system", "content": api_response.choices[0].message.content})
            print_bot_message(api_response.choices[0].message.content)

def print_bot_message(message: str):
    """
    Prints messages from chatbot. Used for formatting only.
    """
    print(f"\nAssistant: ")
    print(f"{message}\n")
    print("\nUser:")

def main():
    db = chromadb.Client()
    collection = db.get_or_create_collection("relevant_documents")
    ai_client = openai.OpenAI(api_key=sys.argv[1])
    relevant_directory_paths = []
    print("\n\n\t\t----Welcome to RAG (Retrieval Augmented Generation) Chat Bot----\n\n")
    menu_to_ask_for_new_path(relevant_directory_paths, ai_client, collection)
    print("-------------------------------------")
    print("-------------------------------------")
    print("The Rag Chat Bot has been terminated")
    print("-------------------------------------")
    print("-------------------------------------")

if __name__ == "__main__":
    main()
