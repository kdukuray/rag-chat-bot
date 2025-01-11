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
     relevant documents that should be used for retrieval. This includes the path to the
     directory as well as the file type of the documents in the directory. This information is added to the
     all_relevant_docs_paths list in the form of a dictionary
    """
    directory_path = {
        "name": "path_string",
        "type": "input",
        "message": "Please enter a path that contains the relevant documents.",
    }

    directory_documents_type = {
        "name": "path_doc_type",
        "type": "list",
        "choices": ["pdf", "txt"],
        "message": "Please enter the file format of the relevant documents in this path",
    }

    questions = [directory_path, directory_documents_type]
    user_response = prompt(questions)

    relevant_directory_paths.append({user_response["path_string"] : user_response["path_doc_type"]})


def menu_to_ask_for_new_path(relevant_directory_paths: List[Dict]):
    """
    Prompts the user to select if they would like to keep adding
    paths to relevant documents or no
    """
    ask_for_new_path = {
        "name": "add_new_path",
        "type": "ist",
        "choices": ["Yes", "No"],
        "message": "Do you wish to add a new directory that contains the relevant documents?",
    }
    questions = [ask_for_new_path]
    user_response = prompt(questions)
    if user_response["add_new_path"] == "Yes":
        get_path_to_directory_containing_relevant_documents(relevant_directory_paths)
        menu_to_ask_for_new_path(relevant_directory_paths)
    else:
        chat_with_bot()

def gat_paths_to_all_files_in_directory(directory_data: Dict):
    """
    Given a dictionary representing data about a given directory,
    it returns the path to all the relevant files in that directory
    """
    paths_to_all_files_in_directory = []
    file_extension: str = ""
    # Determine the file extension for the files in the directory
    match directory_data["directory_documents_type"]:
        case "txt":
            file_extension = ".txt"
        case "pdf":
            file_extension = ".pdf"

    for file_name in os.listdir(directory_data.get("directory_path")):
        if file_name.endswith(file_extension):
            file_path = os.path.join(directory_data.get("directory_path"), file_name)
            paths_to_all_files_in_directory.append(file_path)

    return paths_to_all_files_in_directory


def vectorize_file_contents_and_store(file_path: str, collection: Collection):
    """Takes in the path to a file and returns a list of embeddings,
    where each embed"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
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
            ids = uuid.uuid4(),
            documents = chunk,
            metadatas={"file_path": file_path},
            embeddings=get_embedding(chunk),
        )

def get_embedding(chunk: str):
    openai_client = openai.OpenAI(api_key=sys.argv[1])
    api_response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=chunk,
    )
    return api_response.data[0].embedding

def chat_with_bot(openai_client: openai.OpenAI, collection: Collection):
    system_prompt = "You are a helpful AI assistant"
    all_messages = [{"role": "system", "content": system_prompt}]
    print_message("All of your relevant documents have been saved and are now retrievable. You can type in '$$$'"
                  "to end this chat session. How may I help you?",
                  "Assistant")
    while True:
        user_prompt = input("")
        if user_prompt == "$$$":
            break
        else:
            all_messages.append({"role": "user", "content": user_prompt})
            print_message(user_prompt, "User")
            api_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages = all_messages,
            )
            all_messages.append({"role": "system", "content": api_response.choices[0].message.content})
            print_message(api_response.choices[0].message.content, "Assistant")

def print_message(message: str, sender: str):
    print(f"{sender}: ")
    print(f"\\tt{message}")
    print("\n")




all_relevant_document_paths: List[Dict] = []

get_path_to_directory_containing_relevant_documents(all_relevant_document_paths)
print(all_relevant_document_paths)
