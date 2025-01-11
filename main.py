from typing import List, Dict
from InquirerPy import prompt
import os
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

def gat_path_to_all_files_in_directory(directory_data: Dict):
    """
    Given a dictionary representing data about a given directory,
    it returns the path to all the relevant files in that directory
    """
    path_to_all_files_in_directory = []
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
            path_to_all_files_in_directory.append(file_path)

    return path_to_all_files_in_directory


def vectorize_file_contents_and_store(file_path, collection):
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

    file_content_chunks = text_splitter.split_text(file_content)
    file_chunk_embeddings = []
    for chunk in file_content_chunks:
        chunk_embedding = {
            "id": uuid.uuid4(),
            "document": chunk,
            "metadata": {"file_path": file_path},
            "embedding": get_embedding(chunk)
        }
        file_chunk_embeddings.append(chunk_embedding)

    return file_chunk_embeddings



def get_embedding(chunk: str):
    pass
def chat_with_bot():
    """"""
    pass
all_relevant_document_paths: List[Dict] = []

get_path_to_directory_containing_relevant_documents(all_relevant_document_paths)
print(all_relevant_document_paths)
