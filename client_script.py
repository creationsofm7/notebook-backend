import requests
import os

API_URL = "http://localhost:8000"

def upload_pdf(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    with open(file_path, "rb") as f:
        files = {"files": (os.path.basename(file_path), f, "application/pdf")}
        response = requests.post(f"{API_URL}/upload/", files=files)
    if response.status_code == 200:
        result = response.json()
        doc_id = result[0]["document_id"]
        print(f"PDF uploaded. Document ID: {doc_id}")
        return doc_id
    else:
        print(f"Error uploading PDF: {response.text}")
        return None

def upload_multiple_pdfs(file_paths):
    files = []
    file_handles = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        f = open(path, "rb")
        file_handles.append(f)
        files.append(("files", (os.path.basename(path), f, "application/pdf")))
    if not files:
        print("No valid files to upload.")
        return None
    response = requests.post(f"{API_URL}/upload/", files=files)
    for f in file_handles:
        f.close()
    if response.status_code == 200:
        results = response.json()   
        for result in results:
            print(f"PDF uploaded. Document ID: {result['document_id']} (message: {result['message']})")
        return [r["document_id"] for r in results]
    else:
        print(f"Error uploading PDFs: {response.text}")
        return None

def list_documents():
    response = requests.get(f"{API_URL}/documents/")
    if response.status_code == 200:
        docs = response.json()
        if not docs:
            print("No documents found.")
        else:
            print("\nAvailable Documents:")
            for doc in docs:
                print(f"- ID: {doc['document_id']} | File: {doc['filename']}")
        return docs
    else:
        print(f"Error listing documents: {response.text}")
        return None

def query_document(document_id, query):
    data = {"document_id": document_id, "query": query}
    response = requests.post(f"{API_URL}/query/", json=data)
    if response.status_code == 200:
        result = response.json()
        print("\nAnswer:\n" + result["answer"])
        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")
        return result
    else:
        print(f"Error querying document: {response.text}")
        return None

def query_multiple_documents(document_ids, query):
    # Ensure document_ids is always a list of strings
    if not isinstance(document_ids, list):
        document_ids = [document_ids]
    data = {"document_ids": document_ids, "query": query}
    response = requests.post(f"{API_URL}/query/", json=data)
    if response.status_code == 200:
        result = response.json()
        print("\nAnswer:\n" + result["answer"])
        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")
        return result
    else:
        print(f"Error querying documents: {response.text}")
        return None

def chat_with_document(document_id):
    print(f"\nChatting with document {document_id}. Type 'exit' to quit.")
    chat_history = []
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() == "exit":
            break
        data = {"document_id": document_id, "query": user_input}
        response = requests.post(f"{API_URL}/query/", json=data)
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "")
            print(f"Bot: {answer}")
            chat_history.append([user_input, answer])
        else:
            print(f"Error: {response.text}")
            break

def chat_with_multiple_documents(document_ids):
    print(f"\nChatting with documents {', '.join(document_ids)}. Type 'exit' to quit.")
    chat_history = []
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() == "exit":
            break
        data = {"document_ids": document_ids, "query": user_input}
        response = requests.post(f"{API_URL}/query/", json=data)
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "")
            print(f"Bot: {answer}")
            chat_history.append([user_input, answer])
        else:
            print(f"Error: {response.text}")
            break

def main():
    print("PDF Q&A Client")
    while True:
        print("\n1. Upload PDF\n2. Upload Multiple PDFs\n3. List Documents\n4. Query Document\n5. Chat with Document\n6. Query Multiple Documents\n7. Chat with Multiple Documents\n8. Exit")
        choice = input("Choose (1-8): ")
        if choice == "1":
            path = input("PDF file path: ")
            upload_pdf(path)
        elif choice == "2":
            paths = input("PDF file paths (comma separated): ").split(",")
            paths = [p.strip() for p in paths if p.strip()]
            upload_multiple_pdfs(paths)
        elif choice == "3":
            list_documents()
        elif choice == "4":
            doc_id = input("Document ID: ")
            query = input("Your query: ")
            query_document(doc_id, query)
        elif choice == "5":
            doc_id = input("Document ID: ")
            chat_with_document(doc_id)
        elif choice == "6":
            doc_ids = input("Document IDs (comma separated): ").split(",")
            doc_ids = [d.strip() for d in doc_ids if d.strip()]
            query = input("Your query: ")
            query_multiple_documents(doc_ids, query)
        elif choice == "7":
            doc_ids = input("Document IDs (comma separated): ").split(",")
            doc_ids = [d.strip() for d in doc_ids if d.strip()]
            chat_with_multiple_documents(doc_ids)
        elif choice == "8":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
