from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="Social_Network_Ads.csv", encoding="utf-8")

docs = loader.load()

with open("output.txt", "w", encoding="utf-8") as f:
    for document in docs:
        f.write(document.page_content + "\n\n")
        f.write("---\n\n")
        f.write(str(document.metadata) + "\n\n")
