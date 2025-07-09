from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader


def load_text(file, save_path: str):
    with open(save_path, "wb") as f:
        f.write(file.read())
    if save_path.endswith(".pdf"):
        loader = PyMuPDFLoader(save_path)
    else:
        loader = UnstructuredFileLoader(save_path)
    return loader.load()
