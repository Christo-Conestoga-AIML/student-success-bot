import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle

from utils.constants import Constants


class KnowledgeBaseBuilder:
    @staticmethod
    def build_kb():
        df = pd.read_csv(Constants.cleaned_csv_path())

        documents = (df["Question"] + " " + df["Answer"]).tolist()

        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 4. Generate embeddings
        embeddings = model.encode(documents, show_progress_bar=True)

        # 5. Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # 6. Save index and metadata for future use
        faiss.write_index(index, "data/generated/faq_index.faiss")
        with open("data/generated/faq_metadata.pkl", "wb") as f:
            pickle.dump(df.to_dict(orient="records"), f)

        print("FAQ Knowledge Base created and saved.")



