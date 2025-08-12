import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle

from utils.constants import Constants


class VectorDbBuilder:
    @staticmethod
    def _find_col(df: pd.DataFrame, candidates):
        cand_lower = [c.lower() for c in candidates]
        for col in df.columns:
            if col.lower() in cand_lower:
                return col
        return None

    @staticmethod
    def build_db():
        # 1) Load data
        df = pd.read_csv(Constants.cleaned_csv_path())

        # 2) Resolve question/answer columns (case-insensitive, with fallbacks)
        q_col = VectorDbBuilder._find_col(df, ["question", "questions", "title", "q"])
        a_col = VectorDbBuilder._find_col(df, ["answer", "answers", "response", "reply", "a", "text", "content"])

        if q_col is None or a_col is None:
            raise ValueError(
                f"Could not find question/answer columns. "
                f"Available columns: {list(df.columns)}. "
                f"Expected something like Question/Answer."
            )

        # 3) Clean/normalize
        df[q_col] = df[q_col].fillna("").astype(str).str.strip()
        df[a_col] = df[a_col].fillna("").astype(str).str.strip()

        # 4) Documents for embedding
        documents = (df[q_col] + " " + df[a_col]).tolist()

        # 5) Model + embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(documents, show_progress_bar=True)

        # 6) FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # 7) Ensure output dir
        out_dir = "data/generated"
        os.makedirs(out_dir, exist_ok=True)

        # 8) Save index
        faiss.write_index(index, os.path.join(out_dir, "faq_index.faiss"))

        # 9) Save metadata WITH a guaranteed `question` key
        #    Keep original row data, but also add normalized keys `question`, `answer`
        records = []
        for i, row in df.iterrows():
            base = row.to_dict()
            base["question"] = str(row[q_col]).strip()
            base["answer"] = str(row[a_col]).strip()
            base["row_index"] = int(i)
            records.append(base)

        with open(os.path.join(out_dir, "faq_metadata.pkl"), "wb") as f:
            pickle.dump(records, f)

        print(
            f"FAQ Knowledge Base created and saved.\n"
            f"- Items: {len(records)}\n"
            f"- Question column: {q_col}\n"
            f"- Answer column: {a_col}\n"
            f"- Output dir: {os.path.abspath(out_dir)}"
        )
