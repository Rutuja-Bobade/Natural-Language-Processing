
 # SemanticPlag
 
 **SemanticPlag** is a compact Python pipeline that estimates **textual similarity** between two documents using a **hybrid** of lexical overlap (TF‑IDF + cosine similarity) and semantic similarity (Sentence‑BERT). It is intended for **exploratory analysis** and education—not as a legal or institutional plagiarism verdict.
 
 Optional **Google ADK** integration runs the same logic as a deterministic **SequentialAgent** workflow (no LLM required for scoring).
 
 ---
 
 ## Capabilities
 
 | Feature | Description |
 |--------|-------------|
 | **Dual-signal similarity** | Combines bag-of-words style similarity (TF‑IDF) with embedding similarity (Sentence‑BERT). |
 | **Hybrid score** | `0.4 × TF‑IDF + 0.6 × SBERT` at document level (constants in `similarity.py`). |
 | **Sentence-level highlights** | Top‑K sentence pairs (A vs B) ranked by the same hybrid formula. |
 | **Preprocessing** | Lowercasing, tokenization, English stopword removal; optional NLTK WordNet lemmatization when corpora are available. |
 | **Section weighting (optional)** | If both documents contain recognized headings (e.g. Introduction, Methodology, Results, Conclusion), a section-weighted hybrid can inform the reported percentage. |
 | **ADK orchestration** | Same analysis via `SequentialAgent` + `Runner`, with session `state_delta` for persisted results (`--adk`). |
 
 ---
 
 ## Architecture
 
 ```mermaid
 flowchart TB
     subgraph inputs["Inputs"]
         A["Document A\n(plain text or .txt)"]
         B["Document B\n(plain text or .txt)"]
     end
 
     subgraph prep["Preprocessing — preprocessing.py"]
         P1["Lowercase, tokenize"]
         P2["Stopwords — sklearn"]
         P3["Optional lemmatization — NLTK WordNet"]
     end
 
     subgraph lex["Lexical — tfidf_module.py"]
         T["TfidfVectorizer\n+ cosine similarity"]
     end
 
     subgraph sem["Semantic — bert_module.py"]
         E["Sentence-BERT\nall-MiniLM-L6-v2"]
         C["Cosine on embeddings"]
     end
 
     subgraph fuse["Fusion — similarity.py"]
         H["Hybrid: 0.4·TF-IDF + 0.6·SBERT"]
         S["Optional section-weighted hybrid"]
         U["Sentence pairs: cross-product\n+ top-K by hybrid"]
     end
 
     subgraph out["Output"]
         R["PlagiarismReport\nscores + top pairs + %"]
     end
 
     A --> prep
     B --> prep
     prep --> T
     prep --> E
     T --> H
     C --> H
     H --> S
     H --> U
     S --> R
     U --> R
 ```
 
 **ADK path (optional):** `SequentialAgent` runs `PreprocessAgent → LexicalAgent → SemanticAgent → CombineAgent`, with `EventActions.state_delta` so results persist in the ADK session store.
 
 ---
 
 ## Project structure
 
 ```
 SemanticPlag/
 ├── main.py
 ├── preprocessing.py
 ├── tfidf_module.py
 ├── bert_module.py
 ├── similarity.py
 ├── utils.py
 ├── adk_orchestrator.py
 ├── requirements.txt
 └── README.md
 ```
 
 ---
 
 ## Setup (from GitHub)
 
 **Prerequisites:** Python **3.10+**, `pip`, and disk space for PyTorch and the Sentence‑Transformer model (downloaded on first run).
 
 ```bash
 git clone <YOUR_REPO_URL>
 cd SemanticPlag
 
 python3 -m venv .venv
 source .venv/bin/activate          # Windows: .venv\Scripts\activate
 
 pip install --upgrade pip
 pip install -r requirements.txt
 ```
 
 ### First-run notes
 
 - **Hugging Face:** Model `sentence-transformers/all-MiniLM-L6-v2` is downloaded on first use. Optional: set `HF_TOKEN` for higher Hub rate limits.
 - **NLTK (optional):** WordNet improves lemmatization. If download fails (e.g. SSL), the pipeline still runs without lemmas. Manual: `python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"`.
 
 ---
 
 ## Input format
 
 | Argument | Meaning |
 |----------|---------|
 | `--doc-a` | Document A: path to `.txt` or a **short** plain-text string. |
 | `--doc-b` | Document B: same. |
 | `--top-k` | Number of top sentence pairs (default: `10`). |
 | `--adk` | Use Google ADK `Runner` + `SequentialAgent`. |
 
 Use **`.txt` files for long texts**; extremely long CLI strings can cause path/OS issues.
 
 ---
 
 ## Usage
 
 ```bash
 python main.py --doc-a paper_a.txt --doc-b paper_b.txt --top-k 8
 python main.py --adk --doc-a paper_a.txt --doc-b paper_b.txt
 ```
 
 ---
 
 ## Expected output
 
 Printed report includes:
 
 1. Document-level **TF‑IDF**, **SBERT**, and **hybrid** (≈0–1).
 2. **Estimated similarity / risk** as a **percentage** (hybrid or section-weighted hybrid when applicable).
 3. **Top‑K sentence pairs** with **tfidf**, **bert**, and **hybrid** per pair.
 
 Example (shape only):
 
 ```
 ========== SEMANTIC PLAGIARISM REPORT ==========
 Document TF-IDF (lexical):     0.4300
 Document SBERT (semantic):     0.9100
 Document hybrid (0.4·TF-IDF + 0.6·SBERT): 0.7180
 Estimated similarity / risk: 71.8%
 
 --- Top similar sentence pairs (doc A vs doc B) ---
   [1] hybrid=0.72  tfidf=0.43  bert=0.91
       A: ...
       B: ...
 ==============================================
 ```
 
 ---
 
 ## Limitations
 
 - Similarity **signals**, not legal proof of plagiarism.
 - **English**-oriented stopwords.
 - **Heuristic** sentence splitting.
 - Section weighting only when headings match patterns in `utils.py`.
 
 ---
 
 ## License
 
 Add a license (e.g. MIT) and a `LICENSE` file if you publish the repo.