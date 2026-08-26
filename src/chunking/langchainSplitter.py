# This script is adapted from the LangChain package, developed by LangChain AI.
# Original code can be found at: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/base.py


from transformers import AutoTokenizer
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter ,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from matplotlib import pyplot as plt

def results_plot(chunks: list[str], chunk_size: int) -> None:
    if not chunks:
        print("Keine Chunks zum Plotten.")
        return

    chunk_lengths = [count_characters(chunk) for chunk in chunks]
    x_positions = list(range(len(chunk_lengths)))

    plt.plot(x_positions, chunk_lengths, marker="o")
    plt.axhline(y=chunk_size, color="r", linestyle="-", label="chunk_size")
    plt.title("chunking")
    plt.xlabel("Chunk index")
    plt.ylabel("Chunk length")
    plt.legend()
    plt.show()




class TokenTextSplitterCustom(TokenTextSplitter):
    def __init__(self, encoding_name="gpt2", model_name=None, chunk_size=256, chunk_overlap=64, length_function=None):
        # Wir rufen das Original auf, lassen aber den fehlerhaften Modellnamen weg
        super().__init__(encoding_name=encoding_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Wir merken uns deine hf-Zählfunktion für später
        self.custom_length_function = length_function

# --- AB HIER FOLGT DEIN CODE EINS ZU EINS ---

def count_hf_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
def count_characters(text: str) -> int:
    return len(text)

# Platzhalter für deine Plot-Funktion (falls nicht woanders importiert)

if __name__ == "__main__":

    # 2. Definiere das Modell und lade den Tokenizer ZUERST
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Text einlesen
    with open("data/row/Albers.txt", "r", encoding="utf-8") as f:
        text = f.read()
       
    # 4. Erstelle den Recursive Splitter mit Sicherheits-Separatoren
    recursive_charachter_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,                  
        chunk_overlap=64,               
        length_function=count_hf_tokens,
        #separators=["\n\n", "\n", " ", ""] # Verhindert, dass Chunks zu groß werden
    )

    # 5. Erstelle den Sentence Transformers Splitter (Korrigierte Parameter!)
    sentence_transformer_token_text_splitter = SentenceTransformersTokenTextSplitter(
        model_name=model_name,      # Muss aktiv sein, damit er das richtige Modell lädt
        tokens_per_chunk=256,       # Das ist der korrekte Parameter für die Größe
        chunk_overlap=64            
    )

    # DIESE ZEILEN STÜRZEN JETZT NICHT MEHR AB!
    token_text_splitter = TokenTextSplitterCustom(
        encoding_name="gpt2",  
        model_name=model_name, 
        chunk_size=256, 
        chunk_overlap=64,
        length_function=count_hf_tokens
    )

    charachter_text_splitter = CharacterTextSplitter(
        #separator="\n\n",  # Sicherheits-Separatoren
        chunk_size=256, 
        chunk_overlap=64,
        #length_function=len,
        #is_separator_regex=False
    )

    # 6. Splitten ausführen
    recursive_charachter_text_splitter_chunks = recursive_charachter_text_splitter.split_text(text)
    sentence_transformer_token_text_splitter_chunks = sentence_transformer_token_text_splitter.split_text(text)
    token_text_splitter_chunks = token_text_splitter.split_text(text)
    charachter_text_splitter_chunks = charachter_text_splitter.split_text(text)
    
    # 7. Evaluation/Kontrolle der Ergebnisse
    for i, chunk in enumerate(charachter_text_splitter_chunks):
        real_tokens = count_hf_tokens(chunk)
        print(f"Chunk {i+1}: '{chunk}'")
        print(f"  -> Gemessene Tokens: {real_tokens} (Garantiert <= 256)\n")
 
    # 8. Plot aufrufen
    #results_plot(token_text_splitter_chunks, token_text_splitter._chunk_size)
    #results_plot(recursive_charachter_text_splitter_chunks, recursive_charachter_text_splitter._chunk_size)
    #results_plot(sentence_transformer_token_text_splitter_chunks, sentence_transformer_token_text_splitter.tokens_per_chunk)
    results_plot(charachter_text_splitter_chunks, charachter_text_splitter._chunk_size)
