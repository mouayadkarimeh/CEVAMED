# This script is adapted from the LangChain package, developed by LangChain AI.
# Original code can be found at: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/base.py


from transformers import AutoTokenizer
from langchain_text_splitters import (
    CharacterTextSplitter,
   
)
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Any




def results_plot(chunks: list[str], chunk_size: int, length_fn: Any) -> None:
    """Plottet die exakte Länge der Chunks. 

    Der Index startet bei 1 und die Achsen treffen sich exakt im Ursprung.
    """
    if not chunks:
        print("Keine Chunks zum Plotten.")
        return

    chunk_lengths = [length_fn(chunk) for chunk in chunks]
    
    # --- ÄNDERUNG 1: INDEX STARTET NUN BEI 1 STATT 0 ---
    x_positions = list(range(1, len(chunk_lengths) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(x_positions, chunk_lengths, marker="o", linestyle="-", label="Tatsächliche Chunk-Größe (tokens)")
    plt.axhline(y=chunk_size, color="r", linestyle="--", label=f"Ziel-Chunk-Größe ({chunk_size} tokens)")
    
    # --- ÄNDERUNG 2: AXIS-LIMITS & URSPRUNG ERZWINGEN ---
    ax = plt.gca()
    
    # Setzt die Grenzen der X-Achse exakt von 1 bis zum letzten Chunk (ohne Abstand links/rechts)
    ax.set_xlim(left=1, right=len(chunk_lengths))
    
    # Setzt die Y-Achse exakt bei 0 an und gibt nach oben 10% Puffer über dem Ziel-Limit
    ax.set_ylim(bottom=0, top=max(max(chunk_lengths), chunk_size) * 1.1)
    
    # Deaktiviert das automatische Matplotlib-Padding für den Ursprung
    ax.use_sticky_edges = True

    # Erzwingt ganze Zahlen auf der X-Achse
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.title("Charachter text splitter" , fontsize=22 , pad=22)
    plt.suptitle("Analysierte Datei: Albers.txt", fontsize=13, x=0.5, y=0.92, color="green")
    plt.xlabel("Chunk-Index")
    plt.ylabel("Chunk-Größe (Zeichen)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def count_hf_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
def count_characters(text: str) -> int:
    return len(text)

# Platzhalter für deine Plot-Funktion (falls nicht woanders importiert)

if __name__ == "__main__":

    # 2. Definiere das Modell und lade den Tokenizer ZUERST
    
    model_name = "deutsche-telekom/gbert-large-paraphrase-cosine"
     
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Text einlesen
    with open("data/row/Albers.txt", "r", encoding="utf-8") as f:
        text = f.read()
       
    
   

    charachter_text_splitter = CharacterTextSplitter(
        #separator="",  # Sicherheits-Separatoren
        chunk_size=256, 
        chunk_overlap=64,
        #length_function=len,
        #is_separator_regex=False
    )

    charachter_text_splitter_chunks = charachter_text_splitter.split_text(text)
    
    # 7. Evaluation/Kontrolle der Ergebnisse
    gesamt_char_anzahl = 0
    
    for i, chunk in enumerate(charachter_text_splitter_chunks):
        real_char = count_hf_tokens(chunk)
        print(f"Chunk {i+1}: '{chunk}'")
        print(f"  -> Gemessene zeichen number per chunk : {real_char}")
        gesamt_char_anzahl = gesamt_char_anzahl + real_char 
    print(f"gesamte Zeichen number: {gesamt_char_anzahl}")
    # 8. Plot aufrufen
   
    results_plot(charachter_text_splitter_chunks, charachter_text_splitter._chunk_size, length_fn=len)
    