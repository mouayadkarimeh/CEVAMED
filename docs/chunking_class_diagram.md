# Klassendiagramm des Chunking-Pakets

Das Diagramm zeigt die statische Struktur der Klassen in `src/chunking`. Das
Hochformat ist für die Darstellung auf einer einzelnen LaTeX-Seite optimiert.

![Klassendiagramm des Chunking-Pakets](chunking_class_diagram.svg)

Die exportierten Bilddateien werden aus der PlantUML-Quelle
`chunking_class_diagram.puml` erzeugt. Der nachfolgende Mermaid-Block dient als
direkt lesbare Dokumentationsquelle in Markdown.

```mermaid
classDiagram
    direction LR

    namespace CEVAMED_chunking {
        class BaseChunker {
            <<abstract>>
            +split_text(text: str) List~str~*
        }

        class TextSplitter {
            <<abstract>>
            -_chunk_size: int
            -_chunk_overlap: int
            -_length_function: Callable
            -_keep_separator: bool
            -_strip_whitespace: bool
            +split_text(text: str) List~str~*
            +from_huggingface_tokenizer(...) TextSplitter$
            #_join_docs(docs, separator) str
            #_merge_splits(splits, separator) List~str~
        }

        class Tokenizer {
            <<dataclass>>
            +chunk_overlap: int
            +tokens_per_chunk: int
            +decode: Callable
            +encode: Callable
        }

        class RecursiveTokenChunker {
            -_separators: List~str~
            -_is_separator_regex: bool
            +split_text(text: str) List~str~
            -_split_text(text, separators) List~str~
        }

        class TransformerTokenChunker {
            +model_name: str
            -_model: SentenceTransformer
            +tokenizer
            +tokens_per_chunk: int
            +split_text(text: str) List~str~
            +count_tokens(text: str) int
        }

        class KamradtSemanticChunker {
            +model_name: str
            +device: str
            +epsilon: float
            -_embedding_model: SentenceTransformer
            -_tokenizer: PreTrainedTokenizer
            +split_text(text: str) List~str~
            +combine_sentences(sentences, buffer_size) List
            +calculate_sentence_embeddings(...) List
            +calculate_cosine_distances(...) tuple
        }

        class ClusterSemanticChunker {
            +length_function: Callable
            +splitter: RecursiveTokenChunker
            +embedding_function
            +max_cluster: int
            +split_text(text: str) List~str~
            -_get_similarity_matrix(...) ndarray
            -_optimal_segmentation(...) List
        }

        class RecursiveSemanticChunker {
            +length_function: Callable
            +splitter: RecursiveTokenChunker
            +embedding_function
            +avg_chunk_size: int
            +split_text(text: str) List~str~
            +combine_sentences(sentences, buffer_size) List
            +calculate_cosine_distances(...) tuple
            -_find_optimal_threshold(...) float
        }
    }

    BaseChunker <|-- TextSplitter : Vererbung
    TextSplitter <|-- RecursiveTokenChunker : Vererbung
    TextSplitter <|-- TransformerTokenChunker : Vererbung
    TextSplitter <|-- KamradtSemanticChunker : Vererbung
    TextSplitter <|-- ClusterSemanticChunker : Vererbung
    TextSplitter <|-- RecursiveSemanticChunker : Vererbung

    ClusterSemanticChunker *-- "1" RecursiveTokenChunker : besitzt splitter
    RecursiveSemanticChunker *-- "1" RecursiveTokenChunker : besitzt splitter
    TransformerTokenChunker ..> Tokenizer : erzeugt je Aufruf
```

## Leseschlüssel

| Darstellung | Bedeutung im Quellcode |
|---|---|
| `<|--` | Vererbung |
| `*--` | Komposition: Die linke Klasse erzeugt und speichert das Objekt |
| `..>` | Nutzung: Das Objekt wird nur temporär verwendet oder indirekt referenziert |

## Architekturhinweis

Alle fünf konkreten Chunker erben vom projekteigenen `TextSplitter`, der das
abstrakte Interface `BaseChunker` implementiert. `ClusterSemanticChunker` und
`RecursiveSemanticChunker` besitzen zusätzlich jeweils einen
`RecursiveTokenChunker` zur Vorsplittung.

Die freien Hilfsfunktionen wie `split_text_on_tokens` und die Plot-Funktionen
sind bewusst nicht dargestellt, da ein Klassendiagramm Klassen und ihre
strukturellen Beziehungen beschreibt.