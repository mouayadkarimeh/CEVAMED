# This script is adapted from the LangChain package, developed by LangChain AI.
# Original code can be found at: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/base.py


from abc import ABC, abstractmethod
from enum import Enum
import logging
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)
from base import BaseChunker
import matplotlib.pyplot as plt


from attr import dataclass

logger = logging.getLogger(__name__)

TS = TypeVar("TS", bound="TextSplitter")
class TextSplitter(BaseChunker, ABC):
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator in the chunks
            add_start_index: If `True`, includes chunk's start index in metadata
            strip_whitespace: If `True`, strips whitespace from the start and end of
                              every document
        """
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    # @classmethod
    # def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> TextSplitter:
    #     """Text splitter that uses HuggingFace tokenizer to count length."""
    #     try:
    #         from transformers import PreTrainedTokenizerBase

    #         if not isinstance(tokenizer, PreTrainedTokenizerBase):
    #             raise ValueError(
    #                 "Tokenizer received was not an instance of PreTrainedTokenizerBase"
    #             )

    #         def _huggingface_tokenizer_length(text: str) -> int:
    #             return len(tokenizer.encode(text))

    #     except ImportError:
    #         raise ValueError(
    #             "Could not import transformers python package. "
    #             "Please install it with `pip install transformers`."
    #         )
    #     return cls(length_function=_huggingface_tokenizer_length, **kwargs)

    @classmethod
    def from_tiktoken_encoder(
        cls: Type[TS],
        encoding_name: str = "gpt2",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> TS:
        """Text splitter that uses tiktoken encoder to count length."""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate max_tokens_for_prompt. "
                "Please install it with `pip install tiktoken`."
            )

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)

        def _tiktoken_encoder(text: str) -> int:
            return len(
                enc.encode(
                    text,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )

        if issubclass(cls, FixedTokenChunker):
            extra_kwargs = {
                "encoding_name": encoding_name,
                "model_name": model_name,
                "allowed_special": allowed_special,
                "disallowed_special": disallowed_special,
            }
            kwargs = {**kwargs, **extra_kwargs}

        return cls(length_function=_tiktoken_encoder, **kwargs)
    
class FixedTokenChunker(TextSplitter):
    """Splitting text to tokens using model tokenizer."""

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        chunk_size: int = 128,
        chunk_overlap: int = 28,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for FixedTokenChunker. "
                "Please install it with `pip install tiktoken`."
            )

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def split_text(self, text: str) -> List[str]:
        def _encode(_text: str) -> List[int]:
            return self._tokenizer.encode(
                _text,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=_encode,
        )

        return split_text_on_tokens(text=text, tokenizer=tokenizer)

@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""
    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""
    decode: Callable[[List[int]], str]
    """ Function to decode a list of token ids to a string"""
    encode: Callable[[str], List[int]]
    """ Function to encode a string to a list of token ids"""


def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> List[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: List[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits



def results_plot(chunks: list[str], chunk_size: int) -> None:
    if not chunks:
        print("Keine Chunks zum Plotten.")
        return

    chunk_lengths = [len(chunk) for chunk in chunks]
    x_positions = list(range(len(chunk_lengths)))

    plt.plot(x_positions, chunk_lengths, marker="o")
    plt.axhline(y=chunk_size, color="r", linestyle="-", label="chunk_size")
    plt.title("Recursive Chunk Sizes")
    plt.xlabel("Chunk index")
    plt.ylabel("Chunk length")
    plt.legend()
    plt.show()


if __name__ == "__main__":


    text = (
            """
               Klinik für Innere Medizin und Kardiologie
    Universitätsklinikum Graz
    A-8010 Graz, Auenbruggerplatz
    
    PATIENTENBRIEF / ENTLASSUNGSBERICHT
    
    Patient: Max Mustermann, geb. 14.11.1965
    Station: Kardio-02, Bett 14
    Stationärer Aufenthalt: 02.08.2026 bis 10.08.2026
    
    DIAGNOSEN:
    1. Akutes Koronarsyndrom (ACS): Nicht-ST-Hebungs-Myokardinfarkt (NSTEMI) am 02.08.2026.
    2. Koronare Herzkrankheit (KHK): 2-Gefäß-Erkrankung mit hochgradiger Stenose der RIVA (Ramus interventricularis anterior) und moderater Stenose der RCX (Ramus circumflexus).
    3. Arterielle Hypertonie, Grad II (ICD-10: I10.9).
    4. Hypercholesterinämie (ICD-10: E78.0).
    5. Diabetes mellitus Typ 2, diätetisch und medikamentös eingestellt.
    
    ANAMNESE UND VERLAUF:
    Der 60-jährige Patient wurde am 02.08.2026 über die Notaufnahme wegen akut aufgetretener, retrosternaler Thoraxschmerzen mit Ausstrahlung in den linken Arm und begleitender Dyspnoe aufgenommen. Die Symptomatik bestand seit ca. zwei Stunden vor Erstkontakt. Im präklinischen EKG zeigten sich diskrete T-Inversionen in den Ableitungen V3-V6, jedoch keine signifikanten ST-Hebungen. Die laborchemische Untersuchung ergab ein initial erhöhtes hochsensitives Troponin T (hs-TnT) von 145 ng/l (Norm < 14 ng/l), welches im Verlauf auf 890 ng/l anstieg, vereinbar mit einem NSTEMI.
    
    Am Aufnahmetag erfolgte die dringliche Herzkatheteruntersuchung (Koronarangiographie). Hierbei zeigte sich eine 90%ige, exzentrische und thrombusbehaftete Stenose im mittleren Segment der RIVA. Die übrigen Gefäße wiesen lediglich wanderfüllende Unregelmäßigkeiten auf, mit Ausnahme einer 50%igen Stenose der RCX. Es erfolgte die erfolgreiche perkutane transluminale Koronarangioplastie (PTCA) der RIVA-Läsion mit Implantation eines Drug-Eluting-Stents (DES, Xience 3.5 x 18 mm). Das angiographische Endergebnis zeigte einen ungestörten TIMI-III-Fluss ohne Dissektionszeichen.
    
    Der postinterventionelle Verlauf auf der kardiologischen Überwachungsstation (IMC) gestaltete sich komplikationslos. Es traten keine Rhythmusstörungen, Nachblutungen an der Schleusen-Punktionsstelle (A. femoralis rechts) oder ischämische Rezidive auf. Der Patient war frühzeitig mobilisierbar und ab dem zweiten postinterventionellen Tag beschwerdefrei.
    
    LABORBEFUNDE (Auswahl vom 09.08.2026):
    Hb: 13.8 g/dl, Leukozyten: 8.4 G/l, Thrombozyten: 245 G/l. Kreatinin: 0.95 mg/dl, eGFR: 84 ml/min/1.73m². CRP: 4.2 mg/l. hs-Troponin T prä-Entlassung: 34 ng/l (rückläufig). LDL-Cholesterin: 124 mg/dl, HbA1c: 6.8 %. Potasium: 4.1 mmol/l.
    
    APPARATIVE DIAGNOSTIK:
    Transthorakale Echokardiographie (05.08.2026): Linksventrikuläre Ejektionsfraktion (LVEF) visuell auf ca. 50% leicht reduziert. Geringgradige Hypokinesie der anteroseptalen Wandabschnitte, passend zum Infarktareal. Keine höhergradigen Vitien. Sklerose der Aortenklappe ohne Stenose. Rechter Ventrikel normal groß und gut funktionierend (TAPSE 22 mm). Kein Perikarderguss.
    
    MEDIKAMENTÖSE THERAPIE BEI ENTLASSUNG:
    1. Acetylsalicylsäure (ASS) 100 mg 1-0-0 p.o. (lebenslang)
    2. Ticagrelor 90 mg 1-0-1 p.o. (DAPT für 12 Monate, bis August 2027)
    3. Atorvastatin 80 mg 0-0-1 p.o. (strikte LDL-Zielwert-Einstellung < 55 mg/dl)
    4. Ramipril 2.5 mg 1-0-0 p.o. (Prognoseverbesserung bei LVEF 50%)
    5. Metoprololsuccinat 23.75 mg 1-0-0 p.o.
    6. Pantoprazol 40 mg 1-0-0 p.o. (als Magenschutz unter DAPT)
    7. Metformin 1000 mg 1-0-1 p.o.
    
    WEITERES PROZEDERE UND EMPFEHLUNGEN:
    Wir entlassen den Patienten in gebessertem Allgemeinzustand nach Hause. Eine Fortführung der dualen Plättchenhemmung (DAPT) mit ASS und Ticagrelor ist für die Dauer von 12 Monaten zwingend erforderlich, um eine Stentthrombose zu verhindern. Eine kardiologische Kontrolluntersuchung inklusive Belastungs-EKG und echokardiographischer Verlaufskontrolle wird in 6 bis 8 Wochen beim niedergelassenen Facharzt empfohlen. Die Einleitung einer ambulanten oder stationären kardiologischen Rehabilitation (Phase II) wurde bereits in die Wege geleitet; der Patient hat hierzu seine Zustimmung erteilt. Aufgrund des erhöhten LDL-Wertes ist eine konsequente Statinstherapie notwendig. Eine Kontrolle des Lipidprofils sowie der Leber- und Retentionswerte sollte in 4 Wochen über den Hausarzt erfolgen. Bei erneuter Angina pectoris oder Dyspnoe ist eine sofortige Wiedervorstellung über den Notruf zu veranlassen.
    
    Mit freundlichen Grüßen,
    Dr. med. A. Gruber
    Oberarzt der kardiologischen Station
    
    
            """
    )


    

    fixed_token_chunker  = FixedTokenChunker()
    chunks = fixed_token_chunker.split_text(text)
    print(chunks)
    results_plot(chunks, fixed_token_chunker._chunk_size)