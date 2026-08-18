"""Semantic chunking based on sentence embeddings and distance breakpoints."""

from __future__ import annotations

import sys
from typing import Dict, List
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from .fixedTokenChunker import TextSplitter
except ImportError:  # pragma: no cover - direct script execution fallback
    from fixedTokenChunker import TextSplitter

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EPSILON = 1e-6


def results_plot(distances: list[float], breakpoint_distance_threshold: float) -> None:
    if not distances:
        print("Keine Distanzen zum Plotten.")
        return

    y_upper_bound = max(0.2, float(max(distances)) * 1.2)
    plt.plot(distances)
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))
    plt.axhline(y=breakpoint_distance_threshold, color="r", linestyle="-")

    num_distances_above_threshold = sum(x > breakpoint_distance_threshold for x in distances)
    plt.text(x=(len(distances) * 0.01), y=y_upper_bound / 50, s=f"{num_distances_above_threshold + 1} Chunks")

    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for i, breakpoint_index in enumerate(indices_above_thresh):
        start_index = 0 if i == 0 else indices_above_thresh[i - 1]
        end_index = breakpoint_index
        plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
        plt.text(
            x=np.average([start_index, end_index]),
            y=breakpoint_distance_threshold + (y_upper_bound / 20),
            s=f"Chunk #{i}",
            horizontalalignment="center",
            rotation="vertical",
        )

    if indices_above_thresh:
        last_breakpoint = indices_above_thresh[-1]
        if last_breakpoint < len(distances):
            plt.axvspan(
                last_breakpoint,
                len(distances),
                facecolor=colors[len(indices_above_thresh) % len(colors)],
                alpha=0.25,
            )
            plt.text(
                x=np.average([last_breakpoint, len(distances)]),
                y=breakpoint_distance_threshold + (y_upper_bound / 20),
                s=f"Chunk #{len(indices_above_thresh)}",
                rotation="vertical",
            )

    plt.title("Chunks Based on Embedding Breakpoints")
    plt.xlabel("Sentence index")
    plt.ylabel("Cosine distance between sequential sentences")
    plt.show()


class SemanticChunker(TextSplitter):
    """Port of the original Kamradt-style semantic chunker using HF MiniLM embeddings."""

    def __init__(
        self,
        avg_chunk_size: int = 400,
        min_chunk_size: int = 50,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        epsilon: float = DEFAULT_EPSILON,
        embedding_batch_size: int = 500,
    ) -> None:
        super().__init__(chunk_size=min_chunk_size, chunk_overlap=0, length_function=len)
        self.avg_chunk_size = avg_chunk_size
        self.min_chunk_size = min_chunk_size
        self.model_name = model_name
        self.device = device
        self.epsilon = epsilon
        self.embedding_batch_size = embedding_batch_size
        self._embedding_model = SentenceTransformer(self.model_name, device=self.device)

        self.splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=0,
            model_name=self.model_name,
            tokens_per_chunk=min_chunk_size,
            model_kwargs={"device": self.device},
        )
        self.tokenizer = self.splitter.tokenizer

    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(self.tokenizer.encode(text))

    @staticmethod
    def combine_sentences(sentences: List[Dict[str, object]], buffer_size: int = 3) -> List[Dict[str, object]]:
        for i in range(len(sentences)):
            combined_sentence = ""

            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += str(sentences[j]["sentence"]) + " "

            combined_sentence += str(sentences[i]["sentence"])

            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += " " + str(sentences[j]["sentence"])

            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self._embedding_model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(embeddings)

    def calculate_sentence_embeddings(self, sentences: List[Dict[str, object]]) -> List[Dict[str, object]]:
        texts = [str(sentence["combined_sentence"]) for sentence in sentences]
        embeddings = self._embed_texts(texts)

        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i].tolist()

        return sentences

    def calculate_cosine_distances(self, sentences: List[Dict[str, object]]):
        distances: list[float] = []
        embedding_matrix = None

        for index in range(0, len(sentences), self.embedding_batch_size):
            batch_sentences = sentences[index : index + self.embedding_batch_size]
            batch_texts = [str(sentence["combined_sentence"]) for sentence in batch_sentences]
            batch_embeddings = self._embed_texts(batch_texts)

            if embedding_matrix is None:
                embedding_matrix = batch_embeddings
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embeddings), axis=0)

        if embedding_matrix is None:
            return distances, sentences

        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embedding_matrix = embedding_matrix / norms

        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

        for i in range(len(sentences) - 1):
            similarity = similarity_matrix[i, i + 1]
            distance = float(1 - similarity)
            distances.append(distance)
            sentences[i]["distance_to_next"] = distance

        return distances, sentences

    def _find_breakpoint_threshold(self, distances: List[float], number_of_cuts: int) -> float:
        lower_limit = 0.0
        upper_limit = 1.0
        distances_np = np.asarray(distances, dtype=float)
        threshold = 0.0

        while upper_limit - lower_limit > self.epsilon:
            threshold = (upper_limit + lower_limit) / 2.0
            num_points_above_threshold = np.sum(distances_np > threshold)

            if num_points_above_threshold > number_of_cuts:
                lower_limit = threshold
            else:
                upper_limit = threshold

        return threshold

    @staticmethod
    def _chunks_from_threshold(sentences: List[Dict[str, object]], distances: List[float], threshold: float) -> List[str]:
        indices_above_thresh = [i for i, distance in enumerate(distances) if distance > threshold]
        start_index = 0
        chunks: List[str] = []

        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join(str(item["sentence"]) for item in group)
            chunks.append(combined_text)
            start_index = index + 1

        if start_index < len(sentences):
            combined_text = " ".join(str(item["sentence"]) for item in sentences[start_index:])
            chunks.append(combined_text)

        return chunks

    def split_text(self, text: str) -> List[str]:
        sentences_strips = self.splitter.split_text(text)
        sentences = [{"sentence": sentence, "index": i} for i, sentence in enumerate(sentences_strips)]

        if not sentences:
            return []

        sentences = self.combine_sentences(sentences, 3)
        distances, sentences = self.calculate_cosine_distances(sentences)

        total_tokens = sum(self._count_tokens(sentence["sentence"]) for sentence in sentences)
        number_of_cuts = total_tokens // self.avg_chunk_size

        threshold = self._find_breakpoint_threshold(distances, number_of_cuts)
        return self._chunks_from_threshold(sentences, distances, threshold)

    def chunk(self, text: str) -> List[str]:
        return self.split_text(text)


KamradtModifiedChunker = SemanticChunker


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

    semantic_chunker = SemanticChunker()
    chunks = semantic_chunker.split_text(text)
    sentences = [{"sentence": sentence, "index": i} for i, sentence in enumerate(semantic_chunker.splitter.split_text(text))]
    sentences = semantic_chunker.combine_sentences(sentences, 3)
    distances, _ = semantic_chunker.calculate_cosine_distances(sentences)
    total_tokens = sum(semantic_chunker._count_tokens(sentence["sentence"]) for sentence in sentences)
    threshold = semantic_chunker._find_breakpoint_threshold(distances, total_tokens // semantic_chunker.avg_chunk_size)

    print("Chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

    results_plot(distances, threshold)