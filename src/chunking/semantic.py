"""Semantic chunking based on sentence embeddings and distance breakpoints."""

from __future__ import annotations

import sys
from typing import Dict, List
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    """Semantic chunker using sentence embeddings and distance breakpoints.
    
    CRITICAL: This implementation extracts EXACT SUBSTRINGS from the original text
    without any transformation (no lowercase, no unicode normalization, no whitespace
    changes). This ensures perfect alignment with ground truth for evaluation.
    """

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
        import tiktoken

        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._allowed_special = set()
        self._disallowed_special = "all"

    def _split_into_token_units(self, text: str) -> List[Dict[str, object]]:
        """Create position-aware token units like the upstream recursive splitter."""
        token_ids = self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )
        if not token_ids:
            return []

        encoded = bytearray()
        boundaries: List[tuple[int, int]] = []
        for token_index, token_id in enumerate(token_ids, start=1):
            encoded.extend(self._tokenizer.decode_single_token_bytes(token_id))
            try:
                char_end = len(encoded.decode("utf-8"))
            except UnicodeDecodeError:
                continue
            boundaries.append((token_index, char_end))

        units: List[Dict[str, object]] = []
        start_token = 0
        start_char = 0
        boundary_index = 0
        while start_token < len(token_ids):
            target_token = min(start_token + self.min_chunk_size, len(token_ids))
            while boundary_index < len(boundaries) and boundaries[boundary_index][0] < target_token:
                boundary_index += 1
            if boundary_index >= len(boundaries):
                end_token, end_char = len(token_ids), len(text)
            else:
                end_token, end_char = boundaries[boundary_index]

            units.append({
                "sentence": text[start_char:end_char],
                "start": start_char,
                "end": end_char,
                "index": len(units),
            })
            start_token = end_token
            start_char = end_char
            boundary_index += 1

        return units

    def _count_tokens(self, text: str) -> int:
        """Count tokens with the same tokenizer used for token-based boundaries."""
        return len(
            self._tokenizer.encode(
                text,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )
        )

    def _token_end_positions(self, text: str) -> List[int]:
        """Return character offsets after each complete UTF-8 token."""
        token_ids = self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )
        encoded = bytearray()
        positions: List[int] = []

        for token_id in token_ids:
            encoded.extend(self._tokenizer.decode_single_token_bytes(token_id))
            try:
                position = len(encoded.decode("utf-8"))
            except UnicodeDecodeError:
                # A token can end in the middle of a multibyte character.
                continue
            positions.append(position)

        if positions and positions[-1] != len(text):
            positions.append(len(text))
        elif not positions and text:
            positions.append(len(text))
        return positions

    def _split_group_by_tokens(
        self,
        text: str,
        start: int,
        end: int,
        token_budget: int,
    ) -> List[str]:
        """Split an oversized semantic group at exact token boundaries."""
        group_text = text[start:end]
        token_end_positions = self._token_end_positions(group_text)
        if not token_end_positions:
            return [group_text] if group_text.strip() else []

        chunks: List[str] = []
        chunk_start = 0
        token_start = 0
        token_count = len(token_end_positions)

        while token_start < token_count:
            token_end = min(token_start + token_budget, token_count)
            char_end = token_end_positions[token_end - 1]
            chunk = group_text[chunk_start:char_end]
            if chunk.strip():
                chunks.append(chunk)
            chunk_start = char_end
            token_start = token_end

        return chunks

    def _append_group(
        self,
        text: str,
        group: List[Dict[str, object]],
        token_budget: int,
        chunks: List[str],
    ) -> None:
        """Append a semantic group, enforcing the token budget exactly."""
        if not group:
            return

        group_start = int(group[0]["start"])
        group_end = int(group[-1]["end"])
        group_text = text[group_start:group_end]

        if self._count_tokens(group_text) <= token_budget:
            if group_text.strip():
                chunks.append(group_text)
            return

        chunks.extend(self._split_group_by_tokens(text, group_start, group_end, token_budget))

    @staticmethod
    def combine_sentences(sentences: List[Dict[str, object]], buffer_size: int = 3) -> List[Dict[str, object]]:
        """Add combined sentence context (with buffer) for embedding calculation."""
        for i in range(len(sentences)):
            combined_sentence = ""

            # Add preceding sentences (buffer)
            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += str(sentences[j]["sentence"]) + " "

            # Add center sentence
            combined_sentence += str(sentences[i]["sentence"])

            # Add following sentences (buffer)
            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += " " + str(sentences[j]["sentence"])

            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts using the model."""
        embeddings = self._embedding_model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(embeddings)

    def calculate_sentence_embeddings(self, sentences: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Calculate embeddings for combined sentences."""
        texts = [str(sentence["combined_sentence"]) for sentence in sentences]
        embeddings = self._embed_texts(texts)

        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i].tolist()

        return sentences

    def calculate_cosine_distances(self, sentences: List[Dict[str, object]]):
        """Calculate cosine distances between consecutive sentence embeddings."""
        distances: list[float] = []
        embedding_matrix = None

        # Process in batches
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

        # Normalize embeddings
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embedding_matrix = embedding_matrix / norms

        # Compute similarity matrix
        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

        # Calculate distances between consecutive sentences
        for i in range(len(sentences) - 1):
            similarity = similarity_matrix[i, i + 1]
            distance = float(1 - similarity)
            distances.append(distance)
            sentences[i]["distance_to_next"] = distance

        return distances, sentences

    def _find_breakpoint_threshold(self, distances: List[float], number_of_cuts: int) -> float:
        """Binary search to find distance threshold that gives desired number of cuts."""
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

    def split_text(self, text: str) -> List[str]:
        """Split text using the upstream Kamradt semantic-chunking flow."""
        # The upstream implementation starts with recursive token-sized units.
        sentences = self._split_into_token_units(text)

        if len(sentences) < 2:
            # Too few sentences to chunk semantically
            return [text] if text.strip() else []

        # Step 2: Add context for embedding calculation
        sentences = self.combine_sentences(sentences, buffer_size=3)

        # Step 3: Calculate embeddings and distances
        distances, sentences = self.calculate_cosine_distances(sentences)

        if not distances:
            # No distances calculated
            return [text] if text.strip() else []

        # Match upstream: number_of_cuts is an approximate target, not a hard limit.
        total_tokens = sum(self._count_tokens(sentence["sentence"]) for sentence in sentences)
        number_of_cuts = total_tokens // self.avg_chunk_size

        threshold = self._find_breakpoint_threshold(distances, number_of_cuts)

        # Step 5: Find breakpoint indices
        indices_above_thresh = [i for i, distance in enumerate(distances) if distance > threshold]

        # Step 6: Group sentences at breakpoints and extract exact substrings
        start_idx = 0
        chunks: List[str] = []

        for break_idx in indices_above_thresh:
            # Group sentences from start_idx to break_idx (inclusive)
            group = sentences[start_idx : break_idx + 1]

            if not group:
                continue

            chunk_start = int(group[0]["start"])
            chunk_end = int(group[-1]["end"])
            chunk = text[chunk_start:chunk_end]
            if chunk.strip():
                chunks.append(chunk)

            start_idx = break_idx + 1

        # Add remaining sentences as final chunk
        if start_idx < len(sentences):
            group = sentences[start_idx:]
            if group:
                chunk_start = int(group[0]["start"])
                chunk_end = int(group[-1]["end"])
                chunk = text[chunk_start:chunk_end]
                if chunk.strip():
                    chunks.append(chunk)

        return [c for c in chunks if c and c.strip()]

    def chunk(self, text: str) -> List[str]:
        """Alias for split_text for compatibility."""
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

Der postinterventionielle Verlauf auf der kardiologischen Überwachungsstation (IMC) gestaltete sich komplikationslos. Es traten keine Rhythmusstörungen, Nachblutungen an der Schleusen-Punktionsstelle (A. femoralis rechts) oder ischämische Rezidive auf. Der Patient war frühzeitig mobilisierbar und ab dem zweiten postinterventionellen Tag beschwerdefrei.

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

    print(f"Semantic Chunker: {len(chunks)} chunks\n")
    
    # Verify all chunks are exact substrings
    for i, chunk in enumerate(chunks, 1):
        in_text = chunk in text
        status = "✓" if in_text else "✗"
        print(f"{status} Chunk {i}: len={len(chunk)}")

    print(f"\nAll chunks exact substrings: {all(chunk in text for chunk in chunks)}")
