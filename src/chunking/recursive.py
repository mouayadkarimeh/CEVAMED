try:
    from .fixedTokenChunker import TextSplitter
except ImportError:  # pragma: no cover - direct script execution fallback
    from fixedTokenChunker import TextSplitter
from typing import Literal, Any
import re 
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))



def _split_text_with_regex(
    text: str, separator: str, *, keep_separator: bool | Literal["start", "end"]
) -> list[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            splits_ = re.split(f"({separator})", text)
            splits = (
                ([splits_[i] + splits_[i + 1] for i in range(0, len(splits_) - 1, 2)])
                if keep_separator == "end"
                else ([splits_[i] + splits_[i + 1] for i in range(1, len(splits_), 2)])
            )
            if len(splits_) % 2 == 0:
                splits += splits_[-1:]
            splits = (
                ([*splits, splits_[-1]])
                if keep_separator == "end"
                else ([splits_[0], *splits])
            )
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s]


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


class RecursiveCharacterTextSplitter(TextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
        self,
        separators: list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = True,  # noqa: FBT001,FBT002
        is_separator_regex: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, s_ in enumerate(separators):
            separator_ = s_ if self._is_separator_regex else re.escape(s_)
            if not s_:
                separator = s_
                break
            if re.search(separator_, text):
                separator = s_
                new_separators = separators[i + 1 :]
                break

        separator_ = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(
            text, separator_, keep_separator=self._keep_separator
        )

        # Now go merging things, recursively splitting longer texts.
        good_splits = []
        separator_ = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self._merge_splits(good_splits, separator_)
                    final_chunks.extend(merged_text)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if good_splits:
            merged_text = self._merge_splits(good_splits, separator_)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> list[str]:
        """Split the input text into smaller chunks based on predefined separators.

        Args:
            text: The input text to be split.

        Returns:
            A list of text chunks obtained after splitting.
        """
        return self._split_text(text, self._separators)



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


    

    recursvie = RecursiveCharacterTextSplitter()
    chunks = recursvie.split_text(text)
    print(chunks)
    results_plot(chunks, recursvie._chunk_size)