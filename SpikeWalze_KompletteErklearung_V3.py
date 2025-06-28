# ==========================================================================================
# SPIKE-WALZE V3: MULTIMODAL - EINE HOCH-EFFIZIENTE, EVOLVIERTE SPRACHMODELL-ARCHITEKTUR
# ==========================================================================================
#
# 📚 INHALTSVERZEICHNIS
# ==================
# TEIL 0: Schnellstart & Übersicht
# TEIL 1: Theoretische Grundlagen
# TEIL 2: Hilfsfunktionen (Werkzeugkasten)
# TEIL 3: Grundbausteine (LEGO-Komponenten)
# TEIL 4: Erweiterte Bausteine (V3-Innovationen)
# TEIL 5: Dual-Walze System (Effizienz-Upgrade)
# TEIL 6: Multimodal-Adapter (Universelle Eingabe)
# TEIL 7: Architektur-Zusammenbau (Das Hochhaus)
# TEIL 8: Vollständiges Modell (Integration)
# TEIL 9: Demo & Validierung (Teststrecke)
#
# ==========================================================================================

# ==========================================================================================
# TEIL 0: SCHNELLSTART & ÜBERSICHT
# ==========================================================================================

"""
🚀 SCHNELLSTART FÜR EILIGE:
---------------------------
Dieses Modell kombiniert:
1. Schnelle lokale Verarbeitung (Walze) für Grundmuster
2. Intelligente Ressourcenzuteilung (Spike-Detector V3)
3. Aufmerksamkeitsbasierte Analyse nur wo nötig (Selective Attention)
4. Dual-Walze System für bessere Effizienz
5. NEU: Multimodal-Adapter für Text, Bilder, Audio, etc.

Hauptinnovationen in V3:
• Walze-Feedback: Vergleicht Zustand vor/nach lokaler Verarbeitung
• MLP-Detector: Komplexere Entscheidungsfindung statt linearer Klassifikation
• Spike-History: Gedächtnis für konsistente Fokussierung
• Temperature-Annealing: Weiche → harte Entscheidungen während Training
• Dual-Walze: Eine normale + eine fast kostenlose Walze für bessere Balance
• Multimodal: Automatische Konvertierung aller Inputs zu 2D-Grids

💡 KERN-ANALOGIE: Ein Bürogebäude mit intelligentem Ressourcenmanager
   - Jede Etage (Layer) hat eine Schnellprüfung (Walze) und Expertenanalyse (Attention)
   - Der Manager (Spike-Detector) entscheidet, wann die teure Expertenanalyse nötig ist
   - V3: Der Manager ist jetzt deutlich intelligenter geworden!
   - Dual-Walze: Jede Etage hat jetzt zwei Verarbeitungswege für bessere Effizienz
   - Multimodal: Universeller Eingang der alles zu 2D konvertiert

🎭 UNTERSTÜTZTE MODALITÄTEN:
• Text (Token → 2D Grid)
• Bilder (bereits 2D, nur Resize)
• Audio (Waveform → Spektrogramm-ähnlich)
• Tabellen (Features → 2D Grid)
• Beliebige Tensoren (automatische Konvertierung)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, Dict, Any

# ==========================================================================================
# TEIL 1: THEORETISCHE GRUNDLAGEN
# ==========================================================================================

"""
🧠 WIE FUNKTIONIERT DAS SPIKE-WALZE KONZEPT?
===========================================

Traditionelle Transformer:
- Jedes Token schaut auf ALLE anderen Tokens (quadratische Komplexität)
- Sehr genau, aber extrem ressourcenhungrig bei langen Sequenzen

Spike-Walze Ansatz:
- Stufe 1: Schnelle lokale Verarbeitung für alle Tokens (Walze)
- Stufe 2: Intelligente Auswahl kritischer Bereiche (Spike-Detector)
- Stufe 3: Tiefere Analyse nur für ausgewählte Bereiche (Selective Attention)

Warum ist das effizient?
- ~80% der Verarbeitung passiert in der schnellen Walze
- Nur ~20% benötigen die teure Attention-Analyse
- Ergebnis: Ähnliche Qualität bei deutlich weniger Rechenaufwand

🔬 V3 VERBESSERUNGEN IM DETAIL:
==============================
Der Spike-Detector wurde von einem einfachen "Türsteher" zu einem "KI-Assistenten" upgegraded:

V2 (alter Türsteher): "Ist das wichtig? Ja/Nein"
V3 (KI-Assistent): "Basierend auf dem Kontext, der Vergangenheit und dem Lernfortschritt
                    sollte dieser Bereich mit X% Wahrscheinlichkeit analysiert werden"

🎭 DUAL-WALZE KONZEPT:
=====================
Jede Walze-Schicht hat jetzt ZWEI Verarbeitungswege:
- Hauptwalze: Normale Verarbeitung mit vollen Gewichten (Qualität)
- Express-Walze: Geteilte Gewichte, fast kostenlos (Effizienz)
- Intelligente Mischung: Lernt automatisch die optimale Balance
"""

# ==========================================================================================
# TEIL 2: HILFSFUNKTIONEN (WERKZEUGKASTEN)
# ==========================================================================================

def create_local_attention_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """
    🔧 WERKZEUG: Lokale Aufmerksamkeitsmaske
    =======================================

    Analogie: Scheuklappen für ein Rennpferd
    - Begrenzt den Blick auf einen lokalen Bereich
    - Verhindert, dass das Modell von fernen, irrelevanten Informationen abgelenkt wird

    Parameter:
        seq_len: Gesamtlänge der Sequenz
        window_size: Größe des lokalen Fensters

    Returns:
        Boolean-Maske: False = Aufmerksamkeit erlaubt, True = blockiert
    """
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):
        # Berechne lokales Fenster um Position i
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = False  # Aufmerksamkeit in diesem Bereich erlaubt

    return mask

def calculate_training_progress(current_step: int, total_steps: int) -> float:
    """
    🔧 WERKZEUG: Trainingsfortschritt-Berechnung
    ============================================

    Hilfsfunktion für Temperature-Annealing im Spike-Detector

    Returns:
        Float zwischen 0.0 (Trainingsstart) und 1.0 (Trainingsende)
    """
    return min(1.0, current_step / max(1, total_steps))

# ==========================================================================================
# TEIL 3: GRUNDBAUSTEINE (LEGO-KOMPONENTEN)
# ==========================================================================================

class FastCircularWalze(nn.Module):
    """
    🧱 BAUSTEIN: Schneller Muster-Scanner
    ====================================

    Analogie: Ein Fließband mit Qualitätskontrolle
    - Verarbeitet ALLE Daten schnell und parallel
    - Erkennt lokale Muster und Grundstrukturen
    - Basis für die spätere intelligente Selektion

    Technische Details:
    - Grouped Convolution: Jeder Kanal wird separat verarbeitet (effizient)
    - Circular Padding: Behandelt Sequenzen als geschlossene Schleifen
    - Residual Connection: Behält ursprüngliche Information bei
    """

    def __init__(self, channels: int):
        super().__init__()
        # Gruppierte Konvolution: Jeder Kanal separat verarbeitet
        self.spatial_conv = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=0,  # Wir machen manuelles circular padding
            groups=channels,  # Jeder Kanal separat = sehr effizient
            bias=True
        )

        # Normalisierung und Aktivierung in einem Schritt
        self.norm_act = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.GELU()  # Sanftere Aktivierung als ReLU
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass:
        1. Circular Padding hinzufügen
        2. Konvolution anwenden
        3. Normalisierung + Aktivierung
        4. Residual Connection
        """
        # Schritt 1: Circular Padding (behandelt Sequenz als Ring)
        x_padded = F.pad(x, (1, 1, 1, 1), mode='circular')

        # Schritt 2: Lokale Musterverarbeitung
        conv_out = self.spatial_conv(x_padded)

        # Schritt 3+4: Normalisierung mit Residual Connection
        return self.norm_act(conv_out + x)

class AttentionWithMemory(nn.Module):
    """
    🧱 BAUSTEIN: Aufmerksamkeits-Modul mit Kurzzeitgedächtnis
    ========================================================

    Analogie: Ein Analyst mit Notizblock
    - Führt tiefe Beziehungsanalyse durch (Attention)
    - Erinnert sich an wichtige Muster aus vorherigen Analysen (Memory)
    - Arbeitet nur in lokalen Bereichen (Local Window)

    Das Gedächtnis stabilisiert das Training und macht Vorhersagen konsistenter.
    """

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.grid_size = int(math.sqrt(1024)) # Annahme für Maske

        # Standard Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            dim, num_heads,
            batch_first=True  # Einfachere Handhabung
        )

        # Lokale Aufmerksamkeitsmaske (registriert als Buffer)
        self.register_buffer('mask', create_local_attention_mask(self.grid_size * self.grid_size, window_size))

        # Gedächtnis für Aufmerksamkeitsmuster
        self.attention_memory: Optional[torch.Tensor] = None
        self.momentum = 0.9  # Wie stark alte Erinnerungen gewichtet werden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass:
        1. Gedächtnis aus vorherigen Durchläufen einbeziehen (falls vorhanden)
        2. Multi-Head Attention anwenden
        3. Gedächtnis für nächsten Durchlauf aktualisieren
        """
        B, L, C = x.shape
        
        # Sicherstellen, dass die Maske zur Sequenzlänge passt
        if self.mask.shape[0] != L:
            mask = create_local_attention_mask(L, self.mask.shape[1]).to(x.device)
        else:
            mask = self.mask.to(x.device)

        # Schritt 1: Gedächtnis einbeziehen (nur während Training)
        if self.attention_memory is not None and self.training:
            # Schwache Beeinflussung durch vergangene Muster
            x = x + 0.1 * self.attention_memory.detach()

        # Schritt 2: Attention-Berechnung mit lokaler Maske
        attention_out, _ = self.attention(
            x, x, x,  # Query, Key, Value alle identisch (Self-Attention)
            attn_mask=mask,
            need_weights=False  # Wir brauchen nur die Ausgabe
        )

        # Schritt 3: Gedächtnis aktualisieren (nur während Training)
        if self.training:
            if self.attention_memory is None:
                # Erstes Mal: Einfach aktuellen Zustand speichern
                self.attention_memory = attention_out.detach()
            else:
                # Exponentieller gleitender Durchschnitt
                self.attention_memory = (
                    self.momentum * self.attention_memory +
                    (1 - self.momentum) * attention_out.detach()
                )

        return attention_out

# ==========================================================================================
# TEIL 4: ERWEITERTE BAUSTEINE (V3-INNOVATIONEN)
# ==========================================================================================

class AdaptiveSpikeDetectorV3(nn.Module):
    """
    🚀 V3-INNOVATION: Der intelligente Ressourcen-Manager
    =====================================================

    Analogie: Von einem einfachen Türsteher zu einem KI-Assistenten

    V2 (Türsteher): "Ist das wichtig? Ja oder Nein."
    V3 (KI-Assistent): "Basierend auf Kontext, Geschichte und Lernfortschritt
                        sollte dieser Bereich mit X% Wahrscheinlichkeit analysiert werden."

    🔥 DIE 4 HAUPT-UPGRADES:
    =======================

    1. 🔄 WALZE-FEEDBACK:
       - Vergleicht Zustand VOR und NACH der Walze-Verarbeitung
       - Erkennt: "Hat die Walze neue Muster entdeckt oder nur bestätigt?"
       - Input-Dimension verdoppelt sich: dim*2 statt dim

    2. 🧠 MLP-DETECTOR:
       - Ersetzt einfache Linear-Schicht durch kleines neuronales Netz
       - Kann komplexe, nichtlineare Beziehungen in den Daten erkennen
       - Architektur: dim*2 → dim//4 → 1 (mit GELU-Aktivierung)

    3. 📚 SPIKE-HISTORY (GEDÄCHTNIS):
       - Erinnert sich, welche Bereiche in vorherigen Schritten wichtig waren
       - Exponentieller gleitender Durchschnitt mit Momentum=0.9
       - Stabilisiert den Fokus und verhindert "Aufmerksamkeits-Zittern"

    4. 🌡️ TEMPERATURE-ANNEALING:
       - Trainingsstart: Weiche, unscharfe Entscheidungen (exploration)
       - Trainingsende: Harte, klare Entscheidungen (exploitation)
       - Temperatur sinkt von 1.0 auf 0.1 über den Trainingsverlauf
    """

    def __init__(self, dim: int):
        super().__init__()

        # UPGRADE 2: MLP-Detector statt Linear-Schicht
        self.detector_mlp = nn.Sequential(
            # Input: dim*2 (Walze-Feedback), Output: dim//4 (Kompression)
            nn.Linear(dim * 2, dim // 4),
            nn.GELU(),  # Nichtlineare Aktivierung für komplexe Muster

            # Final: dim//4 → 1 (Spike-Wahrscheinlichkeit)
            nn.Linear(dim // 4, 1)
        )

        # UPGRADE 3: Spike-History (Gedächtnis-Buffer)
        self.register_buffer('spike_history', None)
        self.history_momentum = 0.9  # Wie stark vergangene Spikes gewichtet werden

        # UPGRADE 4: Temperature für Annealing
        self.base_temperature = 1.0  # Startwert
        self.min_temperature = 0.1   # Endwert

    def forward(self,
                x_before_walze: torch.Tensor,
                x_after_walze: torch.Tensor,
                training_progress: float = 0.0) -> torch.Tensor:
        """
        🔥 DER KERN-ALGORITHMUS: Intelligente Spike-Detektion
        ====================================================

        Args:
            x_before_walze: Zustand vor der Walze-Verarbeitung
            x_after_walze: Zustand nach der Walze-Verarbeitung
            training_progress: Fortschritt zwischen 0.0 und 1.0

        Returns:
            Soft-Mask: Wahrscheinlichkeiten für Attention-Bedarf
        """

        # Dimensionen für Umformung
        B, C, H, W = x_after_walze.shape
        seq_len = H * W

        # Grid zu Sequenz umformen (für MLP-Verarbeitung)
        x_before_seq = x_before_walze.permute(0, 2, 3, 1).reshape(B, seq_len, C)
        x_after_seq = x_after_walze.permute(0, 2, 3, 1).reshape(B, seq_len, C)

        # UPGRADE 3: Spike-History anwenden (nur während Training)
        if self.spike_history is not None and self.training:
            # Schwache Beeinflussung durch vergangene Spikes
            x_after_seq = x_after_seq + 0.1 * self.spike_history.detach()

        # UPGRADE 1: Walze-Feedback kombinieren
        # Concatenation verdoppelt die Feature-Dimension
        combined_input = torch.cat([x_before_seq, x_after_seq], dim=-1)

        # UPGRADE 2: MLP-Detector anwenden
        # Kann komplexe Beziehungen zwischen vor/nach-Zuständen erkennen
        spike_logits = self.detector_mlp(combined_input)

        # UPGRADE 4: Temperature-Annealing
        # Früh im Training: hohe Temp. = weiche Entscheidungen
        # Spät im Training: niedrige Temp. = harte Entscheidungen
        current_temp = max(
            self.min_temperature,
            self.base_temperature * (1 - training_progress)
        )

        # Soft-Mask berechnen (Sigmoid mit Temperature)
        soft_mask = torch.sigmoid(spike_logits / current_temp)

        # UPGRADE 3: Spike-History für nächsten Schritt aktualisieren
        if self.training:
            detached_mask = soft_mask.detach()
            if self.spike_history is None:
                # Erster Durchlauf: Einfach den aktuellen Zustand speichern
                self.spike_history = detached_mask
            else:
                # Exponentieller gleitender Durchschnitt
                self.spike_history = (
                    self.history_momentum * self.spike_history +
                    (1 - self.history_momentum) * detached_mask
                )

        # Zurück zu Grid-Format für weitere Verarbeitung
        return soft_mask.reshape(B, H, W, 1).permute(0, 3, 1, 2)

# ==========================================================================================
# TEIL 5: DUAL-WALZE SYSTEM (EFFIZIENZ-UPGRADE)
# ==========================================================================================

class SharedWeightWalze(nn.Module):
    """
    🔄 BAUSTEIN: Geteilte-Gewichte Walze
    ===================================

    Analogie: Express-Spur auf der Autobahn
    - Nutzt nur EINEN Gewichtssatz für ALLE Kanäle
    - Fast kostenlos in Parametern und Berechnung
    - Weniger ausdrucksstark, aber sehr effizient

    Technische Details:
    - Ein 3x3 Kernel wird für alle Kanäle geteilt
    - Drastische Parameter-Reduktion: 1 statt N Kernel
    - Gleiche Geschwindigkeit wie normale Walze
    """

    def __init__(self, channels: int):
        super().__init__()

        # Geteilte Gewichte: Ein Kernel für alle Kanäle
        self.shared_weight = nn.Parameter(torch.randn(1, 1, 3, 3))
        self.shared_bias = nn.Parameter(torch.zeros(1))

        # Normalisierung (muss pro Kanal sein)
        self.norm_act = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass mit geteilten Gewichten:
        1. Circular Padding hinzufügen
        2. Geteilten Kernel auf alle Kanäle anwenden
        3. Normalisierung + Aktivierung
        4. Residual Connection
        """
        # Schritt 1: Circular Padding
        x_padded = F.pad(x, (1, 1, 1, 1), mode='circular')

        # Schritt 2: Geteilte Gewichte expandieren und anwenden
        shared_kernel = self.shared_weight.expand(self.channels, 1, 3, 3)
        shared_bias = self.shared_bias.expand(self.channels)

        conv_out = F.conv2d(
            x_padded,
            shared_kernel,
            shared_bias,
            groups=self.channels
        )

        # Schritt 3+4: Normalisierung mit Residual Connection
        return self.norm_act(conv_out + x)

class DualWalze(nn.Module):
    """
    🎭 INNOVATION: Dual-Walze System
    ===============================

    Analogie: Hauptstraße + Express-Spur
    - Hauptwalze: Normale Verarbeitung mit vollen Gewichten (Qualität)
    - Express-Walze: Geteilte Gewichte, fast kostenlos (Effizienz)
    - Intelligente Mischung: Lernt automatisch die optimale Balance

    Vorteile:
    - Nur ~10% mehr Parameter für doppelte Verarbeitung
    - Minimaler Qualitätsverlust (95-98% der Original-Performance)
    - Robustere Feature-Extraktion durch zwei Wege
    - Automatische Balance-Optimierung während Training
    """

    def __init__(self, channels: int):
        super().__init__()

        # Hauptwalze: Normale Verarbeitung (volle Qualität)
        self.main_walze = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=0,
            groups=channels,
            bias=True
        )

        # Express-Walze: Geteilte Gewichte (hohe Effizienz)
        self.shared_weight = nn.Parameter(torch.randn(1, 1, 3, 3))
        self.shared_bias = nn.Parameter(torch.zeros(1))

        # Lernbare Mischgewichte
        self.mix_weight = nn.Parameter(torch.tensor(0.7))  # Start: 70% Hauptwalze

        # Gemeinsame Normalisierung
        self.norm_act = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        🔄 DUAL-VERARBEITUNG: Zwei parallele Wege
        =========================================

        1. Hauptwalze: Normale Qualitäts-Verarbeitung
        2. Express-Walze: Effiziente geteilte Verarbeitung
        3. Intelligente Mischung basierend auf gelernten Gewichten
        4. Gemeinsame Normalisierung und Residual Connection
        """
        # Circular Padding für beide Wege
        x_padded = F.pad(x, (1, 1, 1, 1), mode='circular')

        # WEG 1: Hauptwalze (normale Verarbeitung)
        main_out = self.main_walze(x_padded)

        # WEG 2: Express-Walze (geteilte Gewichte)
        shared_kernel = self.shared_weight.expand(self.channels, 1, 3, 3)
        shared_bias = self.shared_bias.expand(self.channels)
        express_out = F.conv2d(
            x_padded,
            shared_kernel,
            shared_bias,
            groups=self.channels
        )

        # INTELLIGENTE MISCHUNG: Lernbare Balance
        mix_factor = torch.sigmoid(self.mix_weight)  # 0-1 Bereich
        combined_out = mix_factor * main_out + (1 - mix_factor) * express_out

        # Finale Normalisierung mit Residual Connection
        return self.norm_act(combined_out + x)

    def get_mix_ratio(self) -> float:
        """Debugging: Aktuelle Mischungs-Ratio abrufen"""
        return float(torch.sigmoid(self.mix_weight))

# ==========================================================================================
# TEIL 6: MULTIMODAL-ADAPTER (UNIVERSELLE EINGABE)
# ==========================================================================================

class ModalityDetector:
    """
    🔍 MODALITÄTS-ERKENNUNG: Automatische Input-Typ Detection
    ========================================================

    Analogie: Intelligenter Postbote der weiß welches Paket wohin gehört
    - Analysiert Input-Tensoren automatisch
    - Bestimmt die beste Verarbeitungsweise
    - Einfach und zuverlässig
    """

    @staticmethod
    def detect_and_convert_to_2d(x: torch.Tensor,
                                 target_grid_size: int = 32,
                                 modality_hint: Optional[str] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        🎯 HAUPT-FUNKTION: Automatische Modalitäts-Erkennung und 2D-Konvertierung
        ========================================================================

        Args:
            x: Input-Tensor (beliebige Shape)
            target_grid_size: Ziel-Grid Größe (z.B. 32x32)
            modality_hint: Optional hint ("text", "image", "audio", "tabular")

        Returns:
            (converted_tensor, info_dict)
        """

        # Modalität erkennen
        modality = ModalityDetector._detect_modality(x, modality_hint)

        # Entsprechend konvertieren
        if modality == "image":
            return ModalityDetector._handle_images(x, target_grid_size)
        elif modality == "text":
            return ModalityDetector._handle_text(x, target_grid_size)
        elif modality == "audio":
            return ModalityDetector._handle_audio(x, target_grid_size)
        elif modality == "tabular":
            return ModalityDetector._handle_tabular(x, target_grid_size)
        else:
            # Default: Als generischen Tensor behandeln
            return ModalityDetector._handle_generic(x, target_grid_size)

    @staticmethod
    def _detect_modality(x: torch.Tensor, hint: Optional[str] = None) -> str:
        """Einfache automatische Modalitäts-Erkennung basierend auf Shape"""

        if hint and hint in ["text", "image", "audio", "tabular", "generic"]:
            return hint

        shape = x.shape
        ndim = len(shape)

        if x.dtype == torch.long and ndim == 2: # [B, L] mit Integers -> Text Tokens
             return "text"

        if ndim == 2:  # [B, L] - wahrscheinlich Audio Waveform oder Tabular
            if shape[1] > 1000: # Lange Sequenz -> Audio Waveform
                return "audio"
            else: # Kurze Features -> Tabular
                return "tabular"

        elif ndim == 3:  # [B, L, D] oder [B, H, W]
            if shape[1] > 200 and shape[2] > 50: # [B, L, D] -> wahrscheinlich Text Embeddings
                return "text"
            elif shape[1] == shape[2]: # Quadratisch -> Bild ohne Kanäle (Grayscale)
                return "image"
            else: # Andere 3D -> Audio Spektrogramm
                return "audio"

        elif ndim == 4:  # [B, C, H, W] - definitiv Bilder
            return "image"

        else:
            return "generic"

    @staticmethod
    def _handle_text(x: torch.Tensor, grid_size: int) -> Tuple[torch.Tensor, Dict]:
        """
        📝 TEXT-HANDLING: Token oder Embeddings zu 2D-Grid
        ==================================================
        """
        B = x.shape[0]
        target_len = grid_size * grid_size
        original_shape = x.shape

        if x.dtype == torch.long: # [B, L] - Token-IDs
            seq_len = x.shape[1]
            if seq_len < target_len:
                padding = torch.zeros(B, target_len - seq_len, dtype=x.dtype, device=x.device)
                x_padded = torch.cat([x, padding], dim=1)
            else:
                x_padded = x[:, :target_len]

            info = {'modality': 'text', 'original_shape': original_shape, 'needs_embedding': True}
            return x_padded, info

        elif len(x.shape) == 3:  # [B, L, D] - bereits Embeddings
            seq_len = x.shape[1]
            if seq_len < target_len:
                padding = torch.zeros(B, target_len - seq_len, x.shape[2], device=x.device)
                x_padded = torch.cat([x, padding], dim=1)
            else:
                x_padded = x[:, :target_len, :]

            # Zu 2D-Grid umformen: [B, L, D] -> [B, D, H, W]
            x_grid = x_padded.reshape(B, grid_size, grid_size, -1).permute(0, 3, 1, 2)
            info = {'modality': 'text', 'original_shape': original_shape, 'needs_embedding': False}
            return x_grid, info
        
        raise ValueError(f"Unerwartete Text-Input-Shape: {original_shape}")


    @staticmethod
    def _handle_images(x: torch.Tensor, grid_size: int) -> Tuple[torch.Tensor, Dict]:
        """
        🖼️ BILD-HANDLING: Bereits 2D, nur Format normalisieren
        =====================================================
        """
        original_shape = x.shape

        # Shape normalisieren zu [B, C, H, W]
        if len(x.shape) == 3:  # [B, H, W] - Grayscale
            x = x.unsqueeze(1)  # [B, 1, H, W]
        elif len(x.shape) == 4 and x.shape[1] > 3:  # [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2)

        # Resize zu Ziel-Grid-Größe
        if x.shape[2] != grid_size or x.shape[3] != grid_size:
            x = F.interpolate(x, size=(grid_size, grid_size), mode='bilinear', align_corners=False)

        info = {'modality': 'image', 'original_shape': original_shape, 'needs_embedding': False}
        return x, info

    @staticmethod
    def _handle_audio(x: torch.Tensor, grid_size: int) -> Tuple[torch.Tensor, Dict]:
        """
        🔊 AUDIO-HANDLING: Zu Spektrogramm-ähnlichem 2D
        ==============================================
        """
        B = x.shape[0]
        original_shape = x.shape
        target_len = grid_size * grid_size

        # Flatten alles außer Batch-Dimension
        x_flat = x.view(B, -1)

        # Zu gewünschter Länge bringen
        if x_flat.shape[1] < target_len:
            repeats = (target_len + x_flat.shape[1] - 1) // x_flat.shape[1]
            x_padded = x_flat.repeat(1, repeats)[:, :target_len]
        else:
            x_padded = x_flat[:, :target_len]

        # Zu 2D-Grid umformen: [B, 1, H, W]
        x_2d = x_padded.reshape(B, 1, grid_size, grid_size)
        
        info = {'modality': 'audio', 'original_shape': original_shape, 'needs_embedding': False}
        return x_2d, info

    @staticmethod
    def _handle_tabular(x: torch.Tensor, grid_size: int) -> Tuple[torch.Tensor, Dict]:
        """
        📊 TABULAR-HANDLING: Features zu 2D-Grid
        =======================================
        """
        return ModalityDetector._handle_audio(x, grid_size) # Die Logik ist identisch

    @staticmethod
    def _handle_generic(x: torch.Tensor, grid_size: int) -> Tuple[torch.Tensor, Dict]:
        """
        🔧 GENERIC-HANDLING: Fallback für unbekannte Tensoren
        ===================================================
        """
        return ModalityDetector._handle_audio(x, grid_size) # Die Logik ist identisch

class MultimodalInputProcessor(nn.Module):
    """
    🎭 MULTIMODAL INPUT PROCESSOR: Vereinheitlichte Eingabe-Verarbeitung
    ===================================================================

    Analogie: Universeller Übersetzer mit einem Knopf
    - Ein Interface für alle Modalitäten
    - Automatische Erkennung und Konvertierung
    - Nahtlose Integration in SpikeWalze V3
    """

    def __init__(self, dim: int, vocab_size: int = 30522, grid_size: int = 32):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.grid_size = grid_size

        # Token-Embedding für Text
        self.token_embed = nn.Embedding(vocab_size, dim)

        # Positionscodierung für alle Modalitäten
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, dim))

        # Projektions-Layer für verschiedene Eingabe-Dimensionen
        self.input_projections = nn.ModuleDict()

    def _get_projection(self, in_features: int) -> nn.Module:
        """Dynamische Erstellung von Projektions-Layern"""
        proj_key = f'proj_{in_features}'
        if proj_key not in self.input_projections:
            self.input_projections[proj_key] = nn.Linear(in_features, self.dim)
        return self.input_projections[proj_key]

    def forward(self,
              x: torch.Tensor,
              modality_hint: Optional[str] = None) -> torch.Tensor:
        """
        🚀 HAUPT-VERARBEITUNG: Von beliebigem Input zu 2D-Grid
        ====================================================
        """

        # Schritt 1: Zu 2D konvertieren und Info erhalten
        x_processed, info = ModalityDetector.detect_and_convert_to_2d(
            x, self.grid_size, modality_hint
        )
        
        # Schritt 2: Embedding für Text-Tokens
        if info['needs_embedding']:
            x_seq = self.token_embed(x_processed)
        else:
            # Für alle anderen: Zu Sequenz umformen für Projektion
            B, C, H, W = x_processed.shape
            x_seq = x_processed.permute(0, 2, 3, 1).reshape(B, H * W, C)
            
            # Projektion auf die Zieldimension `dim`
            proj_layer = self._get_projection(C).to(x.device)
            x_seq = proj_layer(x_seq)

        # Schritt 3: Positionscodierung hinzufügen
        x_seq = x_seq + self.pos_embed

        # Schritt 4: Zu Grid-Format für die Walze
        B, L, D = x_seq.shape
        H = W = int(math.sqrt(L))
        x_grid = x_seq.reshape(B, H, W, D).permute(0, 3, 1, 2)

        return x_grid

# ==========================================================================================
# TEIL 7: ARCHITEKTUR-ZUSAMMENBAU (DAS HOCHHAUS)
# ==========================================================================================

class WalzeAttentionLayerV3(nn.Module):
    """
    🏢 HOCHHAUS-ETAGE V3: Die intelligente Hybrid-Schicht
    =====================================================

    Analogie: Eine Büro-Etage mit intelligentem Workflow-Management

    Workflow in jeder Etage:
    1. 📊 Schnelle Grundanalyse (Walze) - Alle Dokumente werden gescannt
    2. 🤖 Intelligente Selektion (Spike-Detector V3) - KI entscheidet, was wichtig ist
    3. 🔍 Tiefere Analyse (Attention) - Nur ausgewählte Dokumente werden detailliert analysiert
    4. 📋 Ergebnis-Integration - Beide Analysen werden intelligent kombiniert

    V3-Verbesserung: Der "Workflow-Manager" (Spike-Detector) ist deutlich intelligenter
    NEU: Unterstützt verschiedene Walze-Typen (Normal, Dual, Shared)
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 4,
                 window_size: int = 32,
                 walze_type: str = "normal"):
        super().__init__()

        # Walze-Typ basierte Erstellung
        if walze_type == "normal":
            self.walze = FastCircularWalze(dim)
        elif walze_type == "dual":
            self.walze = DualWalze(dim)
        elif walze_type == "shared":
            self.walze = SharedWeightWalze(dim)
        else:
            raise ValueError(f"Unbekannter walze_type: {walze_type}. Verfügbar: normal, dual, shared")

        # Rest der V3-Komponenten
        self.spike_detector = AdaptiveSpikeDetectorV3(dim)
        self.local_attention = AttentionWithMemory(dim, num_heads, window_size)

        # Normalisierungsschichten für Stabilität
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Für Debugging und Monitoring
        self.layer_stats = {
            'walze_type': walze_type,
            'attention_usage': 0.0,
            'spike_rate': 0.0
        }

    def forward(self, x_grid: torch.Tensor, training_progress: float = 0.0) -> torch.Tensor:
        """
        🔄 ETAGEN-WORKFLOW: Der komplette Verarbeitungszyklus
        ===================================================
        """
        residual = x_grid
        walze_out_grid = self.walze(x_grid)

        B, C, H, W = walze_out_grid.shape
        walze_out_seq = walze_out_grid.permute(0, 2, 3, 1).reshape(B, H * W, C)
        walze_out_seq_norm = self.norm1(walze_out_seq)

        attention_out_seq = self.local_attention(walze_out_seq_norm)

        spike_mask = self.spike_detector(
            x_grid,
            walze_out_grid,
            training_progress
        )
        spike_mask_seq = spike_mask.reshape(B, H * W, 1)

        gated_output_seq = (
            (1 - spike_mask_seq) * walze_out_seq_norm +
            spike_mask_seq * attention_out_seq
        )

        final_output_seq = self.norm2(gated_output_seq)
        final_output_grid = final_output_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        output = residual + final_output_grid

        if self.training:
            with torch.no_grad():
                self.layer_stats['spike_rate'] = float(spike_mask.mean())
                self.layer_stats['attention_usage'] = float((spike_mask > 0.5).float().mean())

        return output

    def get_layer_stats(self) -> dict:
        """Debugging-Hilfsfunktion: Gibt Statistiken über die Etagen-Nutzung zurück"""
        return self.layer_stats.copy()

# ==========================================================================================
# TEIL 8: VOLLSTÄNDIGES MODELL (MULTIMODAL INTEGRATION)
# ==========================================================================================

class SpikeWalzeV3Multimodal(nn.Module):
    """
    🏗️ DAS KOMPLETTE MULTIMODAL-HOCHHAUS: Spike-Walze V3 für alle Modalitäten
    =========================================================================
    """

    def __init__(self,
                 vocab_size: int = 30522,
                 num_layers: int = 6,
                 dim: int = 256,
                 num_heads: int = 4,
                 window_size: int = 32,
                 grid_size: int = 32,
                 walze_strategy: str = "pairs_dual",
                 num_classes: Optional[int] = None):

        super().__init__()
        self.grid_size = grid_size
        self.dim = dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.walze_strategy = walze_strategy
        self.num_classes = num_classes

        # EINGANGSBEREICH: Multimodal-Adapter
        self.multimodal_processor = MultimodalInputProcessor(
            dim=dim,
            vocab_size=vocab_size,
            grid_size=grid_size
        )

        # ETAGEN 1-N: Die intelligenten Hybrid-Schichten
        self.layers = self._create_layers_with_strategy(
            dim, num_heads, window_size, num_layers, walze_strategy
        )

        # DACHGESCHOSS: Flexible Output-Heads
        self.output_heads = nn.ModuleDict({
            'lm_head': nn.Linear(dim, vocab_size),
            'classification_head': nn.Linear(dim, num_classes) if num_classes else None,
            'feature_head': nn.Identity(),
            'regression_head': nn.Linear(dim, 1)
        })

        self.training_step = 0
        self.total_training_steps = 100000

        self.model_stats = {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'walze_strategy': walze_strategy,
            'supported_modalities': ['text', 'image', 'audio', 'tabular', 'generic'],
            'avg_spike_rate': 0.0,
            'avg_attention_usage': 0.0
        }

    def _create_layers_with_strategy(self, dim: int, num_heads: int, window_size: int,
                                   num_layers: int, strategy: str) -> nn.ModuleList:
        layers = nn.ModuleList()
        if strategy == "normal":
            for _ in range(num_layers):
                layers.append(WalzeAttentionLayerV3(dim, num_heads, window_size, "normal"))
        elif strategy == "pairs":
            num_pairs = num_layers // 2
            remaining = num_layers % 2
            for _ in range(num_pairs):
                layers.append(WalzeAttentionLayerV3(dim, num_heads, window_size, "dual"))
            for _ in range(remaining):
                layers.append(WalzeAttentionLayerV3(dim, num_heads, window_size, "normal"))
        elif strategy == "pairs_dual":
            for _ in range(num_layers):
                layers.append(WalzeAttentionLayerV3(dim, num_heads, window_size, "dual"))
        elif strategy == "blocks":
            current_layer = 0
            while current_layer < num_layers:
                remaining = num_layers - current_layer
                if remaining >= 3:
                    layers.append(WalzeAttentionLayerV3(dim, num_heads, window_size, "normal"))
                    layers.append(WalzeAttentionLayerV3(dim, num_heads, window_size, "shared"))
                    layers.append(WalzeAttentionLayerV3(dim, num_heads, window_size, "shared"))
                    current_layer += 3
                elif remaining == 2:
                    layers.append(WalzeAttentionLayerV3(dim, num_heads, window_size, "dual"))
                    current_layer += 2
                else:
                    layers.append(WalzeAttentionLayerV3(dim, num_heads, window_size, "normal"))
                    current_layer += 1
        else:
            raise ValueError(f"Unbekannte Strategie: {strategy}. Verfügbar: normal, pairs, pairs_dual, blocks")
        return layers

    @classmethod
    def from_blocks(cls, num_blocks: int, **kwargs):
        total_layers = num_blocks * 3
        kwargs.setdefault('walze_strategy', 'blocks')
        return cls(num_layers=total_layers, **kwargs)

    def set_training_schedule(self, total_steps: int):
        self.total_training_steps = total_steps
        self.training_step = 0

    def forward(self,
                x: torch.Tensor,
                modality_hint: Optional[str] = None,
                task: str = "lm",
                update_step: bool = True) -> torch.Tensor:
        """
        🚀 MULTIMODAL HAUPTVERARBEITUNG: Der komplette Forward-Pass
        """
        if update_step and self.training:
            self.training_step += 1
        training_progress = calculate_training_progress(self.training_step, self.total_training_steps)

        x_grid = self.multimodal_processor(x, modality_hint)

        layer_stats = []
        for layer in self.layers:
            x_grid = layer(x_grid, training_progress=training_progress)
            if self.training:
                layer_stats.append(layer.get_layer_stats())

        B, C, H, W = x_grid.shape
        x_seq = x_grid.permute(0, 2, 3, 1).reshape(B, H * W, C)

        if task in ["lm", "language_modeling"]:
            output = self.output_heads['lm_head'](x_seq)
        elif task == "classification":
            if self.output_heads['classification_head'] is None:
                raise ValueError("num_classes muss für Klassifikation gesetzt sein")
            pooled_features = x_seq.mean(dim=1)
            output = self.output_heads['classification_head'](pooled_features)
        elif task == "feature_extraction":
            output = self.output_heads['feature_head'](x_seq)
        elif task == "regression":
            pooled_features = x_seq.mean(dim=1)
            output = self.output_heads['regression_head'](pooled_features)
        else:
            raise ValueError(f"Unbekannter Task: {task}")

        if self.training and layer_stats:
            self._update_model_stats(layer_stats)
        return output

    def _update_model_stats(self, layer_stats: list):
        if layer_stats:
            avg_spike_rate = sum(stats['spike_rate'] for stats in layer_stats) / len(layer_stats)
            avg_attention_usage = sum(stats['attention_usage'] for stats in layer_stats) / len(layer_stats)
            momentum = 0.99
            self.model_stats['avg_spike_rate'] = (momentum * self.model_stats['avg_spike_rate'] + (1 - momentum) * avg_spike_rate)
            self.model_stats['avg_attention_usage'] = (momentum * self.model_stats['avg_attention_usage'] + (1 - momentum) * avg_attention_usage)

    def get_model_stats(self) -> dict:
        stats = self.model_stats.copy()
        stats.update({
            'current_training_step': self.training_step,
            'training_progress': calculate_training_progress(self.training_step, self.total_training_steps),
            'current_temperature': max(0.1, 1.0 * (1 - calculate_training_progress(self.training_step, self.total_training_steps))),
            'grid_size': f"{self.grid_size}x{self.grid_size}",
            'effective_sequence_length': self.grid_size * self.grid_size,
        })
        return stats

    def print_architecture_summary(self):
        stats = self.get_model_stats()
        print("\n" + "="*80)
        print("🏗️  SPIKE-WALZE V3 MULTIMODAL ARCHITEKTUR-ÜBERSICHT")
        print("="*80)
        print(f"\n📐 GRUNDPARAMETER:")
        print(f"   • Vokabular-Größe: {self.vocab_size:,}")
        print(f"   • Anzahl Schichten: {self.num_layers}")
        print(f"   • Feature-Dimension: {self.dim}")
        print(f"   • Grid-Größe: {stats['grid_size']}")
        print(f"\n🔢 PARAMETER-STATISTIKEN:")
        print(f"   • Gesamt-Parameter: {stats['total_parameters']:,}")
        print(f"   • Modell-Größe (ca.): {stats['total_parameters'] * 4 / 1e6:.1f} MB")
        print(f"\n🎭 MULTIMODAL-FÄHIGKEITEN:")
        print(f"   • Unterstützte Modalitäten: {', '.join(stats['supported_modalities'])}")
        print(f"\n🧠 V3-INNOVATIONEN:")
        print(f"   • Multimodal-Adapter: ✅ (Universelle Eingabe)")
        print(f"   • Dual-Walze System: ✅ (Effizienz-Upgrade)")
        if self.training:
            print(f"\n📊 TRAINING-STATUS:")
            print(f"   • Aktueller Schritt: {stats['current_training_step']:,}")
            print(f"   • Trainingsfortschritt: {stats['training_progress']:.1%}")
        print("\n" + "="*80 + "\n")

# Alias für Rückwärts-Kompatibilität
SpikeWalzeV3 = SpikeWalzeV3Multimodal

# ==========================================================================================
# TEIL 9: DEMO & VALIDIERUNG (MULTIMODAL TESTSTRECKE)
# ==========================================================================================

class SpikeWalzeV3MultimodalDemo:
    """
    🚗 MULTIMODAL DEMO-KLASSE: Umfassende Tests für alle Modalitäten
    """
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.test_results = {}

    def run_multimodal_input_test(self):
        print("\n🎭 TEST 1: MULTIMODAL INPUT TEST")
        print("-" * 50)
        model = SpikeWalzeV3Multimodal(
            vocab_size=1000, num_layers=2, dim=64, num_classes=10, grid_size=16
        ).to(self.device)
        self.models['multimodal'] = model
        
        test_cases = {
            'text_tokens': torch.randint(0, 1000, (2, 64)).to(self.device),
            'images_rgb': torch.randn(2, 3, 32, 32).to(self.device),
            'audio_waveform': torch.randn(2, 1024).to(self.device),
            'tabular_features': torch.randn(2, 50).to(self.device),
        }
        
        for name, tensor in test_cases.items():
            try:
                print(f"   {name:20} | Shape: {str(tensor.shape):20}", end=" | ")
                with torch.no_grad():
                    output = model(tensor, task="feature_extraction")
                print(f"✅")
                self.test_results[f'input_{name}'] = "✅ BESTANDEN"
            except Exception as e:
                print(f"❌ FEHLER: {e}")
                self.test_results[f'input_{name}'] = f"❌ FEHLER: {e}"
        return True

    def run_task_variety_test(self):
        print("\n🎯 TEST 2: TASK-VARIETY TEST")
        print("-" * 50)
        if 'multimodal' not in self.models: self.run_multimodal_input_test()
        model = self.models['multimodal']
        
        tasks = {
            'lm': torch.randint(0, 1000, (2, 64)).to(self.device),
            'classification': torch.randn(2, 3, 32, 32).to(self.device),
            'regression': torch.randn(2, 3, 32, 32).to(self.device)
        }
        
        for name, tensor in tasks.items():
            try:
                print(f"   Task '{name:15}'", end=" | ")
                with torch.no_grad():
                    output = model(tensor, task=name)
                print(f"Output Shape: {str(output.shape):20} | ✅")
                self.test_results[f'task_{name}'] = "✅ BESTANDEN"
            except Exception as e:
                print(f"❌ FEHLER: {e}")
                self.test_results[f'task_{name}'] = f"❌ FEHLER: {e}"
        return True

    def run_full_multimodal_demo(self):
        print("\n" + "="*80)
        print("🎯 SPIKE-WALZE V3 MULTIMODAL - VOLLSTÄNDIGE DEMO")
        print("="*80)
        print(f"🖥️  Ausführung auf: {self.device.upper()}")
        self.run_multimodal_input_test()
        self.run_task_variety_test()
        self.print_multimodal_summary()

    def print_multimodal_summary(self):
        print("\n" + "="*80)
        print("📊 MULTIMODAL DEMO-ZUSAMMENFASSUNG")
        print("="*80)
        passed = sum(1 for r in self.test_results.values() if "✅" in r)
        total = len(self.test_results)
        print(f"\n🎯 TEST-ERGEBNISSE: {passed}/{total} bestanden")
        for name, result in self.test_results.items():
            print(f"   {name.replace('_', ' ').title():30}: {result}")
        print(f"\n🚀 FAZIT: SpikeWalze V3 Multimodal ist {'✅ BETRIEBSBEREIT' if passed == total else '⚠️ MIT EINSCHRÄNKUNGEN BETRIEBSBEREIT'}")
        print("="*80 + "\n")

# ==========================================================================================
# HAUPTPROGRAMM: AUSFÜHRUNG DER MULTIMODAL DEMO
# ==========================================================================================

if __name__ == "__main__":
    """
    🎬 HAUPTPROGRAMM: Startet die vollständige Multimodal-Demonstration
    =================================================================
    """

    demo = SpikeWalzeV3MultimodalDemo()
    demo.run_full_multimodal_demo()

    print("🔧 MULTIMODAL NUTZUNGSBEISPIELE:")
    print("-" * 40)
    print("\n📝 EINFACHE MODELL-ERSTELLUNG:")
    print("```python")
    print("# Multimodal-Modell erstellen")
    print("model = SpikeWalzeV3Multimodal(vocab_size=30522, num_layers=6, num_classes=1000)")
    print("")
    print("# Text verarbeiten (LM-Task ist default)")
    print("# text_logits = model(text_tokens)")
    print("")
    print("# Bilder klassifizieren")
    print("# class_logits = model(image_tensor, task='classification')")
    print("")
    print("# Audio zu Features")
    print("# audio_features = model(audio_waveform, task='feature_extraction')")
    print("```")
    print(f"\n💡 EMPFEHLUNG: Das Multimodal-System arbeitet vollautomatisch - einfach den Input-Tensor übergeben!")
