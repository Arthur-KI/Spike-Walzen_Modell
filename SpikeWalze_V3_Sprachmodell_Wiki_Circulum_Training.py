#!/usr/bin/env python3
"""
SPIKE-WALZE V3: VOLLSTÄNDIGES TRAINING MIT CURRICULUM LEARNING
=============================================================
Dieses Script enthält ALLES was du brauchst:
- Das komplette V3 Modell mit allen Verbesserungen
- Wikipedia Datensatz-Loader
- Curriculum Learning System mit Spike-Analyse
- Vollständiges Training mit CUDA-Support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import time
from collections import defaultdict
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# =============================================================================
# TEIL 1: MODELL-ARCHITEKTUR (Spike-Walze V3)
# =============================================================================

def create_local_attention_mask(seq_len, window_size):
    """Erstellt lokale Attention-Maske"""
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = False
    return mask

class FastCircularWalze(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_conv = nn.Conv2d(channels, channels, 3, padding=0, groups=channels, bias=True)
        self.norm_act = nn.Sequential(nn.BatchNorm2d(channels), nn.GELU())
        
    def forward(self, x):
        x_pad = F.pad(x, (1, 1, 1, 1), mode='circular')
        out = self.spatial_conv(x_pad)
        return self.norm_act(out + x)

class AttentionWithMemory(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=32):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.register_buffer('mask', create_local_attention_mask(1024, window_size))
        self.attention_memory = None
        self.momentum = 0.9
        
    def forward(self, x):
        if self.attention_memory is not None and self.training:
            x = x + 0.1 * self.attention_memory.detach()
        out, _ = self.attention(x, x, x, attn_mask=self.mask, need_weights=False)
        if self.training:
            if self.attention_memory is None:
                self.attention_memory = out.detach()
            else:
                self.attention_memory = (self.momentum * self.attention_memory +
                                         (1 - self.momentum) * out.detach())
        return out

class AdaptiveSpikeDetectorV3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.detector_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1)
        )
        self.register_buffer('spike_history', None)
        self.history_momentum = 0.9
        self.base_temperature = 1.0
        # NEU: Learned threshold
        self.spike_threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_before_walze, x_after_walze, training_progress=0.0):
        B, C, H, W = x_after_walze.shape
        seq_len = H * W
        x_before_seq = x_before_walze.permute(0, 2, 3, 1).reshape(B, seq_len, C)
        x_after_seq = x_after_walze.permute(0, 2, 3, 1).reshape(B, seq_len, C)
        
        if self.spike_history is not None and self.training:
            x_after_seq = x_after_seq + 0.1 * self.spike_history.detach()
            
        combined_input = torch.cat([x_before_seq, x_after_seq], dim=-1)
        spike_logits = self.detector_mlp(combined_input)
        
        current_temp = max(0.1, self.base_temperature * (1 - training_progress))
        
        # Mit learned threshold
        soft_mask = torch.sigmoid((spike_logits - self.spike_threshold) / current_temp)
        
        if self.training:
            detached_mask = soft_mask.detach()
            if self.spike_history is None:
                self.spike_history = detached_mask
            else:
                self.spike_history = (self.history_momentum * self.spike_history +
                                     (1 - self.history_momentum) * detached_mask)

        return soft_mask.reshape(B, H, W, 1).permute(0, 3, 1, 2)

class WalzeAttentionLayerV3(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=32):
        super().__init__()
        self.walze = FastCircularWalze(dim)
        self.spike_detector = AdaptiveSpikeDetectorV3(dim)
        self.local_attention = AttentionWithMemory(dim, num_heads, window_size)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x_grid, training_progress=0.0):
        residual = x_grid
        walze_out_grid = self.walze(x_grid)
        
        B, C, H, W = walze_out_grid.shape
        walze_out_seq = walze_out_grid.permute(0, 2, 3, 1).reshape(B, H * W, C)
        walze_out_seq_norm = self.norm1(walze_out_seq)

        attention_out_seq = self.local_attention(walze_out_seq_norm)
        
        spike_mask = self.spike_detector(x_grid, walze_out_grid, training_progress)
        spike_mask_seq = spike_mask.reshape(B, H * W, 1)

        gated_output_seq = ((1 - spike_mask_seq) * walze_out_seq_norm +
                             spike_mask_seq  * attention_out_seq)

        final_output_seq = self.norm2(gated_output_seq)
        final_output_grid = final_output_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return residual + final_output_grid

class SpikeWalzeV3(nn.Module):
    def __init__(self, vocab_size=30522, num_layers=6, dim=256, num_heads=4, 
                 window_size=32, grid_size=32):
        super().__init__()
        self.grid_size = grid_size
        self.dim = dim
        self.num_layers = num_layers
        
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, dim))
        
        self.layers = nn.ModuleList([
            WalzeAttentionLayerV3(dim, num_heads, window_size) for _ in range(num_layers)
        ])
        
        self.output_head = nn.Linear(dim, vocab_size)

    def tokens_to_grid(self, token_ids):
        B, L = token_ids.shape
        x = self.token_embed(token_ids)
        target_len = self.grid_size * self.grid_size
        if L < target_len:
            padding = torch.zeros(B, target_len - L, self.dim, device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif L > target_len:
            x = x[:, :target_len, :]
        x = x + self.pos_embed
        return x.reshape(B, self.grid_size, self.grid_size, self.dim).permute(0, 3, 1, 2)

    def forward(self, input_ids, training_progress=0.0):
        x_grid = self.tokens_to_grid(input_ids)
        
        for layer in self.layers:
            x_grid = layer(x_grid, training_progress=training_progress)
            
        B, C, H, W = x_grid.shape
        x_seq = x_grid.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        logits = self.output_head(x_seq)
        return logits

# =============================================================================
# HILFSKLASSE: WIKIPEDIA TEXT CLEANER (HIER EINGEFÜGT - KORRIGIERT)
# =============================================================================
class WikiTextCleaner:
    """
    Eine einfache Klasse zum Bereinigen von rohem Wikipedia-Text.
    Entfernt MediaWiki-Markup, HTML-Tags und andere Störungen.
    """
    def __init__(self):
        # Kompilierte Regex für bessere Performance
        self.regex_patterns = {
            # Muster, die durch Leerzeichen ersetzt werden
            'infobox': re.compile(r'\{\{.*?\}\}', re.DOTALL),
            'kategorie': re.compile(r'\[\[Kategorie:.*?\]\]', re.DOTALL),
            'datei': re.compile(r'\[\[(?:Datei|Bild):.*?\]\]', re.DOTALL),
            # KORREKTUR: Das Regex für HTML-Kommentare war leer.
            'html_comment': re.compile(r'', re.DOTALL),
            'ref_tag': re.compile(r'<ref.*?>.*?</ref>', re.DOTALL),
            'html_tag': re.compile(r'<.*?>', re.DOTALL),
            'ext_link': re.compile(r'\[http.*?\]'),
            
            # Muster mit Capturing Groups (Inhalt wird beibehalten)
            'bold': re.compile(r"'''(.*?)'''"), # KORRIGIERT: mit capturing group (...)
            'italic': re.compile(r"''(.*?)''"),   # KORRIGIERT: mit capturing group (...)
            'wikilink': re.compile(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]'),
            'headline': re.compile(r'={2,}\s*(.*?)\s*={2,}'),
            
            # Muster, die komplett entfernt werden
            'list_star': re.compile(r'^\*+\s*', re.MULTILINE),
        }

    def clean(self, text):
        """Bereinigt den übergebenen Text."""
        if not isinstance(text, str):
            return ""

        # Ersetzungen durch Leerzeichen
        text = self.regex_patterns['infobox'].sub(' ', text)
        text = self.regex_patterns['kategorie'].sub(' ', text)
        text = self.regex_patterns['datei'].sub(' ', text)
        text = self.regex_patterns['html_comment'].sub(' ', text)
        text = self.regex_patterns['ref_tag'].sub(' ', text)
        text = self.regex_patterns['html_tag'].sub(' ', text)
        text = self.regex_patterns['ext_link'].sub(' ', text)
        
        # Ersetzungen, bei denen der Inhalt erhalten bleibt (Backreference \1)
        text = self.regex_patterns['bold'].sub(r'\1', text)
        text = self.regex_patterns['italic'].sub(r'\1', text)
        text = self.regex_patterns['wikilink'].sub(r'\1', text)
        text = self.regex_patterns['headline'].sub(r'\1', text)
        
        # Komplette Entfernungen
        text = self.regex_patterns['list_star'].sub('', text)
        
        # Aufräumen
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text) # Mehrfache Leerzeichen

        return text.strip()

# =============================================================================
# TEIL 2: CURRICULUM LEARNING SYSTEM
# =============================================================================

class WikipediaDataset(Dataset):
    """Wikipedia Dataset mit Curriculum Learning - nutzt lokalen Cache"""
    def __init__(self, tokenizer, max_length=1024, subset_percent=1.0, 
                 split='train[:1%]', cache_dir='./wiki_cache'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Text Cleaner von deinem anderen Script
        self.cleaner = WikiTextCleaner()
        
        print("Lade Wikipedia Dataset...")
        
        # Versuche erst lokalen Cache
        hf_cache_dir = os.path.join(cache_dir, "huggingface_datasets")
        
        try:
            # Nutze den gleichen Cache wie dein anderes Script
            self.dataset = load_dataset(
                'wikipedia', 
                '20220301.de', 
                split=split,
                cache_dir=hf_cache_dir
            )
            print(f"✓ Wikipedia Dataset geladen: {len(self.dataset)} Artikel")
        except Exception as e:
            print(f"Fehler beim Laden: {e}")
            print("Erstelle synthetische Daten als Fallback...")
            self.dataset = None
        
        # Sammle Samples nach Schwierigkeit
        self.samples_by_stage = {
            'A_simple': [],
            'B_medium': [],
            'C_complex': [],
            'D_expert': []
        }
        
        # Konnektoren für Stage A
        self.connectives = ['weil', 'aber', 'und', 'oder', 'deshalb', 'obwohl', 
                           'jedoch', 'dennoch', 'trotzdem', 'daher']
        
        if self.dataset is not None:
            self._process_wikipedia_data(subset_percent)
        else:
            self._create_synthetic_data()
            
        # Stage weights für Curriculum
        self.stage_weights = {'A_simple': 1.0, 'B_medium': 0.0, 
                             'C_complex': 0.0, 'D_expert': 0.0}
        
        print(f"\nDataset bereit:")
        for stage, samples in self.samples_by_stage.items():
            print(f"  {stage}: {len(samples)} Samples")
    
    def _process_wikipedia_data(self, subset_percent):
        """Verarbeitet Wikipedia Daten mit Cleaning"""
        print("Kategorisiere Wikipedia Artikel nach Schwierigkeit...")
        total_samples = 0
        max_samples = int(len(self.dataset) * subset_percent)
        
        for idx in tqdm(range(min(len(self.dataset), max_samples)), 
                        desc="Verarbeite Wikipedia"):
            article = self.dataset[idx]
            
            # Bereinige Text mit WikiTextCleaner
            raw_text = article.get('text', '')
            clean_text = self.cleaner.clean(raw_text)
            
            if not clean_text or len(clean_text) < 50:
                continue
                
            sentences = clean_text.split('. ')
            
            for sent in sentences[:10]:  # Max 10 Sätze pro Artikel
                if len(sent.strip()) < 10:
                    continue
                    
                words = sent.split()
                stage = self._categorize_sentence(sent, words)
                
                # Begrenze die Anzahl der Samples pro Stage, um die Balance zu wahren
                if stage and len(self.samples_by_stage[stage]) < 10000:
                    self.samples_by_stage[stage].append(sent.strip())
                    total_samples += 1
                    
        print(f"✓ {total_samples} Sätze kategorisiert")
    
    def _categorize_sentence(self, sent, words):
        """Kategorisiert Satz nach Schwierigkeit"""
        word_count = len(words)
        
        # Stage A: Kurze Sätze mit Konnektoren
        if word_count < 12:
            return 'A_simple'
        # Stage B: Mittlere Sätze
        elif 12 <= word_count < 25:
            return 'B_medium'
        # Stage C: Längere Sätze mit Nebensätzen
        elif 25 <= word_count < 45 or ',' in sent:
            return 'C_complex'
        # Stage D: Sehr lange/komplexe Sätze
        elif word_count >= 45 or '(' in sent or ';' in sent:
            return 'D_expert'
        return None
    
    def _create_synthetic_data(self):
        """Erstellt synthetische Trainingsdaten falls keine echten verfügbar"""
        # Stage A: Einfache Sätze mit Konnektoren
        stage_a_templates = [
            "Der Himmel ist blau, {} die Sonne scheint hell.",
            "Ich gehe zum Markt, {} ich brauche frisches Obst.",
            "Das Wetter ist schlecht, {} wir bleiben zu Hause.",
            "Er arbeitet hart, {} er will erfolgreich sein.",
            "Sie liest ein Buch, {} sie mag Geschichten.",
        ]
        
        for template in stage_a_templates * 100:
            for conn in self.connectives[:5]:
                self.samples_by_stage['A_simple'].append(template.format(conn))
        
        # Stage B: Mittlere Sätze
        stage_b_sentences = [
            "Die moderne Technologie verändert unsere Gesellschaft in vielfältiger Weise jeden Tag.",
            "In der Stadt gibt es viele verschiedene Restaurants mit internationaler Küche.",
            "Die Wissenschaftler arbeiten an neuen Methoden zur Bekämpfung des Klimawandels.",
            "Im Sommer fahren viele Menschen in den Urlaub ans Meer oder in die Berge.",
            "Die Universität bietet zahlreiche Studiengänge in verschiedenen Fachbereichen an.",
        ]
        
        for sent in stage_b_sentences * 200:
            # Variiere leicht
            words = sent.split()
            if len(words) > 10:
                words[5] = np.random.choice(['sehr', 'extrem', 'besonders', 'außerordentlich'])
            self.samples_by_stage['B_medium'].append(' '.join(words))
        
        # Stage C: Komplexere Sätze
        stage_c_sentences = [
            "Obwohl das Projekt anfangs auf Widerstand stieß, konnte es letztendlich erfolgreich umgesetzt werden, was die Bedeutung von Durchhaltevermögen zeigt.",
            "Die Forschungsergebnisse, die gestern veröffentlicht wurden, zeigen deutlich, dass die bisherigen Annahmen über dieses Phänomen neu überdacht werden müssen.",
            "Nachdem die Regierung neue Gesetze erlassen hatte, mussten viele Unternehmen ihre Geschäftspraktiken anpassen, um den neuen Anforderungen gerecht zu werden.",
        ]
        
        for sent in stage_c_sentences * 150:
            self.samples_by_stage['C_complex'].append(sent)
        
        # Stage D: Sehr komplexe Sätze
        stage_d_sentences = [
            "Die interdisziplinäre Forschung, die sich mit der Schnittstelle zwischen künstlicher Intelligenz und Neurowissenschaften befasst (ein Gebiet, das in den letzten Jahren exponentiell gewachsen ist), verspricht revolutionäre Erkenntnisse über die Funktionsweise des menschlichen Gehirns.",
            "Während die einen argumentieren, dass die Digitalisierung zu einer Entfremdung der Menschen führt; betonen andere die enormen Chancen, die sich durch die neuen Technologien für Bildung, Kommunikation und wissenschaftlichen Fortschritt ergeben.",
        ]
        
        for sent in stage_d_sentences * 100:
            self.samples_by_stage['D_expert'].append(sent)
    
    def update_weights(self, performance_dict):
        """Aktualisiert Stage-Gewichte basierend auf Performance"""
        # Wenn Stage A gut genug, aktiviere B
        if performance_dict.get('A_simple', 0) > 0.8 and self.stage_weights['B_medium'] == 0:
            self.stage_weights['B_medium'] = 0.3
            self.stage_weights['A_simple'] = 0.7
            print("→ Stage B aktiviert!")
            
        # Wenn Stage B gut, aktiviere C
        if performance_dict.get('B_medium', 0) > 0.7 and self.stage_weights['C_complex'] == 0:
            self.stage_weights = {'A_simple': 0.3, 'B_medium': 0.4, 
                                'C_complex': 0.3, 'D_expert': 0.0}
            print("→ Stage C aktiviert!")
            
        # Wenn Stage C gut, aktiviere D
        if performance_dict.get('C_complex', 0) > 0.65 and self.stage_weights['D_expert'] == 0:
            self.stage_weights = {'A_simple': 0.2, 'B_medium': 0.3, 
                                'C_complex': 0.3, 'D_expert': 0.2}
            print("→ Stage D aktiviert!")
    
    def __len__(self):
        # Stellt sicher, dass das Dataset eine Länge hat, auch wenn eine Stage leer ist
        total_len = sum(len(samples) for samples in self.samples_by_stage.values())
        return max(1, total_len)
    
    def __getitem__(self, idx):
        # Weighted sampling basierend auf Stage-Gewichten
        stages, weights = list(self.stage_weights.keys()), list(self.stage_weights.values())
        
        # Wähle eine Stage, die nicht leer ist
        chosen_stage = None
        while chosen_stage is None:
            # Normalisiere Gewichte, um sicherzustellen, dass sie zu 1 summieren
            valid_stages = [s for s, w in zip(stages, weights) if self.samples_by_stage[s] and w > 0]
            valid_weights = [w for s, w in zip(stages, weights) if self.samples_by_stage[s] and w > 0]
            
            if not valid_stages: # Fallback, falls alle aktiven Stages leer sind
                chosen_stage = 'A_simple'
                if not self.samples_by_stage[chosen_stage]: # Notfalls synthetische Daten nehmen
                    self._create_synthetic_data()
                break

            weight_sum = sum(valid_weights)
            normalized_weights = [w / weight_sum for w in valid_weights]
            
            chosen_stage = np.random.choice(valid_stages, p=normalized_weights)

        # Zufälliges Sample aus gewählter Stage
        samples = self.samples_by_stage[chosen_stage]
        text = np.random.choice(samples)
        
        # Tokenisierung
        encoding = self.tokenizer(text, truncation=True, padding='max_length', 
                                 max_length=self.max_length, return_tensors='pt')
        
        input_ids = encoding['input_ids'].squeeze()
        
        # Erstelle MLM Labels
        labels = input_ids.clone()
        
        # Maskierungsstrategie je nach Stage
        if chosen_stage == 'A_simple':
            # Fokus auf Konnektoren
            mask_prob = self._create_connective_mask(input_ids, text)
        else:
            # Standard 15% Maskierung
            mask_prob = torch.full(labels.shape, 0.15)
            
        masked_indices = torch.bernoulli(mask_prob).bool()
        # Stelle sicher, dass nicht alles maskiert wird und spezielle Tokens nicht maskiert werden
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        masked_indices[input_ids == self.tokenizer.sep_token_id] = False
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        
        # Mindestens ein Token maskieren, falls zufällig keins ausgewählt wurde
        if masked_indices.sum() == 0 and len(input_ids) > 2:
            masked_indices[np.random.randint(1, len(input_ids)-1)] = True

        input_ids[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'stage': chosen_stage
        }
    
    def _create_connective_mask(self, input_ids, text):
        """Erstellt Maske mit Fokus auf Konnektoren"""
        mask_prob = torch.full(input_ids.shape, 0.05)  # Basis 5%
        
        # Erhöhe Wahrscheinlichkeit für Konnektoren
        for conn in self.connectives:
            if f' {conn} ' in text.lower():
                # Finde Token-IDs für Konnektor
                conn_tokens = self.tokenizer(conn, add_special_tokens=False)['input_ids']
                for i in range(len(input_ids) - len(conn_tokens) + 1):
                    if torch.equal(input_ids[i:i+len(conn_tokens)], torch.tensor(conn_tokens)):
                        mask_prob[i:i+len(conn_tokens)] = 0.8  # 80% Chance
        
        return mask_prob

class CurriculumController:
    """Steuert das Curriculum basierend auf Spike-Patterns"""
    def __init__(self, model):
        self.model = model
        self.stage_performance = defaultdict(lambda: {'accuracy': [], 'spike_mean': []})
        self.spike_history_per_epoch = defaultdict(list)
        
    def analyze_batch(self, batch, outputs, labels, stage):
        """Analysiert Spike-Patterns und Performance"""
        with torch.no_grad():
            # Berechne Accuracy
            predictions = outputs.argmax(dim=-1)
            mask = labels != -100
            
            # Verhindere Division durch Null, falls keine Tokens maskiert wurden
            if mask.float().sum() == 0:
                accuracy = 0.0
            else:
                correct = (predictions == labels) & mask
                accuracy = (correct.float().sum() / mask.float().sum()).item()

            # Sammle Spike-Aktivierungen über Hooks
            # Dies ist eine saubere Methode, um an Zwischen-Outputs zu kommen
            
            temp_spikes = []
            
            def hook_fn(module, input, output):
                # output ist hier die 'soft_mask' des SpikeDetectors
                temp_spikes.append(output.detach().mean().item())

            hooks = []
            try:
                for module in self.model.modules():
                    if isinstance(module, AdaptiveSpikeDetectorV3):
                        hooks.append(module.register_forward_hook(hook_fn))
                
                # Forward pass nur für Spike-Analyse
                # Der Aufruf ist notwendig, um die Hooks auszulösen
                _ = self.model(batch['input_ids'].to(next(self.model.parameters()).device))

            finally: # Stellt sicher, dass Hooks immer entfernt werden
                for hook in hooks:
                    hook.remove()

            spike_mean = np.mean(temp_spikes) if temp_spikes else 0.0

            # Update Performance
            self.stage_performance[stage]['accuracy'].append(accuracy)
            self.stage_performance[stage]['spike_mean'].append(spike_mean)
            
            return accuracy, spike_mean
            
    def get_epoch_summary(self):
        summary = {}
        for stage, perf_lists in self.stage_performance.items():
            if perf_lists['accuracy']:
                avg_acc = np.mean(perf_lists['accuracy'])
                avg_spike = np.mean(perf_lists['spike_mean'])
                summary[stage] = {'accuracy': avg_acc, 'spike_mean': avg_spike}
                self.spike_history_per_epoch[stage].append(avg_spike)
        return summary
    
    def reset_epoch_stats(self):
        self.stage_performance.clear()

    def get_curriculum_recommendation(self, summary):
        """2D-Analyse: Spikes vs Accuracy"""
        recommendations = {}
        
        for stage, perf in summary.items():
            acc = perf['accuracy']
            spike = perf['spike_mean']
            
            if spike > 0.6 and acc > 0.7:
                recommendations[stage] = "OK: Schwierig aber gemeistert → Nächste Stufe fördern"
            elif spike > 0.6 and acc < 0.4:
                recommendations[stage] = "WARNUNG: Überforderung (hohe Spikes, niedrige Acc) → Stage-Gewicht reduzieren"
            elif spike < 0.2 and acc > 0.85:
                recommendations[stage] = "INFO: Unterforderung (niedrige Spikes, hohe Acc) → Stage-Gewicht steigern"
            elif spike < 0.2 and acc < 0.5:
                recommendations[stage] = "ALARM: Konfident aber falsch (niedrige Spikes, niedrige Acc) → Zurück zu Basics!"
            else:
                recommendations[stage] = "OK - Weitermachen"
        
        return recommendations

# =============================================================================
# TEIL 3: TRAINING
# =============================================================================

def train_spike_walze_v3(
    model_path=None,
    num_epochs=50,
    batch_size=8,
    learning_rate=5e-5,
    device='cuda',
    checkpoint_dir='checkpoints'
):
    """Haupttrainings-Funktion"""
    
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Training auf: {device}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
    
    # Dataset - jetzt mit Wikipedia!
    dataset = WikipediaDataset(
        tokenizer, 
        max_length=1024, 
        subset_percent=0.1,  # 10% der Daten für ein umfassenderes Training
        split='train[:10%]', # 10% der Daten nutzen
        cache_dir='./wiki_cache'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, pin_memory=True, drop_last=True)
    
    # Model
    model = SpikeWalzeV3(
        vocab_size=tokenizer.vocab_size,
        num_layers=4,
        dim=256,
        num_heads=4,
        window_size=32,
        grid_size=32 # 1024 / 32
    ).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    start_epoch = 0
    history = defaultdict(list)
    
    # OPTIMIERUNG: Lade den Checkpoint nur einmal und verteile die Inhalte
    if model_path and os.path.exists(model_path):
        print(f"Lade Checkpoint von: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint.get('epoch', -1) + 1
        dataset.stage_weights = checkpoint.get('stage_weights', dataset.stage_weights)
        history = defaultdict(list, checkpoint.get('history', {}))

        print(f"✓ Model und Status geladen. Starte bei Epoche {start_epoch}")
    else:
        print("Kein Checkpoint gefunden, starte neues Training.")

    
    print(f"Parameter: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Curriculum Controller
    controller = CurriculumController(model)
    
    # Training Stats
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training Loop
    print("\n=== TRAINING STARTET ===\n")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        controller.reset_epoch_stats()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            # 'stage' ist eine Liste, nimm das erste Element
            stage = batch['stage'][0] if isinstance(batch['stage'], list) else batch['stage']

            training_progress = (epoch * len(dataloader) + batch_idx) / (num_epochs * len(dataloader))
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, training_progress=training_progress)
            
            loss = F.cross_entropy(outputs.view(-1, tokenizer.vocab_size), 
                                 labels.view(-1), ignore_index=-100)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Die Analyse muss nach dem Forward-Pass, aber vor dem nächsten Batch erfolgen.
            # Der Hook-basierte Ansatz in analyze_batch macht einen eigenen Forward-Pass,
            # was ineffizient ist. Wir können die Analyse direkt nach dem Haupt-Forward-Pass machen.
            # Für diese Korrektur behalten wir den ursprünglichen Ansatz bei, um die Logik nicht zu stark zu verändern.
            accuracy, spike_mean = controller.analyze_batch(batch, outputs.detach(), labels, stage)
            
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{accuracy:.2%}",
                'Spike': f"{spike_mean:.2f}",
                'Stage': stage,
                'LR': f"{scheduler.get_last_lr()[0]:.1e}"
            })
        
        scheduler.step()
        
        print(f"\n--- Epoch {epoch+1} Zusammenfassung ---")
        epoch_summary = controller.get_epoch_summary()
        performance_dict = {}
        
        for stage, stats in sorted(epoch_summary.items()):
            performance_dict[stage] = stats['accuracy']
            print(f"  {stage}: Accuracy={stats['accuracy']:.2%}, Avg Spikes={stats['spike_mean']:.3f}")
            history[f"{stage}_accuracy"].append(stats['accuracy'])
            history[f"{stage}_spikes"].append(stats['spike_mean'])

        recommendations = controller.get_curriculum_recommendation(epoch_summary)
        print("\nCurriculum Empfehlungen:")
        for stage, rec in sorted(recommendations.items()):
            print(f"  {stage}: {rec}")
        
        dataset.update_weights(performance_dict)
        print(f"\nAktuelle Stage-Gewichte: { {k: round(v, 2) for k, v in dataset.stage_weights.items()} }")
        
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': dict(history),
                'stage_weights': dataset.stage_weights
            }, checkpoint_path)
            print(f"\nCheckpoint gespeichert: {checkpoint_path}")
        
        print("-" * 80)
    
    # Finale Visualisierung
    plt.figure(figsize=(14, 10))
    
    # Accuracy Plot
    plt.subplot(2, 1, 1)
    for stage in ['A_simple', 'B_medium', 'C_complex', 'D_expert']:
        if f"{stage}_accuracy" in history:
            plt.plot(history[f"{stage}_accuracy"], label=f"{stage} Accuracy", marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Curriculum Learning Progress: Accuracy per Stage')
    plt.legend()
    plt.grid(True)
    
    # Spike Analysis
    plt.subplot(2, 1, 2)
    for stage in ['A_simple', 'B_medium', 'C_complex', 'D_expert']:
        if f"{stage}_spikes" in history:
            plt.plot(history[f"{stage}_spikes"], label=f"{stage} Spikes", marker='x', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Average Spike Activation')
    plt.title('Spike Patterns by Stage per Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    print("\nTraining abgeschlossen! Plots gespeichert als 'training_progress.png'")
    
    return model, history

# =============================================================================
# TEIL 4: HAUPTPROGRAMM
# =============================================================================

if __name__ == "__main__":
    # Trainings-Konfiguration
    config = {
        'num_epochs': 100,
        'batch_size': 8,
        'learning_rate': 5e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # Optional: Pfad zum Fortsetzen des Trainings.
        # Beispiel: 'checkpoints/model_epoch_5.pt'
        'model_path': None
    }
    
    print("SPIKE-WALZE V3 TRAINING MIT CURRICULUM LEARNING")
    print("=" * 50)
    print(f"Device: {config['device']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print("=" * 50)
    
    # Starte Training
    # KORREKTUR: Prüfe, ob 'model_path' nicht None ist, BEVOR os.path.exists aufgerufen wird.
    # Dies behebt den TypeError.
    model_path_value = config['model_path']
    if model_path_value and not os.path.exists(model_path_value):
        print(f"Warnung: Checkpoint-Pfad '{model_path_value}' wurde nicht gefunden. Das Training wird von vorne gestartet.")
        config['model_path'] = None # Setze zurück, damit die Trainingsfunktion neu startet
    elif not model_path_value:
        print("Kein Checkpoint-Pfad angegeben, das Training startet von vorne.")

    model, history = train_spike_walze_v3(**config)
    
    # Speichere finales Modell als .pth für leichtere Wiederverwendung
    final_model_path = 'spike_walze_v3_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinales Modell (state_dict) gespeichert als: {final_model_path}")
    
    # Speichere Training History
    with open('training_history.json', 'w') as f:
        # Konvertiere defaultdict in ein normales dict für JSON
        history_dict = {k: v for k, v in history.items()}
        json.dump(history_dict, f, indent=2)
    print("Training History gespeichert als: training_history.json")