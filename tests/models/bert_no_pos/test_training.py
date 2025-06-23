import argparse
import copy
import os
import random
import statistics

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from transformers.models.bert_no_pos import BertNoPosConfig, BertNoPosForMaskedLM


load_dotenv()

wandb_token = os.getenv("WANDB_TOKEN")
wandb.login(key=wandb_token)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dictionnaires des noms et des IDs de boîtes possibles (cf space_tokenizer_level)
BOX_IDS = {"AM": 57, "OS": 58, "SU": 59, "FS": 60, "UX": 61, "NV": 62, "KV": 63, "PW": 64, "PX": 65, "EQ": 66, "TV": 67, "RU": 68, "JR": 69, "KN": 70, "FR": 71, "IM": 72, "JX": 73, "HO": 74, "BT": 75, "DG": 76, "UZ": 77, "LM": 78, "NZ": 79, "JQ": 80, "YZ": 81, "EM": 82, "IZ": 83, "BR": 84, "FU": 85, "GJ": 86, "JV": 87, "CY": 88, "AD": 89, "AW": 90, "IP": 91, "Z": 92, "E": 93, "PY": 94, "TY": 95, "AG": 96, "AP": 97, "JT": 98, "OU": 99, "GW": 100, "N": 101, "AU": 102, "RV": 103, "BG": 104, "TX": 105, "EZ": 106, "WX": 107, "F": 108, "LA": 109, "OT": 110, "FP": 111, "CD": 112, "LV": 113, "NW": 114, "OY": 115, "UY": 116, "KW": 117, "CN": 118, "IO": 119, "JP": 120, "DJ": 121, "DK": 122, "EV": 123, "OW": 124, "X": 125, "BW": 126, "CG": 127, "NU": 128, "Q": 129, "VZ": 130, "GL": 131, "IT": 132, "U": 133, "CQ": 134, "CE": 135, "HT": 136, "OQ": 137, "DO": 138, "G": 139, "IJ": 140, "PT": 141, "UV": 142, "K": 143, "AT": 144, "BS": 145, "CJ": 146, "CP": 147, "LS": 148, "DS": 149, "HU": 150, "AK": 151, "CF": 152, "DE": 153, "DI": 154, "DM": 155, "BZ": 156, "EO": 157, "FO": 158, "KP": 159, "BJ": 160, "HN": 161, "IL": 162, "JO": 163, "MS": 164, "NP": 165, "HW": 166, "JY": 167, "I": 168, "GU": 169, "M": 170, "SX": 171, "VY": 172, "AZ": 173, "EW": 174, "XZ": 175, "CV": 176, "J": 177, "JM": 178, "JU": 179, "W": 180, "AE": 181, "KX": 182, "BV": 183, "C": 184, "CR": 185, "ER": 186, "FG": 187, "OV": 188, "AI": 189, "EX": 190, "FK": 191, "SV": 192, "GX": 193, "LP": 194, "HI": 195, "AJ": 196, "FI": 197, "OX": 198, "A": 199, "BK": 200, "MY": 201, "SW": 202, "AQ": 203, "BC": 204, "EY": 205, "FJ": 206, "GR": 207, "GY": 208, "QS": 209, "QT": 210, "TW": 211, "WZ": 212, "EN": 213, "GV": 214, "HJ": 215, "HL": 216, "R": 217, "VX": 218, "AB": 219, "AR": 220, "FW": 221, "GO": 222, "IR": 223, "RT": 224, "HK": 225, "LN": 226, "BN": 227, "IN": 228, "NR": 229, "RS": 230, "CK": 231, "DT": 232, "LZ": 233, "MP": 234, "GQ": 235, "IS": 236, "KS": 237, "NQ": 238, "DP": 239, "ES": 240, "HZ": 241, "JW": 242, "LR": 243, "BI": 244, "CU": 245, "EG": 246, "KY": 247, "LX": 248, "SY": 249, "BD": 250, "HV": 251, "KQ": 252, "OZ": 253, "AF": 254, "DY": 255, "DZ": 256, "GS": 257, "HS": 258, "LU": 259, "RW": 260, "RZ": 261, "DV": 262, "FH": 263, "MN": 264, "O": 265, "WY": 266, "LY": 267, "S": 268, "V": 269, "BQ": 270, "CW": 271, "QW": 272, "T": 273, "BO": 274, "EF": 275, "FV": 276, "GP": 277, "HP": 278, "NX": 279, "AX": 280, "AY": 281, "MZ": 282, "CM": 283, "DW": 284, "SZ": 285, "DH": 286, "FQ": 287, "MQ": 288, "DF": 289, "OR": 290, "VW": 291, "DL": 292, "RY": 293, "BP": 294, "DQ": 295, "MU": 296, "MW": 297, "PR": 298, "QY": 299, "AV": 300, "GT": 301, "Y": 302, "GM": 303, "JK": 304, "QX": 305, "BH": 306, "BY": 307, "EK": 308, "HY": 309, "AH": 310, "AS": 311, "BL": 312, "BU": 313, "EL": 314, "FY": 315, "GN": 316, "KL": 317, "AL": 318, "CO": 319, "NT": 320, "EH": 321, "CT": 322, "EJ": 323, "GH": 324, "AC": 325, "KU": 326, "H": 327, "PS": 328, "QZ": 329, "PV": 330, "CS": 331, "FT": 332, "L": 333, "QR": 334, "HQ": 335, "QU": 336, "BM": 337, "JL": 338, "UW": 339, "DU": 340, "GK": 341, "IV": 342, "BX": 343, "KZ": 344, "PU": 345, "DR": 346, "IW": 347, "IY": 348, "XY": 349, "TZ": 350, "FZ": 351, "IU": 352, "AN": 353, "BF": 354, "CZ": 355, "LQ": 356, "NY": 357, "CL": 358, "FM": 359, "HX": 360, "QV": 361, "TU": 362, "EI": 363, "HR": 364, "MV": 365, "DX": 366, "NO": 367, "KM": 368, "KT": 369, "LT": 370, "PZ": 371, "ET": 372, "MX": 373, "GI": 374, "KR": 375, "JN": 376, "IQ": 377, "D": 378, "LW": 379, "EP": 380, "P": 381, "JS": 382, "MT": 383, "RX": 384, "HM": 385, "IX": 386, "GZ": 387, "ST": 388, "NS": 389, "FN": 390, "MO": 391, "OP": 392, "IK": 393, "EU": 394, "KO": 395, "CI": 396, "DN": 397, "FX": 398, "CH": 399, "JZ": 400, "LO": 401, "FL": 402, "CX": 403, "B": 404, "BE": 405, "MR": 406, "AO": 407, "PQ": 408}

NAMES = {
    "plum": 13, "fries": 14, "apple": 15, "lemon": 16, "melon": 17, "nectarine": 18, 
    "coconut": 19, "fennel": 20, "grape": 21, "date": 22, "prune": 23, "raspberry": 24, 
    "mango": 25, "bacon": 26, "fig": 27, "mulberry": 28, "avocado": 29, "banana": 30, 
    "berry": 31, "kiwi": 32, "guava": 33, "rice": 34, "peach": 35, "raisin": 36, "pie": 37, 
    "tangerine": 38, "steak": 39, "papaya": 40, "cherry": 41, "soup": 42, "citrus": 43, 
    "egg": 44, "apricot": 45, "milk": 46, "kale": 47, "pumpkin": 48, "orange": 49, "pear": 50, 
    "bagel": 51, "pineapple": 52, "cantaloupe": 53, "duck": 54, "lime": 55, "citron": 56
}


class SyntheticDataset(Dataset):
    def __init__(self, tokenizer, level=1, size=1000, max_length=512, add_pad_noise=False, pad_noise_prob=0.1):
        self.tokenizer = tokenizer
        self.level = level
        self.size = size
        self.max_length = max_length
        self.box_ids = list(BOX_IDS.keys())
        self.names = list(NAMES.keys())
        self.add_pad_noise = add_pad_noise
        self.pad_noise_prob = pad_noise_prob
        self.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    
    def __len__(self):
        return self.size
    
    def generate_synthetic_example(self):
        """Génère un exemple synthétique en fonction du niveau"""
        if self.level == 1:
            # Sélectionner 2 boîtes et 2 noms différents
            box_ids = random.sample(self.box_ids, 2)
            names = random.sample(self.names, 2)
            
            # Choisir aléatoirement quel nom sera demandé
            target_idx = random.randint(0, 1)
            
            # Construire la phrase
            text = f"Box {box_ids[0]} contains the {names[0]}, Box {box_ids[1]} contains the {names[1]}. "
            text += f"Therefore the {names[target_idx]} is in Box {box_ids[target_idx]}."
            
        elif self.level == 2:
            # Sélectionner 3 boîtes et 3 noms différents
            box_ids = random.sample(self.box_ids, 3)
            names = random.sample(self.names, 3)
            
            # Choisir aléatoirement quel nom sera demandé
            target_idx = random.randint(0, 2)
            
            # Construire la phrase
            text = f"Box {box_ids[0]} contains the {names[0]}, Box {box_ids[1]} contains the {names[1]}, "
            text += f"Box {box_ids[2]} contains the {names[2]}. "
            text += f"Therefore the {names[target_idx]} is in Box {box_ids[target_idx]}."
            
        elif self.level == 3:
            # Sélectionner 4 boîtes et 4 noms différents
            box_ids = random.sample(self.box_ids, 4)
            names = random.sample(self.names, 4)
            
            # Choisir aléatoirement quel nom sera demandé
            target_idx = random.randint(0, 3)
            
            # Construire la phrase
            text = f"Box {box_ids[0]} contains the {names[0]}, Box {box_ids[1]} contains the {names[1]}, "
            text += f"Box {box_ids[2]} contains the {names[2]}, Box {box_ids[3]} contains the {names[3]}. "
            text += f"Therefore the {names[target_idx]} is in Box {box_ids[target_idx]}."
        
        return text
    
    def __getitem__(self, idx):
        text = self.generate_synthetic_example()
        
        # Encoder avec les tokens spéciaux (token [CLS] ajouté en début de séquence)
        encoding = self.tokenizer(
            "[CLS]" + text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        
        # Ajouter du bruit [PAD] si demandé
        if self.add_pad_noise:
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
            # Identifier les positions où un token PAD peut être inséré : ces positions correspondent aux indices où le token précédent ET le token suivant sont non-PAD
            valid_boundaries = [i for i in range(1, len(input_ids))
                                if attention_mask[i-1] == 1 and attention_mask[i] == 1]
            
            if valid_boundaries:
                # Calculer le nombre de tokens PAD à insérer
                num_to_insert = max(1, int(len(valid_boundaries) * self.pad_noise_prob))
                
                # Sélectionner aléatoirement des positions d'insertion parmi les frontières valides
                insert_positions = random.sample(valid_boundaries, min(num_to_insert, len(valid_boundaries)))
                
                # Trier les positions d'insertion pour gérer correctement le décalage lors des insertions
                insert_positions.sort()
                offset = 0
                for pos in insert_positions:
                    pos += offset  # ajustement dû aux insertions précédentes
                    input_ids.insert(pos, self.pad_token_id)
                    attention_mask.insert(pos, 1)  # Conserver le masque à 1 pour que le modèle voie le [PAD]
                    offset += 1
                
                encoding['input_ids'] = input_ids
                encoding['attention_mask'] = attention_mask
        
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'token_type_ids': torch.tensor(encoding.get('token_type_ids', [0] * self.max_length)),
            'text': text # on conserve aussi le texte brut pour debugging
        }

class MixedLevelDataset(Dataset):
    """Dataset qui combine plusieurs niveaux avec des ratios personnalisables"""
    def __init__(self, tokenizer, level_ratios={1: 1.0, 2: 0.0, 3: 0.0}, size=1000, max_length=512, add_pad_noise=False, pad_noise_prob=0.1):
        self.tokenizer = tokenizer
        self.level_ratios = level_ratios  # dico: {niveau: ratio}
        self.size = size
        self.max_length = max_length
        
        # Normaliser les ratios pour qu'ils somment à 1
        total = sum(level_ratios.values())
        self.level_ratios = {k: v/total for k, v in level_ratios.items()}
        
        # Calculer le nombre d'exemples pour chaque niveau
        self.level_counts = {level: int(ratio * size) for level, ratio in self.level_ratios.items()}
        
        # Ajuster pour s'assurer que la somme est exactement size
        diff = size - sum(self.level_counts.values())
        if diff > 0:
            # Ajouter les exemples restants au niveau avec le plus grand ratio
            max_level = max(self.level_ratios, key=self.level_ratios.get)
            self.level_counts[max_level] += diff
        
        # Créer des datasets individuels pour chaque niveau
        self.datasets = {
            level: SyntheticDataset(tokenizer, level=level, size=count, max_length=max_length, 
                                    add_pad_noise=add_pad_noise, pad_noise_prob=pad_noise_prob)
            for level, count in self.level_counts.items() if count > 0
        }
        
        # Créer une liste d'indices pour chaque niveau
        self.level_indices = {}
        start_idx = 0
        for level, count in self.level_counts.items():
            if count > 0:
                self.level_indices[level] = list(range(start_idx, start_idx + count))
                start_idx += count
        
        # Liste de correspondance index -> (dataset_level, dataset_idx)
        self.index_map = []
        for level, indices in self.level_indices.items():
            for i, global_idx in enumerate(indices):
                self.index_map.append((level, i))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        level, dataset_idx = self.index_map[idx]
        return self.datasets[level][dataset_idx]
    
    def get_level_distribution(self):
        """Retourne la distribution des niveaux dans le dataset"""
        return {level: count/self.size for level, count in self.level_counts.items()}

class TextDataset(Dataset):
    """Dataset pour le pré-entraînement MLM uniquement"""
    def __init__(self, filename, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.inputs = []
        self.max_length = max_length

        with open(filename, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        for text in texts:
            encoding = self.tokenizer(
                "[CLS]" + text.strip(),
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
            
            self.inputs.append({
                'input_ids': torch.tensor(encoding['input_ids']),
                'attention_mask': torch.tensor(encoding['attention_mask']),
                'token_type_ids': torch.tensor(encoding.get('token_type_ids', [0] * self.max_length))
            })
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]

def apply_random_masking(input_ids, attention_mask, tokenizer):
    """Stratégie de masquage du pré-entraînement (MLM)"""
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, 0.15)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids('[MASK]')
    return input_ids, labels

def apply_penultimate_masking(input_ids, attention_mask, tokenizer):
    """Stratégie de masquage de l'avant-dernier token (qui est bien le token de la box id)"""
    labels = input_ids.clone()
    labels.fill_(-100)
    batch_size = input_ids.size(0)
    
    mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
    
    for i in range(batch_size):
        non_pad_len = int(attention_mask[i].sum().item())
        
        # Masquer l'avant-dernier token
        idx = non_pad_len - 2 if non_pad_len >= 2 else 0
        labels[i, idx] = input_ids[i, idx]
        input_ids[i, idx] = mask_token_id
    
    return input_ids, labels

def apply_box_id_masking(input_ids, attention_mask, tokenizer):
    """
    Stratégie de masquage des IDs de boîtes qui suivent 'Box' de manière aléatoire
    Ne masque jamais deux fois le même ID de boîte pour permettre l'inférence correcte
    """
    labels = input_ids.clone()
    labels.fill_(-100)  # Par défaut, tous les labels sont ignorés
    
    batch_size = input_ids.size(0)
    mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
    box_token_ids = tokenizer.encode("Box", add_special_tokens=False)  # Obtenir les tokens pour "Box"
    
    for i in range(batch_size):
        # Convertir les IDs en liste pour faciliter la recherche
        input_ids_list = input_ids[i].tolist()
        
        # Chercher toutes les positions du token "Box"
        box_positions = []
        for j in range(len(input_ids_list) - len(box_token_ids) + 1):
            if input_ids_list[j:j+len(box_token_ids)] == box_token_ids:
                box_positions.append(j)
        
        # S'il y a des "Box" dans l'exemple
        if box_positions:
            # Positions juste après "Box" et les token_ids correspondants
            box_id_positions = []
            box_ids_values = []
            
            for box_pos in box_positions:
                # S'assurer que c'est une position valide (position après Box)
                if box_pos + len(box_token_ids) < len(input_ids_list):
                    pos = box_pos + len(box_token_ids)
                    box_id_positions.append(pos)
                    box_ids_values.append(input_ids_list[pos])
            
            # Créer un dictionnaire pour regrouper les positions par valeur de box_id
            box_id_to_positions = {}
            for pos, box_id in zip(box_id_positions, box_ids_values):
                if box_id not in box_id_to_positions:
                    box_id_to_positions[box_id] = []
                box_id_to_positions[box_id].append(pos)
            
            # Sélectionner les box_ids uniques à masquer (entre 1 et min(3, nombre de box_ids uniques))
            unique_box_ids = list(box_id_to_positions.keys())
            num_masks = min(random.randint(1, 3), len(unique_box_ids))
            box_ids_to_mask = random.sample(unique_box_ids, num_masks)
            
            # Pour chaque box_id à masquer, sélectionner une seule position parmi ses occurrences
            positions_to_mask = []
            for box_id in box_ids_to_mask:
                # Choisir une position aléatoire parmi les positions de ce box_id
                pos = random.choice(box_id_to_positions[box_id])
                positions_to_mask.append(pos)
            
            # Masquer les tokens sélectionnés
            for pos in positions_to_mask:
                if attention_mask[i, pos].item() == 1:  # Ignorer les tokens de padding
                    labels[i, pos] = input_ids[i, pos]  # Sauvegarder le token original comme label
                    input_ids[i, pos] = mask_token_id  # Remplacer par un token [MASK]
    
    return input_ids, labels

def evaluate_model_per_level(model, tokenizer, device, epoch, batch_size=16, eval_size=200):
    """Évalue le modèle sur des données synthétiques pour chaque niveau avec exactement 200 exemples par niveau"""
    model.eval()
    
    # Résultats par niveau
    results = {}
    
    # Évaluer sur chaque niveau
    for level in range(1, 4):
        # Créer un dataset d'évaluation pour ce niveau avec exactement 200 exemples
        eval_dataset = SyntheticDataset(tokenizer, level=level, size=eval_size)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        total_eval_loss = 0
        total_correct = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                
                # Appliquer le masquage des IDs de boîtes
                input_ids_masked, labels = apply_penultimate_masking(input_ids.clone(), attention_mask, tokenizer)
                
                outputs = model(
                    input_ids=input_ids_masked,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                total_eval_loss += loss.item()
                
                # Calcul de l'accuracy
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                # Ne compter que les prédictions pour les tokens masqués
                mask = (labels != -100)
                correct = (predictions[mask] == labels[mask])
                
                total_correct += correct.sum().item()
                total_predictions += mask.sum().item()

        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        
        # Stocker les résultats pour ce niveau
        results[level] = {
            'loss': avg_eval_loss,
            'accuracy': accuracy,
            'total_correct': total_correct,
            'total_predictions': total_predictions
        }
    
    # Calculer aussi un score global (moyenne pondérée des résultats par niveau)
    total_accuracy = sum(results[level]['total_correct'] for level in range(1, 4)) / \
                    sum(results[level]['total_predictions'] for level in range(1, 4))
    avg_loss = statistics.mean(results[level]['loss'] for level in range(1, 4))
    
    results['global'] = {
        'loss': avg_loss,
        'accuracy': total_accuracy
    }
    
    model.train()
    return results


def is_accuracy_stagnating(accuracy_history, patience=5, threshold=0.001):
    """Vérifie si l'accuracy stagne sur les dernières époques"""
    if len(accuracy_history) < patience + 1:
        return False
    
    # Prendre les dernières valeurs d'accuracy
    recent_accuracies = accuracy_history[-patience:]
    
    # Calculer la tendance (différence entre dernière et première accuracy dans la fenêtre)
    improvement = recent_accuracies[-1] - recent_accuracies[0]
    
    # Calculer aussi la variance pour détecter les oscillations
    variance = statistics.variance(recent_accuracies) if len(recent_accuracies) > 1 else 0
    
    # Si l'amélioration est faible et la variance est faible (pas d'oscillations importantes): on considère que l'accuracy stagne
    return improvement < threshold and variance < 0.0004


def generate_pretraining_data(tokenizer, size=1000, filename="data/pretrain_synthetic.txt"):
    """Génère des données synthétiques pour le prétraînement MLM"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Répartition des niveaux pour le prétraînement
    level_ratios = {1: 0.4, 2: 0.3, 3: 0.3}  # 40% niveau 1, 30% niveau 2, 30% niveau 3
    level_counts = {level: int(ratio * size) for level, ratio in level_ratios.items()}
    
    # Ajuster pour s'assurer que la somme est exactement size
    diff = size - sum(level_counts.values())
    if diff > 0:
        level_counts[1] += diff  # Ajouter le reste au niveau 1
    
    print(f"Génération de {size} exemples pour le prétraînement:")
    for level, count in level_counts.items():
        print(f"  Niveau {level}: {count} exemples ({level_counts[level]/size*100:.1f}%)")
    
    # Créer un dataset temporaire pour la génération
    temp_dataset = SyntheticDataset(tokenizer, level=1, size=1)  # Juste pour accéder à la méthode
    
    # Générer les exemples
    with open(filename, 'w', encoding='utf-8') as f:
        for level, count in level_counts.items():
            temp_dataset.level = level  # Mettre à jour le niveau
            for i in range(count):
                text = temp_dataset.generate_synthetic_example()
                f.write(text + "\n")
    
    print(f"Données de prétraînement sauvegardées dans {filename}")
    return filename


def train_progressive_model(model, tokenizer, device, model_name=None, batch_size=16, 
                           learning_rate=5e-6, max_epochs=100, patience=2):
    """
    Entraîne le modèle avec une progression automatique basée sur la stagnation de l'accuracy globale
    Revient au meilleur modèle lors de la détection de la stagnation
    """
    best_accuracy = {1: 0.0, 2: 0.0, 3: 0.0, 'global': 0.0}
    best_model = None
    
    # Pour stocker le meilleur modèle de chaque niveau/étape
    best_models_per_stage = {}
    
    # Paramètres de la progression
    current_level_ratios = {1: 1.0, 2: 0.0, 3: 0.0}  # On commence avec 100% de niveau 1
    add_pad_noise = False
    pad_noise_prob = 0.0
    
    # Historique des métriques pour déterminer la stagnation
    loss_history = []
    accuracy_history = {1: [], 2: [], 3: [], 'global': []}
    
    # Étape actuelle de progression
    current_stage = "level_1"  # Commencer avec le niveau 1
    
    # Configuration pour le logging wandb
    config = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "model_type": model_name,
        "initial_level_distribution": current_level_ratios,
    }
    
    # Initialiser wandb pour ce run
    wandb.init(project=model_name, name="Progressive_Training", config=config)
    
    # Créer le premier dataset
    train_dataset = MixedLevelDataset(
        tokenizer, 
        level_ratios=current_level_ratios,
        size=1000, 
        add_pad_noise=add_pad_noise,
        pad_noise_prob=pad_noise_prob
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimiseur et scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_dataloader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    model.train()
    
    # Évaluation initiale avant l'entraînement
    print(f"\n===== ÉVALUATION INITIALE AVANT ENTRAÎNEMENT =====")
    initial_eval_results = evaluate_model_per_level(model, tokenizer, device, epoch=0, batch_size=batch_size)
    # Afficher les résultats par niveau
    for level in range(1, 4):
        print(f"Niveau {level} - Loss initiale: {initial_eval_results[level]['loss']:.4f}, "
              f"Accuracy initiale: {initial_eval_results[level]['accuracy']:.4f}")
        # Initialiser l'historique d'accuracy avec les valeurs initiales
        accuracy_history[level].append(initial_eval_results[level]['accuracy'])
    # Initialiser aussi l'accuracy globale
    accuracy_history['global'].append(initial_eval_results['global']['accuracy'])
    print(f"Global - Loss initiale: {initial_eval_results['global']['loss']:.4f}, "
          f"Accuracy initiale: {initial_eval_results['global']['accuracy']:.4f}")
    # Log à wandb
    wandb.log({
        f"eval_loss_level_{level}": initial_eval_results[level]['loss'] for level in range(1, 4)
    })
    wandb.log({
        f"eval_accuracy_level_{level}": initial_eval_results[level]['accuracy'] for level in range(1, 4)
    })
    wandb.log({
        "eval_loss_global": initial_eval_results['global']['loss'],
        "eval_accuracy_global": initial_eval_results['global']['accuracy']
    })
    
    # Création des répertoires pour les checkpoints
    os.makedirs(f"{model_name}_checkpoints", exist_ok=True)
    
    # Variables pour suivre la meilleure accuracy globale
    current_best_accuracy = 0.0  
    current_best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        # Recréer le dataset si les ratios ont changé
        if epoch > 0:
            train_dataset = MixedLevelDataset(
                tokenizer, 
                level_ratios=current_level_ratios,
                size=1000, 
                add_pad_noise=add_pad_noise,
                pad_noise_prob=pad_noise_prob
            )
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Afficher la distribution actuelle
            print(f"\n===== DISTRIBUTION DES NIVEAUX (Epoch {epoch+1}) =====")
            distribution = train_dataset.get_level_distribution()
            for level, ratio in distribution.items():
                print(f"Niveau {level}: {ratio*100:.1f}%")
            # Log à wandb
            for level, ratio in distribution.items():
                wandb.log({f"level_{level}_ratio": ratio, "epoch": epoch})
        
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1} ({current_stage})')
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            # Appliquer le masquage des IDs de boîtes (masquage du fine-tuning)
            input_ids_masked, labels = apply_penultimate_masking(input_ids.clone(), attention_mask, tokenizer)
            
            outputs = model(
                input_ids=input_ids_masked,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            scheduler.step()
            
            # Afficher la perte
            progress_bar.set_postfix({'train_loss': loss.item()})
            
            # Log à wandb tous les 100 batches pour ne pas surcharger
            if batch_idx % 100 == 0:
                wandb.log({"train_loss": loss.item(), "epoch": epoch, "batch": batch_idx})

        # Calculer la perte moyenne pour l'époque
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Moyenne train loss: {avg_train_loss:.4f}")
        
        # Ajouter à l'historique
        loss_history.append(avg_train_loss)
        
        # Évaluation
        eval_results = evaluate_model_per_level(model, tokenizer, device, epoch+1, batch_size=batch_size)
        
        # Afficher les résultats par niveau et mettre à jour l'historique d'accuracy
        for level in range(1, 4):
            accuracy = eval_results[level]['accuracy']
            accuracy_history[level].append(accuracy)
            print(f"Niveau {level} - Évaluation loss: {eval_results[level]['loss']:.4f}, accuracy: {accuracy:.4f}")
        
        # Mettre à jour l'historique d'accuracy globale
        global_accuracy = eval_results['global']['accuracy']
        accuracy_history['global'].append(global_accuracy)
        
        print(f"Global - Évaluation loss: {eval_results['global']['loss']:.4f}, accuracy: {global_accuracy:.4f}")
        
        # Log à wandb
        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "eval_loss_global": eval_results['global']['loss'],
            "eval_accuracy_global": eval_results['global']['accuracy']
        })
        
        for level in range(1, 4):
            wandb.log({
                f"eval_loss_level_{level}": eval_results[level]['loss'],
                f"eval_accuracy_level_{level}": eval_results[level]['accuracy'],
                "epoch": epoch + 1
            })
        
        # Vérifier si c'est la meilleure accuracy GLOBALE
        if global_accuracy > current_best_accuracy:
            current_best_accuracy = global_accuracy
            current_best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print(f"Nouvelle meilleure accuracy globale: {current_best_accuracy:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"Pas d'amélioration globale depuis {epochs_without_improvement} époques (meilleure: {current_best_accuracy:.4f}, actuelle: {global_accuracy:.4f})")
        
        # Vérifier si c'est le meilleur modèle par niveau (pour les statistiques)
        for level in range(1, 4):
            if eval_results[level]['accuracy'] > best_accuracy[level]:
                best_accuracy[level] = eval_results[level]['accuracy']
                
                # Sauvegarder le meilleur modèle pour ce niveau
                torch.save(
                    model.state_dict(), 
                    f"{model_name}_checkpoints/best_level_{level}_epoch_{epoch+1}.pt"
                )
                print(f"Nouveau meilleur modèle pour le niveau {level}: accuracy = {best_accuracy[level]:.4f}")
        
        # Meilleur modèle global
        if eval_results['global']['accuracy'] > best_accuracy['global']:
            best_accuracy['global'] = eval_results['global']['accuracy']
            best_model = copy.deepcopy(model.state_dict())
            
            # Sauvegarder le meilleur modèle global
            torch.save(
                model.state_dict(), 
                f"{model_name}_checkpoints/best_global_epoch_{epoch+1}.pt"
            )
            print(f"Nouveau meilleur modèle global: accuracy = {best_accuracy['global']:.4f}")
        
        # Vérifier si l'accuracy globale a stagné pendant 'patience' époques
        if epochs_without_improvement >= patience:
            print(f"Stagnation détectée après {patience} époques! Retour au meilleur modèle et adaptation de la stratégie.")
            
            # Revenir au meilleur modèle pour cette étape
            if current_best_model_state is not None:
                model.load_state_dict(current_best_model_state)
                print(f"Modèle restauré à la meilleure performance globale: {current_best_accuracy:.4f}")
            
            # Sauvegarder le meilleur modèle pour cette étape
            best_models_per_stage[current_stage] = {
                'state_dict': current_best_model_state,
                'accuracy': current_best_accuracy
            }
            
            # Logique de progression basée sur les ratios de niveaux actuels
            if current_stage == "level_1":
                # Première stagnation: ajouter du niveau 2
                current_level_ratios = {1: 0.9, 2: 0.1, 3: 0.0}
                current_stage = "adding_level_2"
                print("Début de l'introduction des données de niveau 2 (10%)")
            
            elif current_stage.startswith("adding_level_2"):
                # Augmenter progressivement la part des données de niveau 2
                if current_level_ratios[2] < 0.9:  # Pas encore atteint 90% de niveau 2
                    current_level_ratios[1] -= 0.2
                    current_level_ratios[2] += 0.2
                    print(f"Augmentation de la part de niveau 2 à {current_level_ratios[2]*100:.0f}%")
                else:
                    # Presque tout niveau 2, passer au niveau 3
                    current_level_ratios = {1: 0.0, 2: 0.9, 3: 0.1}
                    current_stage = "adding_level_3"
                    print("Début de l'introduction des données de niveau 3 (10%)")
            
            elif current_stage.startswith("adding_level_3"):
                # Augmenter progressivement la part des données de niveau 3
                if current_level_ratios[3] < 0.9:  # Pas encore atteint 90% de niveau 3
                    current_level_ratios[2] -= 0.2
                    current_level_ratios[3] += 0.2
                    print(f"Augmentation de la part de niveau 3 à {current_level_ratios[3]*100:.0f}%")
                else:
                    # Presque tout niveau 3, finaliser
                    current_level_ratios = {1: 0.0, 2: 0.0, 3: 1.0}
                    current_stage = "level_3_final"
                    print("Passage à 100% niveau 3 pour la phase finale")
            
            elif current_stage == "level_3_final":
                # Si toujours en stagnation après avoir tout essayé, arrêter l'entraînement
                print("Stagnation persistante après toutes les stratégies. Arrêt de l'entraînement.")
                break
            
            # Réinitialiser pour la nouvelle étape
            current_best_accuracy = 0.0  
            current_best_model_state = None
            epochs_without_improvement = 0
        
    # Restaurer le meilleur modèle global
    if best_model is not None:
        model.load_state_dict(best_model)
        print("Modèle final: utilisation du meilleur modèle global")
    
    # Sauvegarder les informations sur les meilleurs modèles par étape
    print("\n===== MEILLEURS MODÈLES PAR ÉTAPE =====")
    for stage, info in best_models_per_stage.items():
        print(f"Étape {stage}: Accuracy = {info['accuracy']:.4f}")
    
    # Sauvegarde finale
    if model_name:
        # Authentification Hugging Face
        hf_token = os.getenv("HF_TOKEN")
        huggingface_username = os.getenv("HF_USERNAME")
        if hf_token:
            login(token=hf_token)
            api = HfApi()
            repo_id = f"{huggingface_username}/{model_name}_progressive"
            # Créer le repo
            api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
            # Sauvegarder et uploader
            save_dir = f"tmp_model_progressive"
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            # Upload
            api.upload_folder(
                folder_path=save_dir,
                repo_id=repo_id,
                repo_type="model"
            )
    
    # Évaluation finale
    final_eval_results = evaluate_model_per_level(model, tokenizer, device, epoch=500, batch_size=batch_size)
    print("\n===== RÉSULTATS FINAUX =====")
    for level in range(1, 4):
        print(f"Niveau {level} - Loss finale: {final_eval_results[level]['loss']:.4f}, "
              f"Accuracy finale: {final_eval_results[level]['accuracy']:.4f}")
    print(f"Global - Loss finale: {final_eval_results['global']['loss']:.4f}, "
          f"Accuracy finale: {final_eval_results['global']['accuracy']:.4f}")

    return model, best_accuracy


def setup_model_config(vocab_size):
    """Configuration du modèle BERT"""
    config = BertNoPosConfig(
        vocab_size=vocab_size,
        hidden_size=248,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=992,
        max_position_embeddings=512,
        hidden_dropout_prob=0.05,
        attention_probs_dropout_prob=0.05,
        layer_norm_eps=1e-12,
        pad_token_id=0)
    return config









def main(model_name):
    SEED = 42
    set_seed(SEED)
    
    # Charger le tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("space_tokenizer_level")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configurer et initialiser le modèle
    config = setup_model_config(vocab_size=len(tokenizer.get_vocab()))
    model = BertNoPosForMaskedLM(config).to(device)

    # ───────────────────────────── 1. PRÉ-ENTRAÎNEMENT MLM ─────────────────────────────
    # Génération de données synthétiques pour le pré-entraînement
    print("\n===== GÉNÉRATION DE DONNÉES SYNTHÉTIQUES POUR LE PRÉ-ENTRAÎNEMENT =====")
    pretrain_temp_file = generate_pretraining_data(tokenizer, size=1000)
    pretrain_dataset = TextDataset(pretrain_temp_file, tokenizer)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=16, shuffle=True)
    
    # Génération de données synthétiques pour l'évaluation
    print("\n===== GÉNÉRATION DE DONNÉES SYNTHÉTIQUES POUR L'ÉVALUATION =====")
    eval_temp_file = generate_pretraining_data(tokenizer, size=1000, filename="data/eval_synthetic.txt")
    pretrain_eval_dataset = TextDataset(eval_temp_file, tokenizer)
    pretrain_eval_dataloader = DataLoader(pretrain_eval_dataset, batch_size=16, shuffle=False)
    
    # Configuration du pré-entraînement
    pretrain_config = {
        "learning_rate": 2e-5,
        "pretrain_epochs": 2,
        "batch_size": 16,
        "vocab_size": len(tokenizer.get_vocab()),
        "model_type": model_name
    }
    
    print("Début du pré-entraînement MLM...")
    wandb.init(project=model_name, name="Pretraining_MLM", config=pretrain_config)

    # Pré-entraînement MLM
    model.train()
    optimizer = AdamW(model.parameters(), lr=pretrain_config["learning_rate"])
    
    for epoch in range(pretrain_config["pretrain_epochs"]):
        total_loss = 0
        progress_bar = tqdm(pretrain_dataloader, desc=f'Préentraînement Epoch {epoch+1}')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            # Masquage aléatoire pour MLM
            input_ids_masked, labels = apply_random_masking(input_ids.clone(), attention_mask, tokenizer)
            
            outputs = model(
                input_ids=input_ids_masked,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'train_loss': loss.item()})
            wandb.log({"train_loss": loss.item()})
        
        # Évaluation MLM
        model.eval()
        eval_loss = 0
        eval_steps = 0
        with torch.no_grad():
            for batch in pretrain_eval_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                # Masquage aléatoire pour évaluation
                input_ids_masked, labels = apply_random_masking(input_ids.clone(), attention_mask, tokenizer)
                outputs = model(
                    input_ids=input_ids_masked,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                eval_loss += outputs.loss.item()
                eval_steps += 1
        avg_eval_loss = eval_loss / eval_steps if eval_steps > 0 else 0
        print(f"Epoch {epoch+1} - Évaluation MLM loss: {avg_eval_loss:.4f}")
        wandb.log({"eval_loss": avg_eval_loss, "epoch": epoch+1})
        
        model.train()
    
    wandb.finish()
    print("Pré-entraînement MLM terminé!")
    


    # ───────────────────────────── 2. ENTRAÎNEMENT PROGRESSIF ─────────────────────────────
    # Configuration pour l'entraînement progressif
    progressive_config = {
        "learning_rate": 2e-5,
        "batch_size": 16,
        "max_epochs": 100,
        "patience": 2 # patience sur l'accuracy avant d'ajouter des données d'un niveau supérieur (accuracy qui stagne)
    }
    
    print("\n======= DÉBUT ENTRAÎNEMENT PROGRESSIF =======")
    # Entraîner le modèle avec progression automatique
    model, accuracies = train_progressive_model(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
        batch_size=progressive_config["batch_size"],
        learning_rate=progressive_config["learning_rate"],
        max_epochs=progressive_config["max_epochs"],
        patience=progressive_config["patience"]
    )
    
    print(f"Entraînement progressif terminé avec accuracies: {accuracies}")
    # Sauvegarder le modèle final
    os.makedirs(model_name, exist_ok=True)
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)

    return model, accuracies





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="nom du modèle pour wandb et huggingface")
    args = parser.parse_args()
    main(args.model_name)