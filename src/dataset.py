import json
import random
import re
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

class SyllogismDataset(Dataset):
    def __init__(self, 
                 file_path: str, 
                 tokenizer: Any, 
                 max_length: int = 512, 
                 is_train: bool = True,
                 augment_distractors: bool = False,
                 num_distractors: int = 2):
        """
        Args:
            file_path: Path to the JSON training data.
            tokenizer: HuggingFace tokenizer.
            max_length: Model context window limit.
            is_train: Whether to include labels.
            augment_distractors: If True, injects irrelevant sentences (Subtask 2 simulation).
            num_distractors: Number of distractors to inject per sample.
        """
        self.data = self._load_data(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.augment_distractors = augment_distractors
        self.num_distractors = num_distractors
        
        # Pre-collect all valid premises for distractor generation
        self.all_valid_premises = []
        if self.augment_distractors:
            self._build_distractor_pool()

    def _load_data(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _split_sentences(self, text: str) -> List[str]:
        """
         robust sentence splitter using regex.
         Splits on '.', '?', '!' followed by whitespace.
        """
        # A simple regex that looks for sentence terminators followed by a space or end of string.
        # It keeps the terminator with the sentence.
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _build_distractor_pool(self):
        """
        Collects all sentences from 'valid' syllogisms in the dataset 
        to use as plausible-sounding but irrelevant distractors.
        """
        for item in self.data:
            if item.get('validity', False):
                # We only want to pull from valid logical statements to make it hard
                sentences = self._split_sentences(item['syllogism'])
                # Exclude the conclusion (usually the last sentence) to avoid confusing logic too much
                if len(sentences) > 1:
                    self.all_valid_premises.extend(sentences[:-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        syllogism_text = item['syllogism']
        label = item.get('validity', False)
        
        # Subtask 2 / 4 Simulation: Distractor Injection
        relevant_indices = []
        final_sentences = []
        
        sentences = self._split_sentences(syllogism_text)
        
        if self.augment_distractors:
            # ORIGINAL sentences are all relevant (simplified assumption for Task 1 data)
            # In Task 1 data, the entire syllogism is relevant logic.
            # So, initially, indices 0 to len(sentences)-1 are relevant.
            
            # 1. Create a list of (sentence, is_relevant=True)
            sentence_objs = [{'text': s, 'relevant': True} for s in sentences]
            
            # 2. Pick random distractors
            if self.all_valid_premises:
                distractors = random.sample(self.all_valid_premises, min(len(self.all_valid_premises), self.num_distractors))
                for d in distractors:
                    sentence_objs.append({'text': d, 'relevant': False})
            
            # 3. Shuffle (preserving order of relevant parts relative to each other might be important? 
            # Actually, standard syllogisms usually have specific order, but distractors break it.
            # Random shuffle is good for robust retrieval training.)
            random.shuffle(sentence_objs)
            
            # 4. Reconstruct text and indices
            final_text_parts = []
            for i, obj in enumerate(sentence_objs):
                final_text_parts.append(f"[{i}] {obj['text']}") # Optional: add index markers [0] for later extraction if model needs it
                if obj['relevant'] and label: # Only valid syllogisms have relevant premises per specs?
                     # Wait, specs say "only 'valid' syllogisms will have relevant premises".
                     # So if label is False, relevant_indices must be [].
                     relevant_indices.append(i)
            
            full_input = " ".join([o['text'] for o in sentence_objs])
            
            if not label: 
                relevant_indices = [] # Invalid logic -> No relevant premises that entail conclusion
            else:
                relevant_indices.sort()
                
        else:
            # Standard Task 1 (Binary Classification only)
            full_input = syllogism_text

        # ------------------------------------------------------------------
        # Prompt Formatting
        # ------------------------------------------------------------------
        # We want the model to output JSON.
        prompt = (
            "Analyze the following syllogism for formal validity. "
            "Ignore whether the statements are plausible in the real world. "
            "Focus ONLY on logical consistency.\n\n"
            f"Syllogism: {full_input}\n\n"
            "Return a JSON object with keys: 'validity' (boolean) and 'relevant_premises' (list of integers). "
            "For 'relevant_premises', list the indices of sentences that entail the conclusion (0-indexed sentences of the input). "
            "If invalid, 'relevant_premises' should be empty.\n"
            "Response:"
        )

        # ------------------------------------------------------------------
        # Prompt Formatting
        # ------------------------------------------------------------------
        # We want the model to output JSON.
        prompt = (
            "Analyze the following syllogism for formal validity. "
            "Ignore whether the statements are plausible in the real world. "
            "Focus ONLY on logical consistency.\n\n"
            f"Syllogism: {full_input}\n\n"
            "Return a JSON object with keys: 'validity' (boolean) and 'relevant_premises' (list of integers). "
            "For 'relevant_premises', list the indices of sentences that entail the conclusion (0-indexed sentences of the input). "
            "If invalid, 'relevant_premises' should be empty.\n"
            "Response:"
        )
        
        # Prepare Completion
        target_json = {
            "validity": label,
            "relevant_premises": relevant_indices if self.augment_distractors else []
        }
        completion = json.dumps(target_json)
        
        # Causal LM Training: Input = Prompt + Completion
        # Label = Masked Prompt + Completion
        
        # We assume the tokenizer handles the concatenation if we pass them as a list, 
        # or we concatenate manually. Safer to concatenate text for Causal LM.
        full_text = prompt + " " + completion + self.tokenizer.eos_token
        
        # Tokenize (ensure tokenizer has padding token set)
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encodings.input_ids.squeeze(0)
        attention_mask = encodings.attention_mask.squeeze(0)
        
        # Create Labels: Copy input_ids, then mask the prompt part
        labels = input_ids.clone()
        
        # Find the length of the prompt to mask it
        # Note: This is an approximation if tokenization merges tokens across the boundary.
        # A safer way is to tokenize prompt separately to get its length.
        prompt_enc = self.tokenizer(
            prompt, 
            add_special_tokens=False, # We want length of prompt logic
            return_tensors="pt"
        )
        prompt_len = prompt_enc.input_ids.shape[1]
        
        # Mask prompt (set to -100)
        # We need to be careful about strict lengths. 
        # If truncation happened, prompt_len might be larger than actual prompt in input_ids (unlikely if max_length is large)
        if prompt_len < len(labels):
            labels[:prompt_len] = -100
            
        # Also mask padding tokens (where attention_mask is 0)
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

if __name__ == "__main__":
    # Quick Test
    from transformers import AutoTokenizer
    print("Testing SyllogismDataset...")
    
    # Mock Tokenizer
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            class Out:
                input_ids = torch.zeros(1, 10)
                attention_mask = torch.ones(1, 10)
            return Out()
            
    dataset = SyllogismDataset(
        file_path="semeval_2026_task_11/data/raw/train_data.json", # Adjust path as needed for local run
        tokenizer=MockTokenizer(),
        augment_distractors=True
    )
    
    print(f"Loaded {len(dataset)} items.")
    sample = dataset[0]
    print("Sample Prompt:", sample['prompt'])
    print("Sample Completion:", sample['completion'])
