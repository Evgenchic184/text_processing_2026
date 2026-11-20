from src.utils import tokenize
from collections import defaultdict
import pandas as pd
class DataLoader:
    def __init__(
        self, 
        data_path: str = 'data/tweets.csv',
        window_size: int = 2
    ):
        data = pd.read_csv('data/tweets.csv')
        data.rename(columns={'tweet_text': 'text'}, inplace=True)
        data.drop_duplicates(subset=['text'], inplace=True)
        data = data.reset_index(drop=True)
        label = 'cyberbullying_type'
        text = 'text'
        text_prepared = 'text_prepared'

        data[text_prepared] = data[text].apply(tokenize)

        self.window_size = window_size
        
        self.pairs = defaultdict(set)
        self.texts = data[text_prepared].tolist()
        self.data = None

    def build_pairs(self):
        for words in self.texts:
            self.build_pair(words)

        for key in self.pairs.keys():
            self.pairs[key] = sorted(list(self.pairs[key]))

        self.data = [
            (context_word, middle_word) for middle_word in sorted(self.pairs.keys())
            for context_word in self.pairs[middle_word]
        ]
    
    def build_pair(self, words):
        for i in range(0, len(words) - 2 * self.window_size - 1):
            middle_word = words[i + self.window_size]
            for j in range(0, 2 * self.window_size + 1):
                if j == self.window_size:
                    continue
                self.pairs[middle_word].update([words[i + j]])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx > len(self):
            return IndexError
        context_word, middle_word = self.data[idx]
        return context_word, middle_word