from torch.utils.data import Dataset


class DatasetModel(Dataset):
    def __init__(self, tokenized_data, pad_token_id):
        self.input_ids = tokenized_data['input_ids']
        self.labels = tokenized_data['labels']
        self.pad_token_id = pad_token_id
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": (self.input_ids[idx] != self.pad_token_id).long(),
            "labels": self.labels[idx]
        }
    