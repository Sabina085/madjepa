import datasets

class SynteticBMADAnomalyDataset():
    def __init__(self, split, subset):
        self.split = split
        if self.split == 'val':
            self.split = 'validation'
        self.dataset = datasets.load_dataset("acroitoru/bbmad_restructured")[self.split]
        self.indices = []
        for i, entry in enumerate(self.dataset['original_path']):
            if subset in entry:
                self.indices.append(i)
        self.dataset = self.dataset.select(self.indices)
        self.return_anomaly = self.split == 'train' 
        if self.return_anomaly:
            self.anomaly_dataset = datasets.load_dataset("acroitoru/medical_anomalies_qwen")['train']
            self.anomaly_dataset = self.anomaly_dataset.select(self.indices)
            assert len(self.dataset) == len(self.anomaly_dataset)
        
    def __getitem__(self, index):
        output = {}
        sample = self.dataset[index]
        image, label = sample['image'], int(sample['label']!='good')
        img = image.convert('RGB')
        if self.return_anomaly:
            abnormal_sample = self.anomaly_dataset[index]
            abnormal_image = abnormal_sample['anomaly'].convert("RGB")
            abnormal_map = abnormal_sample['anomaly_map'].convert("RGB")
            return img, abnormal_image, abnormal_map
        return img

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    from PIL import Image
    
    ds = SynteticBMADAnomalyDataset(split="train", subset= "BraTS")
    normal_img, abnormal_image, abnormal_map = ds[63]
    normal_img, abnormal_image, abnormal_map = normal_img.resize((256, 256)), abnormal_image.resize((256, 256)), abnormal_map.resize((256, 256))
    
    total_width = normal_img.width + abnormal_image.width + abnormal_map.width
    max_height = normal_img.height

    new_img = Image.new('RGB', (total_width, max_height))

    new_img.paste(normal_img, (0, 0))
    new_img.paste(abnormal_image, (normal_img.width, 0))
    new_img.paste(abnormal_map, (normal_img.width + abnormal_image.width, 0))
    new_img.save('normal_abnormal_map.jpg')