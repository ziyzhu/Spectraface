from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm
import numpy as np
import os
import shutil
import torch

class Encoder: 
    def __init__(self, encoder_name, train_dir='./train', model_path='./models/tuned_vggface2', input_shape=(160, 160)):
        self.name = encoder_name
        self.train_dir = train_dir
        self.model_path = model_path
        self.input_shape = input_shape

        if encoder_name == 'vggface2':
            self.model = InceptionResnetV1(classify=False, pretrained='vggface2')
        elif encoder_name == 'tuned_vggface2':
            self.model = InceptionResnetV1(classify=False, pretrained='vggface2')
            self.load_tuned_model()
        else:
            raise
        self.model.eval()

    def __repr__(self):
        return f'Encoder(name={self.name}, input_shape={self.input_shape})'
    
    def encode(self, image):
        resized_image = image.resize(self.input_shape)
        tensor = self.model(ToTensor()(resized_image).unsqueeze(0))
        code = tensor.detach().numpy().flatten()
        return code

    def load_tuned_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def create_train_dataset(self, train_faces):
        if os.path.exists(self.train_dir):
            shutil.rmtree(self.train_dir)
            os.makedirs(self.train_dir)

        resized_imgs = [face.image.resize(self.input_shape) for face in train_faces]
        for face, img in zip(train_faces, resized_imgs):
            subdir = f'{self.train_dir}/{face.name}'
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            fname = face.encode()
            img.save(f'{subdir}/{fname}.jpg')

        return self.get_train_dataset()
    
    def get_train_dataset(self):
        if not os.path.exists(self.train_dir):
            raise Exception('missing training dataset')

        trans = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])

        dataset = datasets.ImageFolder(self.train_dir, transform=trans)
        return dataset

    def train(self, save_model=True):
        batch_size = 32
        epochs = 100
        workers = 0 if os.name == 'nt' else 8

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = MultiStepLR(optimizer, [5, 10])

        dataset = self.get_train_dataset()
        img_inds = np.arange(len(dataset))
        np.random.shuffle(img_inds)
        train_inds = img_inds[:int(0.8 * len(img_inds))]
        val_inds = img_inds[int(0.8 * len(img_inds)):]

        train_loader = DataLoader(
            dataset,
            num_workers=workers,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_inds)
        )
        val_loader = DataLoader(
            dataset,
            num_workers=workers,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(val_inds)
        )

        loss_fn = torch.nn.CrossEntropyLoss()
        metrics = {
            'fps': training.BatchTimer(),
            'acc': training.accuracy
        }

        writer = SummaryWriter()
        writer.iteration, writer.interval = 0, 10

        print('\n\nInitial')
        print('-' * 10)
        self.model.eval()
        training.pass_epoch(
            self.model, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, 
            writer=writer
        )

        for epoch in tqdm(range(epochs)):
            print('\nEpoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)

            self.model.train()
            training.pass_epoch(
                self.model, loss_fn, train_loader, optimizer, scheduler,
                batch_metrics=metrics, show_running=True, 
                writer=writer
            )

            self.model.eval()
            training.pass_epoch(
                self.model, loss_fn, val_loader,
                batch_metrics=metrics, show_running=True,
                writer=writer
            )

            writer.close()

        if save_model:
            self.save_model()
        

