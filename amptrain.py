import torch
import torch.nn as nn
import numpy as np
import wave
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# =========================
# CONFIGURAÇÕES
# =========================
CONFIG = {
    "WINDOW_SIZE": 512,
    "EPOCHS": 100,
    "BATCH_SIZE": 1024,
    "LEARNING_RATE": 0.0000003,
    "MAX_SAMPLES": None,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WAV_PATH": "captura.wav",
    "OUTPUT_FILE": "resultado.am4Data"
}

# =========================
# MODELO
# =========================
class AmpModel(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_size, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# AUDIO
# =========================
def load_wav_stereo(filename):
    with wave.open(filename, 'rb') as wf:
        assert wf.getnchannels() == 2, "WAV precisa ser estéreo!"

        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16)

        left = data[0::2].astype(np.float32)
        right = data[1::2].astype(np.float32)

        # normalização CORRETA (compartilhada)
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        left /= max_val
        right /= max_val

    return left, right

# =========================
# DATASET CUSTOM (EFICIENTE)
# =========================
class AudioDataset(Dataset):
    def __init__(self, input_signal, output_signal, window_size, max_samples=None):
        self.input = input_signal
        self.output = output_signal
        self.window_size = window_size

        total = len(input_signal) - window_size

        if max_samples and max_samples < total:
            self.indices = np.linspace(0, total - 1, max_samples, dtype=int)
        else:
            self.indices = np.arange(total)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.input[i:i+self.window_size]
        y = self.output[i+self.window_size]

        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)

# =========================
# TREINO
# =========================
def train(model, dataloader, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0

        loop = tqdm(dataloader, leave=False)
        for x_batch, y_batch in loop:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f}")

    return model, loss_history

# =========================
# PESOS
# =========================
def extract_weights(model):
    weights = []
    for param in model.parameters():
        weights.extend(param.detach().cpu().numpy().flatten())
    return np.array(weights, dtype=np.float32)

# =========================
# HEADER AM4
# =========================
def create_header(model_id="135325"):
    header = bytearray()
    header.extend(b'AM4')
    header.extend(b'\x00')
    header.extend(model_id.encode('ascii'))

    while len(header) < 16:
        header.extend(b'\x00')

    return header

# =========================
# EXPORTAÇÃO
# =========================
def export_am4(model, filename):
    weights = extract_weights(model)
    header = create_header()

    with open(filename, "wb") as f:
        f.write(header)
        f.write(weights.tobytes())

    print("\nArquivo exportado!")
    print(f"Parâmetros: {len(weights)}")
    print(f"Tamanho: {len(weights)*4/1024:.2f} KB")

# =========================
# MAIN
# =========================
def main():
    print(f"Dispositivo: {CONFIG['DEVICE']}")

    print("Carregando áudio...")
    left, right = load_wav_stereo(CONFIG["WAV_PATH"])

    print("Criando dataset...")
    dataset = AudioDataset(
        left,
        right,
        CONFIG["WINDOW_SIZE"],
        CONFIG["MAX_SAMPLES"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        drop_last=True
    )

    print("Inicializando modelo...")
    model = AmpModel(CONFIG["WINDOW_SIZE"]).to(CONFIG["DEVICE"])

    print("Treinando...")
    model, history = train(
        model,
        dataloader,
        CONFIG["DEVICE"],
        CONFIG["EPOCHS"],
        CONFIG["LEARNING_RATE"]
    )

    print("Exportando modelo...")
    export_am4(model, CONFIG["OUTPUT_FILE"])

    print("Finalizado.")

if __name__ == "__main__":
    main()