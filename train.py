import torch
import torch.nn as nn
import numpy as np
import wave
from tqdm import tqdm


# CONFIGURAÇÕES

CONFIG = {
    "WINDOW_SIZE": 512,
    "EPOCHS": 100,
    "BATCH_SIZE": 8192,        # GPU grande; CPU use 512-2048
    "LEARNING_RATE": 3e-6,     # 0.000003
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WAV_PATH": "captura.wav",
    "OUTPUT_FILE": "resultado.am4Data",
}


# MODELO

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


# ÁUDIO

def load_wav_stereo(filename):
    with wave.open(filename, 'rb') as wf:
        assert wf.getnchannels() == 2, "WAV precisa ser estéreo!"
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16)

        left = data[0::2].astype(np.float32)
        right = data[1::2].astype(np.float32)

        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        left /= max_val
        right /= max_val

    return left, right


# PREPARAÇÃO DE DADOS

def create_windows(input_signal, output_signal, window_size, device):
    X = np.lib.stride_tricks.sliding_window_view(input_signal, window_size)
    Y = output_signal[window_size:]
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(1)
    return X_tensor, Y_tensor


# TREINO

def train(model, X, Y, epochs, lr, batch_size, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    n_samples = X.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        X_shuffled = X[perm]
        Y_shuffled = Y[perm]

        total_loss = 0
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch}", leave=False):
            x_batch = X_shuffled[i:i+batch_size]
            y_batch = Y_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / n_samples
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f}")

    return model

# EXPORTAÇÃO EM INT16

def extract_weights_int16(model):
    weights_bytes = bytearray()
    for p in model.parameters():
        w = p.detach().cpu().numpy()
        # normaliza para -1..1 e converte para int16
        w_norm = np.clip(w, -1, 1)
        w_int16 = (w_norm * 32767).astype(np.int16)
        weights_bytes.extend(w_int16.tobytes())
    return weights_bytes

def create_header(model_id="135325"):
    h = b"AM4" + b"\x00" + model_id.encode('ascii')
    return h.ljust(16, b'\x00')

def export_am4(model, filename):
    weights_bytes = extract_weights_int16(model)
    header = create_header()
    with open(filename, "wb") as f:
        f.write(header)
        f.write(weights_bytes)
    print(f"\nArquivo exportado! Tamanho: {len(weights_bytes)+16} bytes (~{(len(weights_bytes)+16)/1024:.2f} KB)")


# MAIN

def main():
    device = CONFIG["DEVICE"]
    print(f"Dispositivo: {device}")

    print("Carregando áudio...")
    left, right = load_wav_stereo(CONFIG["WAV_PATH"])

    print("Criando janelas e movendo para device...")
    X, Y = create_windows(left, right, CONFIG["WINDOW_SIZE"], device)
    print(f"Total de janelas: {X.shape[0]}")

    print("Inicializando modelo...")
    model = AmpModel(CONFIG["WINDOW_SIZE"]).to(device)

    print("Treinando...")
    model = train(model, X, Y, CONFIG["EPOCHS"], CONFIG["LEARNING_RATE"], CONFIG["BATCH_SIZE"], device)

    print("Exportando modelo...")
    export_am4(model, CONFIG["OUTPUT_FILE"])
    print("Finalizado.")

if __name__ == "__main__":
    main()