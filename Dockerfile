# Usa l'immagine base di PyTorch con supporto per CUDA 12.6 e CuDNN 9
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Imposta il working directory all'interno del container
WORKDIR /proformer_ce

# Copia i file necessari nel container
COPY requirements.txt .

# Installa le dipendenze Python dalla lista in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt



# Copia i file del progetto all'interno del container
COPY . /proformer_ce