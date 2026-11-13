# Imagen de produccion del servicio Lumen (CPU). Compatible con Hugging Face Spaces (Docker).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch \
    LUMEN_DB=/tmp/lumen/lumen.sqlite \
    LUMEN_HILOS=2

WORKDIR /srv
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0 \
 && pip install -r requirements.txt \
 # pesos de SqueezeNet dentro de la imagen: el servicio no depende de internet al arrancar
 && python -c "import torchvision.models as M; M.squeezenet1_1(weights=M.SqueezeNet1_1_Weights.IMAGENET1K_V1)" \
 && chmod -R a+rX /opt/torch

COPY lumen/ lumen/
COPY app/ app/
COPY modelos/ modelos/

# usuario sin privilegios (uid 1000, como pide Hugging Face Spaces)
RUN useradd -m -u 1000 lumen
USER lumen

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/salud', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
