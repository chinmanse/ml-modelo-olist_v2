FROM python:3.11.2

# evitar buffering en logs
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
               wget vim netcat python3-magic \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./compose/local/api/requirements.txt /app/
RUN cat requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

CMD ["bash", "-c", "python main.py"]