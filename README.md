# ml-modelo-olist

## Creacion del entorno

```bash
python3 -m venv proyecto
```

### Activacion del entorno

Para LINUX

```bash
source proyecto/bin/activate
```

Para WINDOWS

```bash
proyecto\Scripts\activate
```

### Instalacion de Dependencias

```bash
pip install -r requirements.txt
```

# Docker

Para simular el registro de eventa vamos a usar kafka, por lo que para crear las imagenes necesarias se debe seguir los siguientes comandos

```bash
docker build . -f compose/local/worker/Dockerfile -t ml_worker
```

