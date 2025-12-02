from kafka import KafkaConsumer
from process import Process
import json
import os

topic = os.environ.get("TOPIC_SHARED", "default")


consumer = KafkaConsumer(
    topic,
    bootstrap_servers='kafka:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    group_id='worker-group',
    auto_offset_reset='earliest',
    enable_auto_commit= True
)

print("Worker escuchando mensajes...")
for msg in consumer:
  print("Validando Acciones")
  print("service" in msg.value)
  if("service" in msg.value):
    service = Process()
    service.proccess_petition(msg.value)
  print("🧾 Recibido:", msg.value)