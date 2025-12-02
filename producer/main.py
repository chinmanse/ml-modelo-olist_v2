from interface.schedulerInterface import ScheduleInterface
from apscheduler.schedulers.blocking import BlockingScheduler
from kafka import KafkaProducer
import json, pytz
from datetime import datetime
import os

topic = os.environ.get("TOPIC_SHARED", "default")

print("🕒 Scheduler iniciado...")

scheduler = BlockingScheduler(timezone=pytz.timezone("America/La_Paz"))
producer = KafkaProducer(
  bootstrap_servers='kafka:9092',
  value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


def add_scheduler(callback_task, type, task_configuration):
  scheduler.add_job(
    callback_task, 
    type, 
    **task_configuration,
    max_instances=10
  )

def task_proof(self):
    msg = {
      "type": "registri",
      "data": "data",
    }
    self.producer.send(topic, msg)
    task_time = datetime.now()
    print(f"🛍 Registro de venta enviada {task_time}")

task_type = 'interval'
task_config = {
  'seconds':10, 
}

add_scheduler(task_proof, task_type, task_config)

ScheduleInterface.scheduler.start()
