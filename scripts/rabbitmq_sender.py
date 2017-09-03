import sys, pika, time
import numpy as np

# setup connection to rabbitmq 'localhost'
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
# get a channel
channel = connection.channel()
# declare an exchange
channel.exchange_declare(exchange="logs",
                         exchange_type='direct')
# create a msg
cnt = int(sys.argv[1])
message = ' '.join(sys.argv[2:]) or "Hello World!"
# declare a queue
# channel.queue_declare(queue='task_q')
for id in range(0, cnt):
   msg = ' '.join([message, str(id)])
   time.sleep(2*np.random.rand())
   # publish the msg
   if id%2 == 0:
      channel.basic_publish(exchange='logs',
                            routing_key='even',
                            body='even number: ' + msg,
                            properties=pika.BasicProperties(delivery_mode=2))
   else:
      channel.basic_publish(exchange='logs',
                            routing_key='odd',
                            body='odd number: ' + msg,
                            properties=pika.BasicProperties(delivery_mode=2))
   print "[x] Sent %r" % msg
# close the channel
channel.close()