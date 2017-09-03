# msg consumer
import pika, time, sys

def callback(ch, method, properties, body):
   print ("[x] Received %r" % body)
   time.sleep(body.count(b'.'))
   # print '[x] done!'
   ch.basic_ack(delivery_tag=method.delivery_tag)

# setup connection
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
# get a channel
channel = connection.channel()
# declare an exchange
channel.exchange_declare(exchange="logs",
                         exchange_type="direct")
# declare a queue and bind exchange
q = channel.queue_declare(exclusive=True)
p = channel.queue_declare(exclusive=True)

# setup routing key
key = sys.argv[1]
channel.queue_bind(exchange="logs",
                   routing_key=key,
                   queue=q.method.queue)
channel.queue_bind(exchange="logs",
                   routing_key='default',
                   queue=p.method.queue)
# consuming
# channel.basic_qos(prefetch_count=1)
channel.basic_consume(callback,
                      queue=q.method.queue)
channel.basic_consume(callback,
                      queue=p.method.queue)
print '[*] waiting for messages. To exit press CTRL+C'
channel.start_consuming()
