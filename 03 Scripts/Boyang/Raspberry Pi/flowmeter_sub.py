import paho.mqtt.client as mqtt
from influxdb import InfluxDBClient
import json

MQTT_SERVER = "localhost"
MQTT_TOPICS = "boyang/iot/loadcell/#"
 
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
 
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_TOPICS)
 
# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))
    
    # more callbacks, etc
    data_entry = json.loads(msg.payload)
    dbclient.write_points(data_entry)
 
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
 
client.connect(MQTT_SERVER, 1883, 60)


# Set up a client for InfluxDB
dbclient = InfluxDBClient('localhost', 8086, 'admin', 'Password', 'test')


 
# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_forever()