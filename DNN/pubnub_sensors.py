from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub, SubscribeListener
from dotenv import load_dotenv
import os
import json
import RPi.GPIO as GPIO
import time

load_dotenv()

sensors_list = ["buzzer"]
data = {}

pnconfig = PNConfiguration()
pnconfig.subscribe_key = os.getenv("PUBNUB_SUBSCRIBE_KEY")
print(os.getenv("PUBNUB_SUBSCRIBE_KEY"))
pnconfig.publish_key = os.getenv("PUBNUB_PUBLISH_KEY")
pnconfig.user_id = "pi-0"
pnconfig.enable_subscribe = True

pubnub = PubNub(pnconfig)


def handle_message(message):
    print(message.message)
    msg = json.loads(json.dumps(message.message))
    if 'message' in msg:
        buzzer = msg['message']['buzzer']
        if buzzer == 'on':
            data['alarm'] = True
        elif buzzer == 'off':
            data['alarm'] = False

#Create a custom listener
class StatusListener(SubscribeListener):
    def status(self, pubnub, status):
        print(f'Status: {status.category.name}')

status_listener = StatusListener()
pubnub.add_listener(status_listener)

my_channel = "johns-pi-0"
subscription = pubnub.channel(my_channel).subscription()

subscription.on_message = lambda message: handle_message(message)

subscription.subscribe()

pir_pin = 23
buzzer_pin = 24

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(pir_pin, GPIO.IN)
GPIO.setup(buzzer_pin, GPIO.OUT)


def main():
    motion_detection()
    
    
def motion_detection():
    
    


