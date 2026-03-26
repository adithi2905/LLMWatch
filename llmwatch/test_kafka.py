import time
from bus import get_bus, TOPIC_REQUESTS

def test_publish_subscribe():
    bus = get_bus(reset=True)  
    
    received = []             

    def my_handler(event):
        received.append(event)

    bus.subscribe(TOPIC_REQUESTS, my_handler)
    bus.publish(TOPIC_REQUESTS, {"test": "hello"})

    bus.join(TOPIC_REQUESTS)

    assert len(received) == 1
    assert received[0] == {"test": "hello"}
    print("passed")

test_publish_subscribe()

def test_two_subscribers():
    bus = get_bus(reset=True)

    received_a = []
    received_b = []

    bus.subscribe(TOPIC_REQUESTS, lambda e: received_a.append(e))
    bus.subscribe(TOPIC_REQUESTS, lambda e: received_b.append(e))
    bus.publish(TOPIC_REQUESTS, "event1")

    bus.join(TOPIC_REQUESTS)

    assert received_a == ["event1"]
    assert received_b == ["event1"]
    print("passed")

test_two_subscribers()