from pythonosc.udp_client import SimpleUDPClient


class OSCClient:
    """Simple wrapper around python-osc's SimpleUDPClient to centralize OSC logic."""

    def __init__(self, ip: str = "127.0.0.1", port: int = 8000):
        self.ip = ip
        self.port = port
        self.client = SimpleUDPClient(ip, port)

    def send_people(self, count: int):
        self.client.send_message("/people", count)

    def send_energy(self, tid: int, level: str):
        # Level is expected to be a string like 'idle', 'light', etc.
        self.client.send_message("/energy", [tid, level])

    def send_raw(self, address: str, payload):
        self.client.send_message(address, payload)

    def send_group_avg(self, avg: float):
        self.client.send_message("/group/avg", float(avg) if avg is not None else [])

    def send_group_max(self, mx: float):
        self.client.send_message("/group/max", float(mx) if mx is not None else [])

    def send_group_std(self, std: float):
        self.client.send_message("/group/std", float(std) if std is not None else [])
