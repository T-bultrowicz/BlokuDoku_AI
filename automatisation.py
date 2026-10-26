import network_lib as net
import blokudoku_lib as bl
from ppadb.client import Client as AdbClient
import warnings
import sys

def main():
    client = AdbClient(host="127.0.0.1", port=5037)
    print(client.version())

    devices = client.devices()
    my_phone = devices[0]

    try:
        result = my_phone.screencap()
        with open("screen.png", "wb") as f:
            f.write(result)
    except Exception as e:
        print(e)
        sys.exit(1)

    print("All went well!")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    main()