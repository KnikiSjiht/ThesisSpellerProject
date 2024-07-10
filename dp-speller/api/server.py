from fire import Fire

from speller.utils.logging import logger
from speller.speller import training, online
from speller.decode_test import readFromStream

from dareplane_utils.default_server.server import DefaultServer


def main(port: int = 8080, ip: str = "127.0.0.1", loglevel: int = 10):
    logger.setLevel(loglevel)

    pcommand_map = {
        "TRAINING": training,
        "ONLINE": online,
        "TEST STREAM": readFromStream
    }

    server = DefaultServer(
        port, ip=ip, pcommand_map=pcommand_map, name="speller_server"
    )

    # initialize to start the socket
    server.init_server()
    # start processing of the server
    server.start_listening()


    return 0


if __name__ == "__main__":
    Fire(main)
