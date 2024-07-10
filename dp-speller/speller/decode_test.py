from speller.utils.logging import logger
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher
import time

def readFromStream():
    logger.setLevel(10)
    sw = StreamWatcher(name="decoder")

    sw.connect_to_stream()

    t_end = time.time() + 60

    while time.time() < t_end:
        data = sw.unfold_buffer()
        data_pred = data[data != 0]
        logger.debug("Data: " + str(data_pred))
        time.sleep(1)

